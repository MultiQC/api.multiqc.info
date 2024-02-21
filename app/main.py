from typing import List, Dict

import sys

from pathlib import Path

import http

import csv
import datetime
import logging
import os
from threading import Lock

import uvicorn
from enum import Enum
from os import getenv

import pandas as pd
import plotly.express as px
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi_utilities import repeat_every
from github import Github
from plotly.graph_objs import Layout
from sqlalchemy.exc import IntegrityError, ProgrammingError

from app import __version__, db, models
from app.db import engine
from app.downloads import daily

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Make sure logs are printed to stdout:
logger.addHandler(logging.StreamHandler(sys.stdout))

app = FastAPI(
    title="MultiQC API",
    description="MultiQC API service, providing run-time information about available updates.",
    version=__version__,
    license_info={
        "name": "Source code available under the MIT Licence",
        "url": "https://github.com/MultiQC/api.multiqc.info/blob/main/LICENSE",
    },
)


def get_latest_release() -> models.LatestRelease:
    """Get the latest release from the database."""
    token = getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable is not set")
    g = Github(token)
    repo = g.get_repo("ewels/MultiQC")
    release = repo.get_latest_release()
    return models.LatestRelease(
        version=release.tag_name,
        release_date=release.published_at.date(),
        html_url=release.html_url,
    )


app.latest_release = get_latest_release()


@app.on_event("startup")
async def startup():
    # Initialise the DB and tables on server startup
    db.create_db_and_tables()
    # Sync latest version tag using GitHub API
    app.latest_release = get_latest_release()


@app.on_event("startup")
@repeat_every(seconds=15 * 60)  # every 15 minutes
def update_version():
    """Sync latest version tag using GitHub API"""
    app.latest_release = get_latest_release()


# Fields to store per visit
visit_fieldnames = [
    "version_multiqc",
    "version_python",
    "operating_system",
    "installation_method",
    "ci_environment",
]

# Thread-safe in-memory buffer to accumulate recent visits before writing to the CSV file
visit_buffer: List[Dict[str, str]] = []
visit_buffer_lock = Lock()


@app.get("/version")  # log a visit
async def version(
    version_multiqc: str = "",
    version_python: str = "",
    operating_system: str = "",
    installation_method: str = "",
    ci_environment: str = "",
):
    """
    Endpoint for MultiQC that returns the latest release, and logs
    the visit along with basic user environment detail.
    """
    global visit_buffer
    with visit_buffer_lock:
        visit_buffer.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "version_multiqc": version_multiqc,
                "version_python": version_python,
                "operating_system": operating_system,
                "installation_method": installation_method,
                "ci_environment": ci_environment,
            }
        )
        logger.info(f"Logging visit, total visits: {len(visit_buffer)}")
    return models.VersionResponse(latest_release=app.latest_release)


# Path to a buffer CSV file to persist recent visits before dumping to the database
# In the same folder as this script
CSV_FILE_PATH = Path(os.getenv("TMPDIR", "/tmp")) / "visits.csv"


def _persist_visits() -> Response:
    """
    Write visits from memory to a CSV file
    """
    global visit_buffer_lock
    global visit_buffer
    with visit_buffer_lock:
        n_visits_file = 0
        if CSV_FILE_PATH.exists():
            with open(CSV_FILE_PATH, mode="r") as file:
                n_visits_file = sum(1 for _ in file)
        if not visit_buffer:
            return PlainTextResponse(content=f"No new visits to persist. File contains {n_visits_file} entries")
        logger.info(
            f"Appending {len(visit_buffer)} visits to {CSV_FILE_PATH} that currently contains {n_visits_file} visits"
        )
        with open(CSV_FILE_PATH, mode="a") as file:
            writer: csv.DictWriter = csv.DictWriter(file, fieldnames=["timestamp"] + visit_fieldnames)
            writer.writerows(visit_buffer)

        logger.info(f"Persisted {len(visit_buffer)} visits to CSV {CSV_FILE_PATH}")
        visit_buffer = []
        with open(CSV_FILE_PATH, mode="r") as file:
            n_visits_file = sum(1 for _ in file)
        logger.info(f"CSV {CSV_FILE_PATH} now contains {n_visits_file} visits")
        return PlainTextResponse(
            content=f"Successfully persisted visits to {CSV_FILE_PATH}, file now contains {n_visits_file} entries"
        )


@app.on_event("startup")
@repeat_every(
    seconds=10,
    wait_first=True,
    logger=logger,
)
async def persist_visits():
    return _persist_visits()


def _summarize_visits() -> Response:
    """
    Summarize visits from the CSV file and write to the database
    """
    global visit_buffer_lock
    with visit_buffer_lock:
        df = pd.read_csv(
            CSV_FILE_PATH,
            sep=",",
            names=["timestamp"] + visit_fieldnames,
            dtype="string",
            na_filter=False,  # prevent empty strings from converting to nan or <NA>
        )
        df["start"] = pd.to_datetime(df["timestamp"])
        df["end"] = df["start"] + pd.to_timedelta("1min")
        df["start"] = df["start"].dt.strftime("%Y-%m-%d %H:%M")
        df["end"] = df["end"].dt.strftime("%Y-%m-%d %H:%M")
        df["ci_environment"] = df["ci_environment"].apply(lambda val: strtobool(val) if val else False)
        df = df.drop(columns=["timestamp"])

        # Summarize visits per user per minute
        minute_summary = df.groupby(["start", "end"] + visit_fieldnames).size().reset_index(name="count")
        if len(minute_summary) == 0:
            return PlainTextResponse(content="No new visits to summarize")

        logger.info(f"Summarizing {len(df)} visits in {CSV_FILE_PATH} and writing {len(minute_summary)} rows to the DB")
        try:
            db.insert_usage_stats(minute_summary)
        except Exception as e:
            return PlainTextResponse(
                status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR, 
                content=f"Failed to write to the database: {e}",
            )
        else:
            logger.info(f"Successfully wrote {len(minute_summary)} rows to the DB")
            open(CSV_FILE_PATH, "w").close()  # Clear the CSV file on successful write
            return PlainTextResponse(
                content=f"Successfully summarized {len(df)} visits to {len(minute_summary)} per-minute entries",
            )


@app.on_event("startup")
@repeat_every(
    seconds=60 * 60 * 1,  # every hour
    wait_first=True,
    logger=logger,
)
async def summarize_visits():
    """
    Repeated task to summarize visits.
    """
    return _summarize_visits()


def _update_download_stats():
    """
    Update the daily download statistics in the database
    """
    try:
        existing_downloads = db.get_download_stats()
    except ProgrammingError:
        logger.error("The table does not exist, will create and populate with historical data")
        existing_downloads = []
    if len(existing_downloads) == 0:  # first time, populate historical data
        logger.info("Collecting historical data...")
        df = daily.collect_daily_download_stats()
        logger.info(f"Adding {len(df)} historical entries to the table...")
        db.insert_download_stats(df)
        logger.info(f"Successfully populated {len(df)} historical entries")
    else:  # recent days only
        n_days = 4
        logger.info(f"Updating data for the last {n_days} days...")
        df = daily.collect_daily_download_stats(days=n_days)
        logger.info(f"Adding {len(df)} recent entries to the table. Will update existing entries at the same date")
        db.insert_download_stats(df)
        logger.info(f"Successfully updated {len(df)} new daily download statistics")


@app.on_event("startup")
@repeat_every(
    seconds=60 * 60 * 24,  # every day
    wait_first=True,
    logger=logger,
)
async def update_downloads(background_tasks: BackgroundTasks):
    """
    Repeated task to update the daily download statistics.
    """
    background_tasks.add_task(_update_download_stats)


if os.getenv("ENVIRONMENT") == "DEV":
    # Add endpoints to trigger the cron jobs manually, available only when developing

    @app.post("/persist_visits")
    async def persist_visits_endpoint():
        try:
            return _persist_visits()
        except Exception as e:
            raise HTTPException(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))

    @app.post("/summarize_visits")
    async def summarize_visits_endpoint():
        try:
            return _summarize_visits()
        except Exception as e:
            raise HTTPException(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))

    @app.post("/update_download_stats")
    async def update_downloads_endpoint(background_tasks: BackgroundTasks):
        try:
            background_tasks.add_task(_update_download_stats)
            return PlainTextResponse(content="Queued updating the download stats in the DB")
        except Exception as e:
            raise HTTPException(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/version.php", response_class=PlainTextResponse)
async def version_legacy(background_tasks: BackgroundTasks, v: str | None = None):
    """
    Legacy endpoint that mimics response from the old PHP script.

    Accessed by MultiQC versions 1.14 and earlier,
    after being redirected to by https://multiqc.info/version.php
    """
    global visit_buffer
    with visit_buffer_lock:
        visit_buffer.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "version_multiqc": v,
            }
        )
    return app.latest_release.version


@app.get("/")
async def index(background_tasks: BackgroundTasks):
    """
    Root endpoint for the API.

    Returns a list of available endpoints.
    """
    return {
        "message": "Welcome to the MultiQC service API",
        "available_endpoints": [
            {"path": route.path, "name": route.name} for route in app.routes if route.name != "swagger_ui_redirect"
        ],
    }


class PlotlyImageFormats(str, Enum):
    """Available Plotly image export formats."""

    html = "html"
    json = "json"
    svg = "svg"
    pdf = "pdf"
    png = "png"


class PlotlyTemplates(str, Enum):
    """Available Plotly image export formats."""

    plotly = "plotly"
    plotly_white = "plotly_white"
    plotly_dark = "plotly_dark"
    ggplot2 = "ggplot2"
    seaborn = "seaborn"
    simple_white = "simple_white"
    none = "none"


@app.get("/plot_usage")
async def plot_usage(
    categories: models.UsageCategory | None = None,
    interval: models.IntervalTypes = models.IntervalTypes.D,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    limit: int | None = None,
    format: PlotlyImageFormats = PlotlyImageFormats.png,
    template: PlotlyTemplates = PlotlyTemplates.simple_white,
):
    """Plot usage metrics."""
    # Get visit data
    visit_stats = db.get_visit_stats(start=start, end=end, limit=limit)
    if not visit_stats:
        return PlainTextResponse(content="No usage data available to plot")
    df = pd.DataFrame.from_records([i.model_dump() for i in visit_stats])
    legend_title_text = models.usage_category_nicenames[categories] if categories else None

    # Simplify version numbers if requested
    if categories in (models.UsageCategory.version_multiqc_simple, models.UsageCategory.version_python_simple):
        categories = models.UsageCategory[categories.name.replace("_simple", "")]
        df[categories.name] = df[categories.name].str.replace(r"^v?(\d+\.\d+).+", lambda m: m.group(1), regex=True)

    # Plot histogram of df.count per interval from df.start
    fig = px.histogram(
        df,
        x="start",
        y="count",
        color=categories.name if categories else None,
        title="Usage per version per week",
    )
    fig.update_layout(
        legend_title_text=legend_title_text,
    )
    return plotly_image_response(plotly_to_image(fig, format, template), format)


def plotly_to_image(
    fig, format: PlotlyImageFormats = PlotlyImageFormats.png, template: PlotlyTemplates = PlotlyTemplates.simple_white
) -> str | bytes:
    """Return a Plotly plot in the specified format."""
    fig.update_layout(
        Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            template=template,
        )
    )
    if format == "html":
        return fig.to_html()
    if format == "json":
        return fig.to_json()
    elif format == "svg":
        return fig.to_image(format="svg")
    elif format == "pdf":
        return fig.to_image(format="pdf")
    elif format == "png":
        return fig.to_image(format="png", scale=4)
    else:
        raise ValueError(f"Invalid format: {format}")


def plotly_image_response(plot, format: PlotlyImageFormats = PlotlyImageFormats.png) -> Response | HTMLResponse:
    """Wrap a Plotly figure with the appropriate FastAPI response type."""
    if format == "html":
        return HTMLResponse(plot)
    if format == "json":
        # Don't use JSONResponse as it excepts a dict, and we have a JSON string
        return Response(content=plot, media_type="application/json")
    elif format == "svg":
        return Response(content=plot, media_type="image/svg+xml")
    elif format == "pdf":
        return Response(content=plot, media_type="application/pdf")
    elif format == "png":
        return Response(content=plot, media_type="image/png")
    return Response(content=plot)


def strtobool(val) -> bool:
    """
    Replaces deprecated https://docs.python.org/3.9/distutils/apiref.html#distutils.util.strtobool
    The deprecation recommendation is to re-implement the function https://peps.python.org/pep-0632/

    ------------------------------------------------------------

    Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val_str = str(val).lower()
    if val_str in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val_str in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

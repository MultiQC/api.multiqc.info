import logging

from typing import List, Dict, Optional

from pathlib import Path

import csv
import datetime
import os
from threading import Lock

import uvicorn
from enum import Enum
from os import getenv

import pandas as pd
import plotly.express as px
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi_utilities import repeat_every
from github import Github
from plotly.graph_objs import Layout
from sqlalchemy.exc import ProgrammingError

from app import __version__, db, models
from app.downloads import daily


logger = logging.getLogger("multiqc_api")

logger.info("Starting MultiQC API service")


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
    # Add timestamp to the uvicorn logger
    logger = logging.getLogger("uvicorn.access")
    for h in logger.handlers:
        h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

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
    "is_docker",
    "is_singularity",
    "is_conda",
    "is_ci",
]

# Thread-safe in-memory buffer to accumulate recent visits before writing to the CSV file
visit_buffer: List[Dict[str, str]] = []
visit_buffer_lock = Lock()


@app.get("/version")  # log a visit
async def version(
    background_tasks: BackgroundTasks,
    version_multiqc: str = "",
    version_python: str = "",
    operating_system: str = "",
    is_docker: str = "",
    is_singularity: str = "",
    is_conda: str = "",
    is_ci: str = "",
):
    """
    Endpoint for MultiQC that returns the latest release, and logs
    the visit along with basic user environment detail.
    """
    background_tasks.add_task(
        _log_visit,
        timestamp=datetime.datetime.now().isoformat(),
        version_multiqc=version_multiqc,
        version_python=version_python,
        operating_system=operating_system,
        is_docker=is_docker,
        is_singularity=is_singularity,
        is_conda=is_conda,
        is_ci=is_ci,
    )
    return models.VersionResponse(latest_release=app.latest_release)


@app.get("/health")
async def health():
    """
    Health check endpoint. Checks if the visits table contains records
    in the past 15 minutes.
    """
    try:
        visits = db.get_visit_stats(start=datetime.datetime.now() - datetime.timedelta(minutes=15))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    if not visits:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No recent visits found")
    return PlainTextResponse(content=str(len(visits)))


def _log_visit(
    timestamp: str,
    version_multiqc: str = "",
    version_python: str = "",
    operating_system: str = "",
    is_docker: str = "",
    is_singularity: str = "",
    is_conda: str = "",
    is_ci: str = "",
):
    global visit_buffer
    with visit_buffer_lock:
        visit_buffer.append(
            {
                "timestamp": timestamp,
                "version_multiqc": version_multiqc,
                "version_python": version_python,
                "operating_system": operating_system,
                "is_docker": is_docker,
                "is_singularity": is_singularity,
                "is_conda": is_conda,
                "is_ci": is_ci,
            }
        )
        logger.debug(f"Logged visit, total visits in buffer: {len(visit_buffer)}")


# Path to a buffer CSV file to persist recent visits before dumping to the database
# In the same folder as this script
CSV_FILE_PATH = Path(os.getenv("TMPDIR", "/tmp")) / "visits.csv"


def _persist_visits(verbose=False) -> Optional[Response]:
    """
    Write visits from memory to a CSV file.
    """
    global visit_buffer_lock
    global visit_buffer
    with visit_buffer_lock:
        if verbose:
            n_visits_file = 0
            if CSV_FILE_PATH.exists():
                with open(CSV_FILE_PATH, mode="r") as file:
                    n_visits_file = sum(1 for _ in file)
            if not visit_buffer:
                return PlainTextResponse(content=f"No new visits to persist. File contains {n_visits_file} entries")
            logger.debug(
                f"Appending {len(visit_buffer)} visits to {CSV_FILE_PATH} that currently contains {n_visits_file} visits"
            )

        with open(CSV_FILE_PATH, mode="a") as file:
            writer: csv.DictWriter = csv.DictWriter(file, fieldnames=["timestamp"] + visit_fieldnames)
            writer.writerows(visit_buffer)

        if verbose:
            with open(CSV_FILE_PATH, mode="r") as file:
                n_visits_file = sum(1 for _ in file)
            msg = (
                f"Successfully persisted {len(visit_buffer)} visits to {CSV_FILE_PATH}, "
                f"file now contains {n_visits_file} entries"
            )
            logger.debug(msg)

        visit_buffer = []  # Reset the buffer

        if verbose:
            return PlainTextResponse(content=msg)

    return None


@app.on_event("startup")
@repeat_every(
    seconds=10,
    wait_first=True,
    logger=logger,
)
async def persist_visits():
    return _persist_visits(verbose=True)


def _summarize_visits(interval="5min") -> Response:
    """
    Summarize visits from the CSV file and write to the database
    """
    _persist_visits(verbose=True)
    global visit_buffer_lock
    with visit_buffer_lock:
        if not CSV_FILE_PATH.exists():
            msg = f"File {CSV_FILE_PATH} doesn't yet exist, no visits to summarize"
            logger.info(msg)
            return PlainTextResponse(content=msg)

        df = pd.read_csv(
            CSV_FILE_PATH,
            sep=",",
            names=["timestamp"] + visit_fieldnames,
            dtype="string",
            na_filter=False,  # prevent empty strings from converting to nan or <NA>
        )
        df["start"] = pd.to_datetime(df["timestamp"])
        df["end"] = df["start"] + pd.to_timedelta(interval)
        df["start"] = df["start"].dt.strftime("%Y-%m-%d %H:%M")
        df["end"] = df["end"].dt.strftime("%Y-%m-%d %H:%M")

        def strtobool(val) -> bool:
            return str(val).lower() in ("y", "yes", "t", "true", "on", "1")

        df["is_docker"] = df["is_docker"].apply(strtobool)
        df["is_singularity"] = df["is_singularity"].apply(strtobool)
        df["is_conda"] = df["is_conda"].apply(strtobool)
        df["is_ci"] = df["is_ci"].apply(strtobool)
        df = df.drop(columns=["timestamp"])

        # Summarize visits per user per time interval
        interval_summary = df.groupby(["start", "end"] + visit_fieldnames).size().reset_index(name="count")
        if len(interval_summary) == 0:
            msg = "No new visits to summarize"
            logger.info(msg)
            return PlainTextResponse(content=msg)

        logger.info(
            f"Summarizing {len(df)} visits in {CSV_FILE_PATH} and writing {len(interval_summary)} rows to the DB"
        )
        try:
            db.insert_visit_stats(interval_summary)
        except Exception as e:
            msg = f"Failed to write to the database: {e}"
            logger.error(msg)
            return PlainTextResponse(
                status_code=status.INTERNAL_SERVER_ERROR,
                content=msg,
            )
        else:
            msg = f"Successfully summarized {len(df)} visits to {len(interval_summary)} per-interval entries"
            logger.info(msg)
            open(CSV_FILE_PATH, "w").close()  # Clear the CSV file on successful write
            return PlainTextResponse(content=msg)


@app.on_event("startup")
@repeat_every(
    seconds=10 * 60 * 1,  # every 10 minutes
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
        logger.info("Collecting historical downloads data...")
        df = daily.collect_daily_download_stats()
        logger.info(f"Adding {len(df)} historical entries to the table...")
        db.insert_download_stats(df)
        logger.info(f"Successfully populated {len(df)} historical entries")
    else:  # recent days only
        n_days = 4
        logger.info(f"Updating downloads data for the last {n_days} days...")
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
async def update_downloads():
    """
    Repeated task to update the daily download statistics.
    """
    _update_download_stats()


@app.post("/persist_visits")
async def persist_visits_endpoint():
    try:
        return _persist_visits(verbose=True)
    except Exception as e:
        msg = f"Failed to persist the visits: {e}"
        logger.error(msg)
        raise HTTPException(status_code=status.INTERNAL_SERVER_ERROR, detail=msg)


@app.post("/summarize_visits")
async def summarize_visits_endpoint():
    try:
        return _summarize_visits()
    except Exception as e:
        msg = f"Failed to summarize the visits: {e}"
        logger.error(msg)
        raise HTTPException(status_code=status.INTERNAL_SERVER_ERROR, detail=msg)


@app.post("/update_downloads")
async def update_downloads_endpoint(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(_update_download_stats)
        msg = "Queued updating the download stats in the DB"
        logger.info(msg)
        return PlainTextResponse(content=msg)
    except Exception as e:
        msg = f"Failed to update the download stats: {e}"
        raise HTTPException(status_code=status.INTERNAL_SERVER_ERROR, detail=msg)


if os.getenv("ENVIRONMENT") == "DEV":

    @app.post("/clean_visits_csv_file")
    async def clean_visits_csv_file():
        try:
            if CSV_FILE_PATH.exists():
                CSV_FILE_PATH.unlink()
                msg = f"Removed {CSV_FILE_PATH}"
                logger.info(msg)
                return PlainTextResponse(content=msg)
            else:
                msg = f"File {CSV_FILE_PATH} doesn't exist"
                logger.info(msg)
                return PlainTextResponse(content=msg)
        except Exception as e:
            msg = f"Failed to remove {CSV_FILE_PATH}: {e}"
            raise HTTPException(status_code=status.INTERNAL_SERVER_ERROR, detail=msg)


@app.get("/version.php", response_class=PlainTextResponse)
async def version_legacy(background_tasks: BackgroundTasks, v: str = ""):
    """
    Legacy endpoint that mimics response from the old PHP script.

    Accessed by MultiQC versions 1.14 and earlier,
    after being redirected to by https://multiqc.info/version.php
    """
    background_tasks.add_task(
        _log_visit,
        timestamp=datetime.datetime.now().isoformat(),
        version_multiqc=v,
        version_python="",
        operating_system="",
        is_docker="",
        is_singularity="",
        is_conda="",
        is_ci="",
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
    category: models.UsageCategory | None = None,
    interval: models.IntervalType = models.IntervalType.min,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    limit: int | None = None,
    format: PlotlyImageFormats = PlotlyImageFormats.png,
    template: PlotlyTemplates = PlotlyTemplates.simple_white,
):
    """Plot usage metrics."""
    visit_stats = db.get_visit_stats(start=start, end=end, limit=limit)
    if not visit_stats:
        msg = "No usage data available to plot"
        logger.info(msg)
        return PlainTextResponse(content=msg)
    df = pd.DataFrame.from_records([i.model_dump() for i in visit_stats])
    legend_title_text = models.usage_category_nice_names[category] if category else None

    # Simplify version numbers if requested
    if category in (models.UsageCategory.version_multiqc_simple, models.UsageCategory.version_python_simple):
        category = models.UsageCategory[category.name.replace("_simple", "")]
        df[category.name] = df[category.name].str.replace(r"^v?(\d+\.\d+).+", lambda m: m.group(1), regex=True)

    # Plot histogram of df.count per interval from df.start
    logger.debug(
        f"Plotting usage data, color by: {category.name if category else None}, start: {start}, "
        f"end: {end}, interval: {interval.value}, limit: {limit}, format: {format.name}"
    )
    fig = px.histogram(
        df,
        x=df["start"].dt.to_period(interval.name).astype(str),
        y="count",
        color=category.name if category else None,
        title=f"MultiQC usage per {interval.value}",
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

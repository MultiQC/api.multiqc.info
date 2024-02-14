import csv
import datetime
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import uvicorn
from enum import Enum
from os import getenv

import pandas as pd
import plotly.express as px
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi_utilities import repeat_every
from github import Github
from plotly.graph_objs import Layout

from app import __version__, db, models
from app.db import engine


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
    g = Github(getenv("GITHUB_TOKEN"))
    repo = g.get_repo("ewels/MultiQC")
    release = repo.get_latest_release()
    return models.LatestRelease(
        version=release.tag_name,
        release_date=release.published_at.date(),
        html_url=release.html_url,
    )


app.latest_release = get_latest_release()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise the DB and tables on server startup
    db.create_db_and_tables()
    # Sync latest version tag using GitHub API
    app.latest_release = get_latest_release()
    yield


@repeat_every(seconds=15 * 60)  # every 15 minutes
def update_version():
    """Sync latest version tag using GitHub API"""
    app.latest_release = get_latest_release()


# Fields to store per visit
fieldnames = [
    "version_multiqc",
    "version_python",
    "operating_system",
    "installation_method",
    "ci_environment",
]

# Thread-safe in-memory buffer to accumulate recent visits before writing to the CSV file
visit_data = []
visit_data_lock = Lock()


@app.get("/version")  # log a visit
async def version(
    version_multiqc: str | None = None,
    version_python: str | None = None,
    operating_system: str | None = None,
    installation_method: str | None = None,
    ci_environment: str | None = None,
):
    """
    Endpoint for MultiQC that returns the latest release, and logs
    the visit along with basic user environment detail.
    """
    with visit_data_lock:
        visit_data.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "version_multiqc": version_multiqc,
                "version_python": version_python,
                "operating_system": operating_system,
                "installation_method": installation_method,
                "ci_environment": ci_environment,
            }
        )
    return models.VersionResponse(latest_release=app.latest_release)


# Path to a buffer CSV file to persist recent visits before dumping to the database
# In the same folder as this script
CSV_FILE_PATH = Path(__file__).parent / "visits.csv"


@repeat_every(seconds=10)  # every 10 seconds
async def persist_visits():
    """Write in-memory visits to a CSV file"""
    global visit_data
    with visit_data_lock:
        if visit_data:
            logger.debug(f"Persisting {len(visit_data)} visits to CSV {CSV_FILE_PATH}")
            with open(CSV_FILE_PATH, mode="a") as file:
                writer = csv.DictWriter(file, fieldnames=["timestamp"] + fieldnames)
                for row in visit_data:
                    writer.writerow(row)
            visit_data = []


@repeat_every(seconds=60 * 60 * 1)  # every hour
async def summarize_visits():
    with visit_data_lock:
        df = pd.read_csv(CSV_FILE_PATH, sep=",", names=["timestamp"] + fieldnames)
        df["start"] = pd.to_datetime(df["timestamp"])
        df["end"] = df["start"] + pd.to_timedelta("1min")
        df["start"] = df["start"].dt.strftime("%Y-%m-%d %H:%M")
        df["end"] = df["end"].dt.strftime("%Y-%m-%d %H:%M")
        df = df.drop(columns=["timestamp"])
        # replace nan with "Unknown"
        df = df.fillna("Unknown")  # df.groupby will fail if there are NaNs
        # Summarize visits per user per minute
        minute_summary = df.groupby(["start", "end"] + fieldnames).size().reset_index(name="count")
        logger.debug(f"Summarizing {len(df)} visits in {CSV_FILE_PATH} and writing {len(minute_summary)} rows to DB")
        minute_summary.to_sql("visits", con=engine, if_exists="append", index=False)


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


@app.get("/downloads")
async def downloads():
    """
    MultiQC package downloads across difference sources, and when available,
    different versions.

    Fetch from the database which will be populated by the script in the
    https://github.com/MultiQC/usage repo, run on a chron job.
    """
    return {}


@app.get("/version.php", response_class=PlainTextResponse)
async def version_legacy(background_tasks: BackgroundTasks, v: str | None = None):
    """
    Legacy endpoint that mimics response from the old PHP script.

    Accessed by MultiQC versions 1.14 and earlier,
    after being redirected to by https://multiqc.info/version.php
    """
    with visit_data_lock:
        visit_data.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "version_multiqc": v,
            }
        )
    return app.latest_release.version


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
    visits = db.get_visits(start=start, end=end, limit=limit)
    if not visits:
        return Response(status_code=204)
    df = pd.DataFrame.from_records([i.dict() for i in visits])
    df.fillna("Unknown", inplace=True)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

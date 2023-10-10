import datetime
from enum import Enum
from os import getenv

import pandas as pd
import plotly.express as px
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from github import Github
from plotly.graph_objs import Layout

from . import __version__, db, models
from .downloads import download_stats


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


app = FastAPI(
    title="MultiQC API",
    description="MultiQC API service, providing run-time information about available updates.",
    version=__version__,
    license_info={
        "name": "Source code available under the MIT Licence",
        "url": "https://github.com/MultiQC/api.multiqc.info/blob/main/LICENSE",
    },
)


@app.on_event("startup")
def on_startup():
    """Initialise the DB and tables on server startup."""
    db.create_db_and_tables()
    app.latest_release = get_latest_release()


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
    """
    return download_stats()


@app.get("/version")
async def version(
    background_tasks: BackgroundTasks,
    version_multiqc: str | None = None,
    version_python: str | None = None,
    operating_system: str | None = None,
    installation_method: str | None = None,
    ci_environment: str | None = None,
):
    """Endpoint for MultiQC that returns the latest release, plus bonus info."""
    background_tasks.add_task(
        db.add_visit,
        db.Visit(
            version_multiqc=version_multiqc,
            version_python=version_python,
            operating_system=operating_system,
            installation_method=installation_method,
            ci_environment=ci_environment,
        ),
    )
    return models.VersionResponse(latest_release=app.latest_release)


@app.get("/version.php", response_class=PlainTextResponse)
async def version_legacy(background_tasks: BackgroundTasks, v: str | None = None):
    """
    Legacy endpoint that mimics response from the old PHP script.

    Accessed by MultiQC versions 1.14 and earlier,
    after being redirected to by https://multiqc.info/version.php
    """
    background_tasks.add_task(db.add_visit, db.Visit(version_multiqc=v))
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
async def usage_raw(
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
    df = pd.DataFrame.from_records([i.dict() for i in visits])
    df.fillna("Unknown", inplace=True)
    legend_title_text = models.usage_category_nicenames[categories] if categories else None

    # Simplify version numbers if requested
    if categories in (models.UsageCategory.version_multiqc_simple, models.UsageCategory.version_python_simple):
        categories = models.UsageCategory[categories.name.replace("_simple", "")]
        df[categories.name] = df[categories.name].str.replace(r"^v?(\d+\.\d+).+", lambda m: m.group(1), regex=True)

    # Plot
    fig = px.histogram(
        df,
        x=df["called_at"].dt.to_period(interval.name).astype("datetime64[M]"),
        color=categories,
        title="MultiQC usage",
    )
    fig.update_traces(xbins_size=models.interval_types_plotly[interval])
    fig.update_layout(legend_title_text=legend_title_text)
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

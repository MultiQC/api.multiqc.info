from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import PlainTextResponse

from . import db

LATEST_RELEASE = "v1.14"

app = FastAPI()


@app.on_event("startup")
def on_startup():
    """Initialise the DB and tables on server startup."""
    db.create_db_and_tables()


@app.get("/")
async def index(background_tasks: BackgroundTasks):
    """
    Root endpoint for the API.

    Returns a list of available endpoints.
    """
    return {
        "message": "Welcome to the MultiQC service API",
        "available_endpoints": [{"path": route.path, "name": route.name} for route in app.routes],
    }


@app.get("/version")
async def version(
    background_tasks: BackgroundTasks,
    version_multiqc: str | None = None,
    version_python: str | None = None,
    operating_system: str | None = None,
    installation_method: str | None = None,
):
    """
    Endpoint for MultiQC that returns the latest release.

    Also return additional info such as a broadcast message, if needed.

    Returns JSON response.
    """
    background_tasks.add_task(
        db.add_visit,
        db.Visit(
            version_multiqc=version_multiqc,
            version_python=version_python,
            operating_system=operating_system,
            installation_method=installation_method,
        ),
    )
    return {
        "latest_release": LATEST_RELEASE,
        "broadcast_message": "",
        "latest_release_date": "2023-01-23",
        "module_warnings": [],
    }


@app.get("/version.php", response_class=PlainTextResponse)
async def version_legacy(background_tasks: BackgroundTasks, v: str | None = None):
    """
    Legacy endpoint that mimics response from the old PHP script.

    Accessed by MultiQC versions 1.14 and earlier,
    after being redirected to by https://multiqc.info/version.php
    """
    background_tasks.add_task(db.add_visit, db.Visit(version_multiqc=v))
    return LATEST_RELEASE


@app.get("/usage")
async def usage_raw():
    """Return raw usage data from today."""
    return db.get_visits()

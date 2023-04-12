from datetime import datetime
from typing import Optional
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import PlainTextResponse
from sqlmodel import Field, Session, SQLModel, create_engine, select
from os import getenv

LATEST_RELEASE = "v1.14"

app = FastAPI()

sql_url = getenv("DATABASE_URL")
engine = create_engine(sql_url)


class Visit(SQLModel, table=True):
    """Table to record raw individual visits to the version endpoint."""

    id: Optional[int] = Field(default=None, primary_key=True)
    version_multiqc: Optional[str] = None
    version_python: Optional[str] = None
    operating_system: Optional[str] = None
    installation_method: Optional[str] = None
    called_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow, nullable=False
    )


def create_db_and_tables():
    """Create the database and tables if they don't exist."""
    SQLModel.metadata.create_all(engine)


def add_visit(visit: Visit):
    """Add a visit to the database."""
    with Session(engine) as session:
        session.add(visit)
        session.commit()


@app.on_event("startup")
def on_startup():
    """Initialise the DB and tables on server startup."""
    create_db_and_tables()


@app.get("/")
async def index(background_tasks: BackgroundTasks):
    """
    Root endpoint for the API.

    Returns a list of available endpoints.
    """
    return {
        "message": "Welcome to the MultiQC service API",
        "available_endpoints": [
            {"path": route.path, "name": route.name} for route in app.routes
        ],
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
        add_visit,
        Visit(
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
    background_tasks.add_task(add_visit, Visit(version_multiqc=v))
    return LATEST_RELEASE


@app.get("/usage")
async def usage_raw():
    """Return raw usage data from today."""
    visits = []
    with Session(engine) as session:
        statement = select(Visit)
        results = session.exec(statement)
        for visit in results:
            visits.append(visit)
    return visits

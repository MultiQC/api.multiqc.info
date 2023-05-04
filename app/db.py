"""Functions to interact with the database."""

from datetime import datetime
from os import getenv

from sqlmodel import create_engine, Field, select, Session, SQLModel

sql_url = getenv("DATABASE_URL")
engine = create_engine(sql_url)


class Visit(SQLModel, table=True):  # type: ignore # mypy doesn't like this, not sure why
    """Table to record raw individual visits to the version endpoint."""

    id: int | None = Field(default=None, primary_key=True)
    version_multiqc: str | None = None
    version_python: str | None = None
    operating_system: str | None = None
    installation_method: str | None = None
    ci_environment: str | None = None
    called_at: datetime | None = Field(default_factory=datetime.utcnow, nullable=False)


def create_db_and_tables() -> None:
    """Create the database and tables if they don't exist."""
    SQLModel.metadata.create_all(engine)


def add_visit(visit: Visit) -> None:
    """Add a visit to the database."""
    with Session(engine) as session:
        session.add(visit)
        session.commit()


def get_visits() -> list[Visit]:
    """Return list of raw visits from the DB."""
    visits = []
    with Session(engine) as session:
        statement = select(Visit)
        results = session.exec(statement)
        for visit in results:
            visits.append(visit)
    return visits

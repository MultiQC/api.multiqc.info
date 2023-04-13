from datetime import datetime
from typing import Optional
from sqlmodel import Field, Session, SQLModel, create_engine, select
from os import getenv

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


def get_visits():
    """Return list of raw visits from the DB."""
    visits = []
    with Session(engine) as session:
        statement = select(Visit)
        results = session.exec(statement)
        for visit in results:
            visits.append(visit)
    return visits

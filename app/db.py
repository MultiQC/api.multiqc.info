"""Functions to interact with the database."""

import datetime
from os import getenv

from sqlalchemy.sql import func
from sqlmodel import create_engine, Field, select, Session, SQLModel

from .models import StatTypes

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
    called_at: datetime.datetime | None = Field(default_factory=datetime.datetime.utcnow, nullable=False)


class DailyCounts(SQLModel, table=True):  # type: ignore # mypy doesn't like this, not sure why
    """Daily summary counts."""

    id: int | None = Field(default=None, primary_key=True)
    stat_type: StatTypes
    count: int
    date: datetime.date


def create_db_and_tables() -> None:
    """Create the database and tables if they don't exist."""
    SQLModel.metadata.create_all(engine)


def add_visit(visit: Visit) -> None:
    """Add a visit to the database."""
    with Session(engine) as session:
        session.add(visit)
        session.commit()


def update_daily_counts() -> None:
    """Generate daily counts from the Visits table."""
    # Should do SELECT COUNT(*) GROUP BY CAST(myDateTime AS DATE), stat_type
    # But can't figure out how to do that with SQLModel / SQLalchemy
    print("Let's go..")
    with Session(engine) as session:
        for StatType in StatTypes:
            statement = select(
                func.date(Visit.called_at).label("visit_date"),
                getattr(Visit, StatType),
                func.count().label("count"),
            ).group_by(
                func.date(Visit.called_at),
                getattr(Visit, StatType),
            )
            print(statement)
            return session.exec(statement).all()


def get_visits(
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    limit: int | None = None,
) -> list[Visit]:
    """Return list of raw visits from the DB."""
    with Session(engine) as session:
        statement = select(Visit.version_multiqc)
        if start:
            # Ignore type because Visit.called_at can be None for default value
            statement.where(Visit.called_at > start)  # type: ignore
        if end:
            statement.where(Visit.called_at < end)  # type: ignore
        if limit:
            statement.limit(limit)
        statement.order_by(Visit.called_at.desc())  # type: ignore
        return session.exec(statement).all()

"""Functions to interact with the database."""
import os

import datetime

from sqlmodel import create_engine, Field, select, Session, SQLModel

sql_url = os.getenv("DATABASE_URL")
assert sql_url is not None, sql_url
engine = create_engine(sql_url)


class Visits(SQLModel, table=True):  # type: ignore # mypy doesn't like this, not sure why
    """
    Table to record per-minute visit summaries. Start is a primary key,
    and start and end are both indexed.
    """

    start: datetime.datetime = Field(primary_key=True)
    end: datetime.datetime = Field(primary_key=True)
    count: int
    version_multiqc: str = Field(index=True)
    version_python: str = Field(default=None, index=True)
    operating_system: str = Field(default=None, index=True)
    installation_method: str = Field(default=None, index=True)
    ci_environment: str = Field(default=None, index=True)


def create_db_and_tables() -> None:
    """Create the database and tables if they don't exist."""
    SQLModel.metadata.create_all(engine)


def get_visits(
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    limit: int | None = None,
) -> list[Visits]:
    """Return list of per-minute visit summary from the DB."""
    with Session(engine) as session:
        statement = select(Visits)
        if start:
            # Ignore type because Visit.called_at can be None for default value
            statement.where(Visits.start >= start)  # type: ignore
        if end:
            statement.where(Visits.end <= end)  # type: ignore
        if limit:
            statement.limit(limit)
        statement.order_by(Visits.start.desc())  # type: ignore
        return session.exec(statement).all()

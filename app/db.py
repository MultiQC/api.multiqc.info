"""Functions to interact with the database."""
import os

import datetime

from sqlmodel import create_engine, Field, select, Session, SQLModel

sql_url = os.getenv("DATABASE_URL")
assert sql_url is not None, sql_url
engine = create_engine(sql_url)


class Visits(SQLModel, table=True):  # type: ignore # mypy doesn't like this, not sure why
    """
    Table to record per-minute visit summaries
    """

    id: int | None = Field(default=None, primary_key=True)
    start: datetime.datetime | None = None
    end: datetime.datetime | None = None
    count: int = 0
    version_multiqc: str | None = None
    version_python: str | None = None
    operating_system: str | None = None
    installation_method: str | None = None
    ci_environment: str | None = None


def create_db_and_tables() -> None:
    """Create the database and tables if they don't exist."""
    SQLModel.metadata.create_all(engine)


# def add_visit(visit: Visits) -> None:
#     """Add a visit to the database."""
#     with Session(engine) as session:
#         session.add(visit)
#         session.commit()


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

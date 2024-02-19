"""Functions to interact with the database."""
from typing import Optional

import logging
import os

import datetime
import sys

import pandas as pd

from sqlmodel import create_engine, Field, select, Session, SQLModel

sql_url = os.getenv("DATABASE_URL")
assert sql_url is not None, sql_url
engine = create_engine(sql_url)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Make sure logs are printed to stdout:
logger.addHandler(logging.StreamHandler(sys.stdout))


class VisitStats(SQLModel, table=True):  # type: ignore # mypy doesn't like this, not sure why
    """
    Table to record per-minute visit summaries.

    Start is a primary key, and start and end are both indexed.
    """

    start: datetime.datetime = Field(primary_key=True)
    end: datetime.datetime = Field(primary_key=True)
    count: int
    version_multiqc: Optional[str] = Field(index=True, default=None)
    version_python: Optional[str] = Field(default=None, index=True)
    operating_system: Optional[str] = Field(default=None, index=True)
    installation_method: Optional[str] = Field(default=None, index=True)
    ci_environment: Optional[bool] = Field(default=None, index=True)


class DownloadStatsDaily(SQLModel, table=True):
    """Daily download statistics"""

    date: datetime.datetime = Field(primary_key=True)
    pip_new: Optional[int] = None
    pip_total: Optional[int] = None
    bioconda_total: Optional[int] = None
    bioconda_new: Optional[int] = None
    biocontainers_quay_new: Optional[int] = None
    biocontainers_quay_total: Optional[int] = None
    prs_new: Optional[int] = None
    contributors_pr: Optional[int] = None
    contributors_new: Optional[int] = None
    prs_total: Optional[int] = None
    contributors_total: Optional[int] = None
    modules_new: Optional[int] = None
    modules_total: Optional[int] = None
    biocontainers_aws_total: Optional[int] = None
    dockerhub_total: Optional[int] = None
    clones_total: Optional[int] = None


def create_db_and_tables() -> None:
    """Create the database and tables if they don't exist."""
    SQLModel.metadata.create_all(engine)


def get_visit_stats(
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    limit: int | None = None,
) -> list[VisitStats]:
    """Return list of per-minute visit summary from the DB."""
    with Session(engine) as session:
        statement = select(VisitStats)
        if start:
            # Ignore type because Visit.called_at can be None for default value
            statement.where(VisitStats.start >= start)  # type: ignore
        if end:
            statement.where(VisitStats.end <= end)  # type: ignore
        if limit:
            statement.limit(limit)
        statement.order_by(VisitStats.start.desc())  # type: ignore
        return session.exec(statement).all()


def get_download_stats(
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    limit: int | None = None,
) -> list[DownloadStatsDaily]:
    """Return list of daily download statistics from the DB."""
    with Session(engine) as session:
        statement = select(DownloadStatsDaily)
        if start:
            statement.where(DownloadStatsDaily.date >= start)  # type: ignore
        if end:
            statement.where(DownloadStatsDaily.date <= end)  # type: ignore
        if limit:
            statement.limit(limit)
        statement.order_by(DownloadStatsDaily.date.desc())  # type: ignore
        return session.exec(statement).all()


def insert_download_stats(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df.index)  # adding a separate field date with a type datetime
    df = df[["date"] + [c for c in df.columns if c != "date"]]  # place date first
    with Session(engine) as session:
        for index, row in df.iterrows():
            row = row.where(pd.notna(row), None)
            existing_entry = session.exec(
                select(DownloadStatsDaily).where(DownloadStatsDaily.date == row["date"])
            ).first()
            if existing_entry:
                for key, value in row.items():
                    setattr(existing_entry, key, value)
            else:
                new_entry = DownloadStatsDaily(**row)
                session.add(new_entry)
        session.commit()
    return df

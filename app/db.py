"""Functions to interact with the database."""
from typing import Optional, Sequence

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


class VisitStats(SQLModel, table=True):
    """
    Table to record per-interval visit summaries.

    All keys describing the platform are primary, so we have separate a usage record
    coming from each source.
    """

    id: int | None = Field(default=None, primary_key=True)
    start: datetime.datetime
    end: datetime.datetime
    version_multiqc: str
    version_python: str
    operating_system: str
    is_docker: bool
    is_singularity: bool
    is_conda: bool
    is_ci: bool
    count: int


class DownloadStatsDaily(SQLModel, table=True):
    """
    Daily download statistics.
    """

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
) -> Sequence[VisitStats]:
    """Return list of per-interval visit summary from the DB."""
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
) -> Sequence[DownloadStatsDaily]:
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


def insert_usage_stats(visit_stats: pd.DataFrame):
    with Session(engine) as session:
        for index, row in visit_stats.iterrows():
            new_entry = VisitStats(**row)
            session.add(new_entry)
        session.commit()


def insert_download_stats(df: pd.DataFrame) -> pd.DataFrame:
    # df has "date" as an index. Re-adding it as a separate field with a type datetime
    df["date"] = pd.to_datetime(df.index)
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

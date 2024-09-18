"""Functions to interact with the database."""

import logging
from typing import Optional, Sequence

import os

import datetime

import pandas as pd

from sqlmodel import create_engine, Field, select, Session, SQLModel

from app.utils import strtobool

sql_url = os.getenv("DATABASE_URL")
assert sql_url is not None, sql_url
engine = create_engine(sql_url)

logger = logging.getLogger(__name__)


class VisitStats(SQLModel, table=True):
    """
    Table to record per-interval visit summaries.

    All keys describing the platform are primary, so we have separate a usage record
    coming from each source.
    """

    __tablename__ = "multiqc_api_visits_stats"

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


class User(SQLModel, table=True):
    __tablename__ = "multiqc_api_user"

    id: int = Field(primary_key=True)
    ip: str


class Run(SQLModel, table=True):
    __tablename__ = "multiqc_api_run"

    id: int = Field(primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="multiqc_api_user.id")
    timestamp: datetime.datetime
    duration_seconds: int
    version_multiqc: str
    version_python: str
    operating_system: str
    is_docker: bool
    is_singularity: bool
    is_conda: bool
    is_ci: bool


class Module(SQLModel, table=True):
    __tablename__ = "multiqc_api_module"

    id: int = Field(primary_key=True)
    name: str


class SuccessRunToModule(SQLModel, table=True):
    __tablename__ = "multiqc_api_success_run_to_module"

    id: int = Field(primary_key=True)
    run_id: int = Field(foreign_key="multiqc_api_run.id")
    module_id: int = Field(foreign_key="multiqc_api_module.id")


class FailureRunToModule(SQLModel, table=True):
    __tablename__ = "multiqc_api_failure_run_to_module"

    id: int = Field(primary_key=True)
    run_id: int = Field(foreign_key="multiqc_api_run.id")
    module_id: int = Field(foreign_key="multiqc_api_module.id")


def log_run(
    timestamp_str: str,
    duration: str,
    modules: str,
    modules_failed: str,
    version_multiqc: str,
    version_python: str,
    operating_system: str,
    is_docker: str,
    is_singularity: str,
    is_conda: str,
    is_ci: str,
    user_ip: Optional[str] = None,
) -> None:
    """Log a run of MultiQC."""
    with Session(engine) as session:
        try:
            duration_seconds = int(duration)
        except ValueError:
            logger.warning(f"Could not parse duration: {duration}")
            duration_seconds = -1

        timestamp = datetime.datetime.fromisoformat(timestamp_str)

        run = Run(
            timestamp=timestamp,
            duration_seconds=duration_seconds,
            version_multiqc=version_multiqc,
            version_python=version_python,
            operating_system=operating_system,
            is_docker=strtobool(is_docker),
            is_singularity=strtobool(is_singularity),
            is_conda=strtobool(is_conda),
            is_ci=strtobool(is_ci),
        )
        session.add(run)
        session.commit()
        for _mod in modules.split(","):
            module = session.exec(select(Module).where(Module.name == _mod)).first()
            if not module:
                module = Module(name=_mod)
                session.add(module)
                session.commit()
            session.add(SuccessRunToModule(run_id=run.id, module_id=module.id))
        for _mod in modules_failed.split(","):
            module = session.exec(select(Module).where(Module.name == _mod)).first()
            if not module:
                module = Module(name=_mod)
                session.add(module)
                session.commit()
            session.add(FailureRunToModule(run_id=run.id, module_id=module.id))
        session.add(run)

        if user_ip:
            user = User(ip=user_ip)
            session.add(user)
            session.commit()
            run.user_id = user.id

        session.commit()


class DownloadStats(SQLModel, table=True):
    """
    Daily download statistics.
    """

    __tablename__ = "multiqc_api_downloads_stats"

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
            statement = statement.where(VisitStats.start >= start)  # type: ignore
        if end:
            statement = statement.where(VisitStats.end <= end)  # type: ignore
        if limit:
            statement = statement.limit(limit)
        statement.order_by(VisitStats.start.desc())  # type: ignore
        return session.exec(statement).all()


def get_download_stats(
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    limit: int | None = None,
) -> Sequence[DownloadStats]:
    """Return list of daily download statistics from the DB."""
    with Session(engine) as session:
        statement = select(DownloadStats)
        if start:
            statement.where(DownloadStats.date >= start)  # type: ignore
        if end:
            statement.where(DownloadStats.date <= end)  # type: ignore
        if limit:
            statement.limit(limit)
        statement.order_by(DownloadStats.date.desc())  # type: ignore
        return session.exec(statement).all()


def insert_visit_stats(visit_stats: pd.DataFrame):
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
            existing_entry = session.exec(select(DownloadStats).where(DownloadStats.date == row["date"])).first()
            if existing_entry:
                for key, value in row.items():
                    setattr(existing_entry, key, value)
            else:
                new_entry = DownloadStats(**row)
                session.add(new_entry)
        session.commit()
    return df

"""Functions to interact with the database."""
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
    version_multiqc: str = Field(index=True)
    version_python: str = Field(default=None, index=True)
    operating_system: str = Field(default=None, index=True)
    installation_method: str = Field(default=None, index=True)
    ci_environment: str = Field(default=None, index=True)


class DownloadStatsDaily(SQLModel, table=True):
    """Daily download statistics"""

    date: datetime.datetime = Field(primary_key=True)
    pip_new: int = Field(default=None)
    pip_total: int = Field(default=None)
    bioconda_total: int = Field(default=None)
    bioconda_new: int = Field(default=None)
    biocontainers_quay_new: int = Field(default=None)
    biocontainers_quay_total: int = Field(default=None)
    prs_new: int = Field(default=None)
    contributors_pr: int = Field(default=None)
    contributors_new: int = Field(default=None)
    prs_total: int = Field(default=None)
    contributors_total: int = Field(default=None)
    modules_new: int = Field(default=None)
    modules_total: int = Field(default=None)
    biocontainers_aws_total: int = Field(default=None)
    dockerhub_total: int = Field(default=None)
    clones_total: int = Field(default=None)


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


def insert_download_stats(
    df: pd.DataFrame,
    db_table="downloadstatsdaily",
    days: int | None = None,
) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df.index)  # adding a separate field date with a type datetime
    df = df[["date"] + [c for c in df.columns if c != "date"]]  # place date first

    if days is None:
        # Initiating the database with historical data, making sure we are not
        # overriding th entire database.
        try:
            # Add a new date column separate from index in order to ensure the db uses Date type
            df.to_sql(db_table, engine, if_exists="fail", index=False, index_label="date")
        except ValueError as e:
            logger.error(
                f"Failed to save historical data to table '{db_table}', the table might already exist? "
                f"Clean manually if you want to replace the historical data: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to write historical downloads stats to table '{db_table}': {e}")
            raise
        # Adding date as a primary key. Not wrapping in try-except here because if the DB was
        # populated without problems but failed creating a primary key, something is wrong here
        # and needs to vbe cleaned up manually.
        with engine.connect() as c:
            cursor = c.connection.cursor()
            cursor.execute(f"ALTER TABLE {db_table} ADD PRIMARY KEY (date);")
        print(f"Wrote historical downloads stats to table '{db_table}'")
    else:
        logger.debug(
            f"Adding recent {len(df)} entries to the '{db_table}' table one by one, updating if"
            f"an entry at this date already exists"
        )
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
        logger.debug(f"Updated last {days} days to in daily downloads table '{db_table}'")
    return df

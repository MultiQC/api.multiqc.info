"""
Collect current and historic download counts per day for MultiQC:
    * PyPI
    * BioConda
    * BioContainers (quay.io and AWS mirror)
    * DockerHub
    * GitHub clones
    * GitHub pull requests
    * GitHub contributors

Write a combined CSV file that can be sent to a database.

Can be re-run regularly to update the data, so only new data will be added to 
an existing CSV file. Can be adjusted accordingly with a database.
"""


import json
import logging
import os
from pathlib import Path

import click
import numpy as np
import packaging.version
import pandas as pd
import pypistats
import requests
import sqlalchemy
from github import Github
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from the .env file
load_dotenv()


logging.basicConfig(level=logging.INFO)

out_path = Path(__file__).parent / "daily.csv"
cache_path = Path(__file__).parent / "cache"


@click.command()
@click.option("--db", "--populate-db", is_flag=True, help="Populate $DATABASE_URL in addition to the CSV.")
@click.option("--days", help="Only collect data for the past n days.")
def main(populate_db: bool = False, days: int | None = None):
    """
    Combine all the data sources into a single CSV file, with both new and daily stats for each day.
    """
    collect_daily_download_stats(populate_db=populate_db, db_table="downloads", days=days)


def collect_daily_download_stats(
    populate_db: bool = False,
    db_con: sqlalchemy.Engine | None = None,
    db_table="downloads",
    days: int | None = None,
) -> pd.DataFrame:
    df = _collect_daily_download_stats_to_df(days=days)
    if not populate_db:
        return df

    if db_con is None:
        if url := os.environ.get("DATABASE_URL"):
            db_con = create_engine(url)
        else:
            logging.error("No DATABASE_URL environment variable found, skipping database writing")
            return df

    df["date"] = pd.to_datetime(df.index)  # adding a separate field date with a type datetime
    df = df[["date"] + [c for c in df.columns if c != "date"]]  # place date first

    if days is None:
        # Initiating the database with historical data, making sure we are not
        # overriding th entire database.
        try:
            # Add a new date column separate from index in order to ensure the db uses Date type
            df.to_sql(db_table, db_con, if_exists="fail", index=False, index_label="date")
        except ValueError as e:
            logging.error(
                f"Failed to save historical data to table '{db_table}', the table might already exist? "
                f"Clean manually if you want to replace the historical data: {e}"
            )
            raise
        except Exception as e:
            logging.error(f"Failed to write historical downloads stats to table '{db_table}': {e}")
            raise
        # Adding date as a primary key. Not wrapping in try-except here because if the DB was
        # populated without problems but failed creating a primary key, something is wrong here
        # and needs to vbe cleaned up manually.
        with db_con.connect() as c:
            cursor = c.connection.cursor()
            cursor.execute(f"ALTER TABLE {db_table} ADD PRIMARY KEY (date);")
        print(f"Wrote historical downloads stats to table '{db_table}'")
    else:
        logging.debug(
            f"Adding recent {len(df)} entries to the '{db_table}' table one by one, updating if"
            f"an entry at this date already exists"
        )
        with db_con.connect() as con:
            cursor = con.connection.cursor()
            for index, row in df.iterrows():
                date = row["date"].strftime("%Y-%m-%d")
                values = ", ".join([f"'{date}'"] + [str(x) if not pd.isna(x) else "NULL" for x in row[1:]])
                query = f"""
    INSERT INTO {db_table} ({", ".join(df.columns)}) VALUES ({values})
    ON DUPLICATE KEY UPDATE 
    {',\n'.join([f'{col} = VALUES({col})' for col in df.columns if col != 'date'])};
                """
                try:
                    cursor.execute(query)
                except Exception as e:
                    logging.error(f"Failed to execute query: {query}: {e}")
                    cursor.close()
                    raise
            con.commit()
            cursor.close()
        logging.debug(f"Updated last {days} days to in daily downloads table '{db_table}'")
    return df


def _collect_daily_download_stats_to_df(days: int | None = None) -> pd.DataFrame:
    df = get_pypi(days=days)

    if (df_bioconda := get_bioconda()) is not None:
        df = df.merge(df_bioconda, on="date", how="outer").sort_values("date")

    df_quay = get_biocontainers_quay()
    df = df.merge(df_quay, on="date", how="outer").sort_values("date")

    df_prs = get_github_prs()
    df = df.merge(df_prs, on="date", how="outer").sort_values("date")

    df_modules = github_modules()
    df = df.merge(df_modules, on="date", how="outer").sort_values("date")

    today = pd.to_datetime("today").strftime("%Y-%m-%d")

    if aws_total := biocontainers_aws_total():
        df_aws = pd.DataFrame([[today, aws_total]], columns=["date", "biocontainers_aws_total"]).set_index("date")
        df = df.merge(df_aws, on="date", how="outer")

    if dh_count := dockerhub_total():
        df_dockerhub = pd.DataFrame([[today, dh_count]], columns=["date", "dockerhub_total"]).set_index("date")
        df = df.merge(df_dockerhub, on="date", how="outer")

    if clones := github_clones_total():
        df_clones = pd.DataFrame([[today, clones]], columns=["date", "clones_total"]).set_index("date")
        df = df.merge(df_clones, on="date", how="outer")

    # globally convert all NaN to NA - call "apply" to all columns and all rows
    # this is needed to convert NaN to NA in the "date" column
    keys = [k for k in df.keys() if k != "date"]
    # for k in keys:
    #     df[k] = df[k].apply(lambda x: int(float(x)) if not pd.isna(x) else pandas.NA)

    # Update the existing data frame.
    # Only add data where it's not already present.
    # For existing data, check that it's the same as the new data.
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        existing_df = existing_df.set_index("date")

        # Only update the entries in existing_df with non-NaN entries in df
        for k in keys:
            existing_df[k] = existing_df[k].fillna(df[k])
        df = existing_df

    for k in keys:
        df[k] = df[k].apply(lambda x: int(float(x)) if not pd.isna(x) else np.nan)
        df[k] = df[k].astype("Int64")  # Int64 is a nullable integer version of int64

    df.to_csv(out_path, index=True)
    print(f"Saved daily downloads stats to {out_path}")
    if days is not None:
        df = df.tail(days)
    return df


def get_pypi(days: int | None = None):
    """
    Combined recent and historic PyPI stats.
    """
    df = get_pipy_recent()
    if days is None or days > 150:
        df_historic = get_pypi_historic()
        df_historic.rename(columns={"downloads": "historic"}, inplace=True)
        # Outer merge recent and historic data
        df.rename(columns={"downloads": "recent"}, inplace=True)
        df = df_historic.merge(df, on="date", how="outer")

        def merge_entries(row):
            if pd.isna(row["recent"]):
                return row["historic"]
            return row["recent"]

        df["pip_new"] = df.apply(merge_entries, axis=1).fillna(0).astype(int)
        df.drop("recent", axis=1, inplace=True)
        df.drop("historic", axis=1, inplace=True)
        df["pip_total"] = df["pip_new"].cumsum()
    else:
        df["pip_new"] = df["downloads"]
        df.drop("downloads", axis=1, inplace=True)
        df["pip_total"] = df["pip_new"].cumsum()

    return df


def get_pypi_historic():
    """
    Get historic download counts from BigQuery. The data is available daily from 2016.
    It was pulled using the SQL query below and downloaded as a CSV file, that we load here.
    ```
    pandas.read_gbq('''
    SELECT
      FORMAT_TIMESTAMP('%Y-%m-%d', timestamp) AS download_day,
      COUNT(*) AS total_downloads
    FROM `bigquery-public-data.pypi.file_downloads`
    WHERE file.project = 'multiqc' AND timestamp >= '2014-01-01 00:00:00'
    GROUP BY download_day
    ORDER BY download_day;
    ''', project_id=os.environ["GCP_PROJECT"])
    ```
    """
    pypi_path = cache_path / "pypi-daily.csv"
    df = pd.read_csv(
        pypi_path,
        parse_dates=["download_day"],
        dtype={"total_downloads": int},
    ).rename(columns={"download_day": "date", "total_downloads": "downloads"})
    # Fill missing dates
    df = df.set_index("date").resample("D").sum().fillna(0).reset_index()
    df["date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    return df.set_index("date")


def get_pipy_recent(since=None):
    """
    Use pypistats to get PyPI download count for the past 5 months.
    """
    df = pypistats.overall("multiqc", mirrors=True, total="daily", format="pandas")
    df = df[df["category"] == "with_mirrors"]
    df.sort_values("date", inplace=True)
    df = df[["date", "downloads"]]
    return df.set_index("date")


def get_bioconda():
    """
    For the past 2 weeks, BioConda download stats.
    """
    url = "https://raw.githubusercontent.com/bioconda/bioconda-plots/main/plots/multiqc/versions.json"
    # [{"date":"2023-09-05","total":15949,"delta":18,"version":"1.10.1"}, ...

    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch data from {url}:", response.status_code, response.text)
        return None

    df = json.loads(response.text)
    df = pd.DataFrame(df)
    df["version"] = df["version"].apply(lambda x: str(packaging.version.parse(x)))
    df = df.rename(columns={"total": "bioconda_total", "delta": "bioconda_new"})
    df["bioconda_new"] = df["bioconda_new"].astype(int)
    df["bioconda_total"] = df["bioconda_total"].astype(int)
    df.sort_values("date", inplace=True)
    df = df.groupby("date").sum().drop("version", axis=1)
    return df


def biocontainers_aws_total():
    """
    Total number of BioContainers AWS mirror downloads.
    """
    url = "https://api.us-east-1.gallery.ecr.aws/getRepositoryCatalogData"
    headers = {"Content-Type": "application/json"}
    data = {"registryAliasName": "biocontainers", "repositoryName": "multiqc"}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        logging.error(f"Failed to fetch data from {url}:", response.status_code, response.text)
        return None
    try:
        count = response.json()["insightData"]["downloadCount"]
    except IndexError:
        logging.error("Cannot extract insightData/downloadCount from response:", response.text)
        return None
    return count


def get_biocontainers_quay(since=None):
    """
    For the last 3 months, total numbers of BioContainers Quay.io mirror downloads.
    """
    url = "https://quay.io/api/v1/repository/biocontainers/multiqc?includeStats=true&includeTags=false"
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch data from Quay.io, status code: {response.status_code}, url: {url}")
        return None
    data = json.loads(response.text)

    stats = data["stats"]
    df = pd.DataFrame(stats)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    # index by date
    df = df.set_index("date")

    path = cache_path / "biocontainers-quay-daily.csv"
    if path.exists():
        print(f"Previous Quay stats found at {path}, appending")
        existing_df = pd.read_csv(path)
        # index by date
        existing_df["date"] = pd.to_datetime(existing_df["date"])
        existing_df = existing_df.set_index("date")
        # append new data
        df = pd.concat([existing_df, df])
        # remove duplicates
        df = df[~df.index.duplicated(keep="last")]
        # sort by date
        df.sort_index(inplace=True)

    df.to_csv(path)
    df = pd.read_csv(path)
    print(f"Saved {path}")

    df.rename(columns={"count": "biocontainers_quay_new"}, inplace=True)
    df["biocontainers_quay_total"] = df["biocontainers_quay_new"].cumsum()
    return df


def dockerhub_total() -> int | None:
    """
    Total number of DockerHub downloads.
    """
    url = "https://hub.docker.com/v2/repositories/ewels/multiqc/"
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch data from DockerHub, status code: {response.status_code}, url: {url}")
        return None
    data = json.loads(response.text)
    return data.get("pull_count", 0)


def github_clones_total():
    """
    Total number of GitHub clones.
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    url = "https://api.github.com/repos/ewels/MultiQC/traffic/clones"
    headers = {"Authorization": f"token {github_token}"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logging.error(f"Failed to fetch data from GitHub, status code: {response.status_code}, url: {url}")
        return None

    github_data = json.loads(response.text)
    return github_data.get("count", 0)


def get_github_prs(days: int | None = None):
    """
    Daily and total PRs and contributors.
    """
    path = cache_path / "github-pull-requests.csv"
    if path.exists():
        print(f"{path} exists, loading")
        df = pd.read_csv(path)
    else:
        g = Github(os.environ["GITHUB_TOKEN"])
        repo = g.get_repo("ewels/MultiQC")
        entries = []
        for pr in repo.get_pulls(state="all", sort="created", direction="asc"):
            author = pr.user.login
            entry = {
                "date": pr.created_at,
                "author": author,
                "prs": 1,
            }
            entries.append(entry)

        df = pd.DataFrame(entries)
        df.to_csv(path, index=False)
        print(f"Saved {path}")

    df["date"] = pd.to_datetime(df["date"])
    df["author"] = df["author"].apply(lambda x: [x])

    # Fill in missing dates with 0
    df = df.set_index("date").resample("D").sum().fillna(0)
    # Replace author=0 with []
    df["author"] = df["author"].apply(lambda x: x if x != 0 else [])

    # Collect the number of new contributors per day
    contributors = set()
    entries = []
    for date, row in df.iterrows():
        authors = set(row.author)
        d = {
            "date": date.date(),
            "prs_new": row.prs,
            "contributors_pr": len(authors),
            "contributors_new": len(authors - contributors),
        }
        contributors |= authors
        entries.append(d)

    df = pd.DataFrame(entries)
    df = df.groupby("date").sum().reset_index()
    df["prs_new"] = df["prs_new"].astype(int)
    df["prs_total"] = df["prs_new"].cumsum()
    df["contributors_new"] = df["contributors_new"].astype(int)
    df["contributors_total"] = df["contributors_new"].cumsum().astype(int)

    df["date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    return df.set_index("date")


def github_modules(since=None):
    """
    Daily and total new MultiQC modules.
    """
    path = cache_path / "new-modules.csv"
    if path.exists():
        print(f"{path} exists, loading")
        df = pd.read_csv(path)
    else:
        entries = []
        REPO_ROOT = Path("..").resolve()
        for module_path in (REPO_ROOT / "multiqc/modules").iterdir():
            if not module_path.is_dir():
                continue
            init_path = module_path / "__init__.py"
            if not init_path.exists():
                continue
            cmd = f"git -C {REPO_ROOT} log --follow --format='%ai' -- {module_path} | tail -1"
            date = os.popen(cmd).read().strip()
            entries.append({"date": date, "name": module_path.name})

        df = pd.DataFrame(entries)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df.sort_values("date", inplace=True)
        df.to_csv(path, index=False)
        print(f"Saved {path}")

    # Take the day part only, and sum up multiple entries per one day
    try:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    except ValueError:
        print("Failed to convert date to datetime")
        print(df)
        raise
    df["modules"] = 1
    df["name"] = df["name"].apply(lambda x: [x])
    df = df.groupby("date").sum().reset_index()

    # Remove the name column
    df.drop("name", axis=1, inplace=True)

    # Fill in missing dates with 0
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").resample("D").sum().fillna(0).reset_index()
    df["date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))

    df = df.rename(columns={"modules": "modules_new"})
    df["modules_total"] = df["modules_new"].cumsum()
    return df.set_index("date")


if __name__ == "__main__":
    main()

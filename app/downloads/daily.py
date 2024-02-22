"""
A script to collect current and historic MultiQC download counts, per day:
    * PyPI
    * BioConda
    * BioContainers (quay.io and AWS mirror)
    * DockerHub
    * GitHub clones
    * GitHub pull requests
    * GitHub contributors

Stores data in a CSV file that can be sent to a database.

Can be re-run regularly to update the data, so only new data will be added to 
an existing CSV file.
"""
import logging

import json
import os
from pathlib import Path

import click
import numpy as np
import packaging.version
import pandas as pd
import pypistats
import requests
from github import Github
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

logger = logging.getLogger(__name__)

PYPI_HISTORIC_PATH = Path(__file__).parent / "sources" / "pypi-historic.csv"


@click.command()
@click.option("--days", type=int, help="Only collect data for the past N days.")
def main(days: int | None):
    """
    Combine all the data sources into a single CSV file, with both new and daily stats for each day.
    """
    collect_daily_download_stats(days=days)


def collect_daily_download_stats(cache_dir: Path | None = None, days: int | None = None) -> pd.DataFrame:
    cache_dir = cache_dir or Path(__file__).parent / "cache"

    df = _collect_daily_download_stats(cache_dir, days=days)

    # Update the existing CSV.
    # Only add data where it's not already present.
    # For existing data, check that it's the same as the new data.
    keys = [k for k in df.keys() if k != "date"]
    cache_path = cache_dir / "daily.csv"
    if cache_path.exists():
        logger.info(f"Loading existing daily downloads stats from cache location {cache_path}")
        existing_df = pd.read_csv(
            cache_path,
            dtype={k: "date" if k == "date" else "Int64" for k in keys},  # Int64 is a nullable integer version of int64
        ).set_index("date")
        # Fixing <NA> as nan and converting values to int
        for k in keys:
            existing_df[k] = existing_df[k].apply(lambda x: int(float(x)) if not pd.isna(x) else np.nan)
            existing_df[k] = existing_df[k].astype("Int64")  # Int64 is a nullable integer version of int64
        full_df = existing_df.combine_first(df)
    else:
        full_df = df
    logger.info(f"Saving daily downloads stats to {cache_path}")
    full_df.to_csv(cache_path, index=True)
    return df


def _collect_daily_download_stats(cache_dir: Path, days: int | None = None) -> pd.DataFrame:
    logger.info("Collecting PyPI stats...")
    df = get_pypi(days=days)

    logger.info("Collecting BioConda stats...")
    if (df_bioconda := get_bioconda(days=days)) is not None:
        df = df.merge(df_bioconda, on="date", how="outer").sort_values("date")

    logger.info("Collecting BioContainers (Quay mirror) stats...")
    df_quay = get_biocontainers_quay(cache_dir, days=days)
    df = df.merge(df_quay, on="date", how="outer").sort_values("date")

    logger.info("Collecting GitHub PRs...")
    df_prs = get_github_prs(days=days)
    df = df.merge(df_prs, on="date", how="outer").sort_values("date")

    logger.info("Collecting GitHub modules...")
    df_modules = github_modules(cache_dir, days=days)
    df = df.merge(df_modules, on="date", how="outer").sort_values("date")

    today = pd.to_datetime("today").strftime("%Y-%m-%d")

    logger.info("Getting BioContainers (AWS mirror) stats...")
    if aws_total := biocontainers_aws_total():
        df_aws = pd.DataFrame([[today, aws_total]], columns=["date", "biocontainers_aws_total"]).set_index("date")
        df = df.merge(df_aws, on="date", how="outer")

    logger.info("Getting DockerHub stats...")
    if dh_count := dockerhub_total():
        df_dockerhub = pd.DataFrame([[today, dh_count]], columns=["date", "dockerhub_total"]).set_index("date")
        df = df.merge(df_dockerhub, on="date", how="outer")

    logger.info("Getting GitHub clones...")
    if clones := github_clones_total():
        df_clones = pd.DataFrame([[today, clones]], columns=["date", "clones_total"]).set_index("date")
        df = df.merge(df_clones, on="date", how="outer")

    if days is not None:
        df = df.tail(days)

    keys = [k for k in df.keys() if k != "date"]
    for k in keys:
        df[k] = df[k].apply(lambda x: int(float(x)) if not pd.isna(x) else np.nan)
        df[k] = df[k].astype("Int64")  # Int64 is a nullable integer version of int64

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

    if days is not None:
        df = df.tail(days)
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
    logger.info(f"Loading historic PyPI stats from {PYPI_HISTORIC_PATH}")
    df = pd.read_csv(
        PYPI_HISTORIC_PATH,
        parse_dates=["download_day"],
        dtype={"total_downloads": int},
    ).rename(columns={"download_day": "date", "total_downloads": "downloads"})
    # Fill missing dates
    df = df.set_index("date").resample("D").sum().fillna(0).reset_index()
    df["date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    return df.set_index("date")


def get_pipy_recent(days: int | None = None):
    """
    Use pypistats to get PyPI download count for the past 5 months.
    """
    df = pypistats.overall("multiqc", mirrors=True, total="daily", format="pandas")
    df = df[df["category"] == "with_mirrors"]
    df.sort_values("date", inplace=True)
    df = df[["date", "downloads"]]
    if days is not None:
        df = df.tail(days)
    return df.set_index("date")


def get_bioconda(days: int | None = None):
    """
    For the past 2 weeks, BioConda download stats.
    """
    url = "https://raw.githubusercontent.com/bioconda/bioconda-plots/main/plots/multiqc/versions.json"
    # [{"date":"2023-09-05","total":15949,"delta":18,"version":"1.10.1"}, ...

    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Failed to fetch data from {url}:", response.status_code, response.text)
        return None

    df = json.loads(response.text)
    df = pd.DataFrame(df)
    df["version"] = df["version"].apply(lambda x: str(packaging.version.parse(x)))
    df = df.rename(columns={"total": "bioconda_total", "delta": "bioconda_new"})
    df["bioconda_new"] = df["bioconda_new"].astype(int)
    df["bioconda_total"] = df["bioconda_total"].astype(int)
    df.sort_values("date", inplace=True)
    df = df.groupby("date").sum().drop("version", axis=1)
    if days is not None:
        df = df.tail(days)
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
        logger.error(f"Failed to fetch data from {url}:", response.status_code, response.text)
        return None
    try:
        count = response.json()["insightData"]["downloadCount"]
    except IndexError:
        logger.error("Cannot extract insightData/downloadCount from response:", response.text)
        return None
    return count


def get_biocontainers_quay(cache_dir: Path, days: int | None = None):
    """
    For the last 3 months, total numbers of BioContainers Quay.io mirror downloads.
    """
    url = "https://quay.io/api/v1/repository/biocontainers/multiqc?includeStats=true&includeTags=false"
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Failed to fetch data from Quay.io, status code: {response.status_code}, url: {url}")
        return None
    data = json.loads(response.text)

    stats = data["stats"]
    df = pd.DataFrame(stats)
    df.sort_values("date", inplace=True)
    df = df.set_index("date")
    if days is None or days > 90:
        path = cache_dir / "biocontainers-quay-historic.csv"
        if path.exists():
            print(f"Previous Quay stats found at {path}, appending")
            existing_df = pd.read_csv(path, index_col="date", dtype={"count": "Int64"})
            # append new data
            df = pd.concat([existing_df, df])
            # remove duplicates
            df = df[~df.index.duplicated(keep="last")]
            # sort by date
            df.sort_index(inplace=True)
        df.to_csv(path)
        df = pd.read_csv(path)
        df = df.set_index("date")
        print(f"Saved {path}")

    df.rename(columns={"count": "biocontainers_quay_new"}, inplace=True)
    df["biocontainers_quay_total"] = df["biocontainers_quay_new"].cumsum()
    if days is not None:
        df = df.tail(days)
    return df


def dockerhub_total() -> int | None:
    """
    Total number of DockerHub downloads.
    """
    url = "https://hub.docker.com/v2/repositories/ewels/multiqc/"
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Failed to fetch data from DockerHub, status code: {response.status_code}, url: {url}")
        return None
    data = json.loads(response.text)
    return data.get("pull_count", 0)


def github_clones_total():
    """
    Total number of GitHub clones.
    """
    github_token = os.environ["GITHUB_TOKEN"]
    url = "https://api.github.com/repos/MultiQC/MultiQC/traffic/clones"
    headers = {"Authorization": f"token {github_token}"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Failed to fetch data from GitHub, status code: {response.status_code}, url: {url}")
        return None

    github_data = json.loads(response.text)
    return github_data.get("count", 0)


def get_github_prs(days: int | None = None):
    """
    Daily and total PRs and contributors.
    """
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("MultiQC/MultiQC")
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
    if days is not None:
        df = df.tail(days)
    return df.set_index("date")


def github_modules(cache_dir: Path, days: int | None = None):
    """
    Daily and total new MultiQC modules.
    """
    entries = []

    # Clone MultiQC/MultiQC repo and get the date of the last commit for each module
    from git import Repo

    repo_url = "https://github.com/MultiQC/MultiQC.git"
    clone_path = cache_dir / "MultiQC"
    if not clone_path.exists():
        Repo.clone_from(repo_url, clone_path)
        logger.debug(f"{repo_url} cloned at {clone_path}")
    else:
        Repo(clone_path).remotes.origin.pull("main")
        logger.debug(f"Pulled main in {clone_path}")

    logger.debug(f"Collecting module commit dates from {clone_path}")
    for module_path in (clone_path / "multiqc/modules").iterdir():
        if not module_path.is_dir():
            continue
        init_path = module_path / "__init__.py"
        if not init_path.exists():
            continue
        cmd = f"git -C {clone_path} log --follow --format='%ai' -- {module_path} | tail -1"
        date = os.popen(cmd).read().strip()
        entries.append({"date": date, "name": module_path.name})

    df = pd.DataFrame(entries)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.sort_values("date", inplace=True)

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

    if days is not None:
        df = df.tail(days)
    return df.set_index("date")


if __name__ == "__main__":
    main()

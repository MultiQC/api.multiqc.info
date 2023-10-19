#!/usr/bin/env python

"""
Fetch the MultiQC package download numbers from various sources.

Assumes GITHUB_TOKEN is set in the environment.
"""

import json
import logging
import os
from collections import Counter

import packaging.version
import requests

logging.basicConfig(level=logging.INFO)

PayloadType = dict[str, int | None | dict[str, int]]


def download_stats() -> PayloadType:
    """
    Fetch the MultiQC package download numbers from various sources and return them
    as a JSON string.
    """
    stats: PayloadType = dict()
    stats |= pypi_stats()
    stats |= bioconda_stats()
    stats |= github_clones_stats()
    stats |= github_releases_stats()
    stats |= dockerhub_stats()
    stats |= biocontainers_stats()
    return stats


def pypi_stats() -> PayloadType:
    """
    Download count from [PyPI](https://pypi.org/project/multiqc), also split by version.
    """
    pepy_url = "https://api.pepy.tech/api/v2/projects/multiqc"

    total_downloads = None
    downloads_counter: Counter[str] = Counter()

    # Fetch download count from pepy.tech
    response = requests.get(pepy_url)
    if response.status_code == 200:
        data = json.loads(response.text)
        total_downloads = data["total_downloads"]
        for date, date_downloads_by_version in data["downloads"].items():
            for version, downloads in date_downloads_by_version.items():
                downloads_counter[version] += downloads
    else:
        logging.error("Failed to fetch data from pepy.tech")

    downloads_by_version: dict[str, int] = {str(packaging.version.parse(k)): v for k, v in downloads_counter.items()}
    return {
        "pypi": total_downloads,
        "pypi_by_version": downloads_by_version,
    }


def bioconda_stats() -> PayloadType:
    """
    Download count from [BioConda](https://bioconda.github.io/recipes/multiqc),
    also split by version.
    """
    url = "https://raw.githubusercontent.com/bioconda/bioconda-plots/main/plots/multiqc/versions.json"
    # [{"date":"2023-09-05","total":15949,"delta":18,"version":"1.10.1"}, ...

    downloads_counter: Counter[str] = Counter()

    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        for version in data:
            downloads_counter[version["version"]] += version["total"]
    else:
        logging.error("Failed to fetch data from bioconda")

    downloads_by_version: dict[str, int] = {str(packaging.version.parse(k)): v for k, v in downloads_counter.items()}
    return {
        "bioconda": sum(downloads_by_version.values()),
        "bioconda_by_version": downloads_by_version,
    }


def github_clones_stats() -> dict[str, int | None]:
    """
    Number of [GitHub clones](https://github.com/ewels/MultiQC/graphs/traffic).
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    url = "https://api.github.com/repos/ewels/MultiQC/traffic/clones"
    headers = {"Authorization": f"token {github_token}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        github_data = json.loads(response.text)
        clone_count = github_data.get("count", 0)
    else:
        logging.error(f"Failed to fetch data from GitHub, status code: {response.status_code}, url: {url}")
        clone_count = None

    return {"github_clones": clone_count}


def github_releases_stats() -> PayloadType:
    """
    Number of [GitHub releases](https://github.com/ewels/MultiQC/releases). Currently
    not working properly, returning information about ancient releases only.
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    url = "https://api.github.com/repos/ewels/MultiQC/releases"
    headers = {"Authorization": f"token {github_token}"}

    downloads_counter: Counter[str] = Counter()

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = json.loads(response.text)
        for release in data:
            version = packaging.version.parse(release["tag_name"])
            for asset in release["assets"]:
                downloads_counter[str(version)] += asset["download_count"]
    else:
        logging.error(f"Failed to fetch release data from GitHub, status code: {response.status_code}, url: {url}")

    return {
        "github_releases": sum(downloads_counter.values()),
        "github_releases_by_version": dict(downloads_counter),
    }


def _fetch_dockerhub_count(repo: str) -> int | None:
    url = f"https://hub.docker.com/v2/repositories/{repo}"
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch data from DockerHub, status code: {response.status_code}, url: {url}")
        return None
    data = json.loads(response.text)
    return data.get("pull_count", 0)


def _fetch_quay_count(repo: str) -> int | None:
    url = f"https://quay.io/api/v1/repository/{repo}".strip("/")
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch data from Quay.io, status code: {response.status_code}, url: {url}")
        return None
    data = json.loads(response.text)
    return data.get("pull_count", 0)


def dockerhub_stats() -> dict[str, int | None]:
    """
    Number of [DockerHub](https://hub.docker.com/r/ewels/multiqc) pulls.
    """
    return {"dockerhub": _fetch_dockerhub_count("ewels/multiqc/")}


def _biocontainers_aws() -> int | None:
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
        logging.error(f"Cannot extract insightData/downloadCount from response:", response.text)
        return None
    return count


def biocontainers_stats() -> dict[str, int | None]:
    """
    Number of BioContainers pulls, currently only supported for the images hosted on
    the [AWS mirror](https://api.us-east-1.gallery.ecr.aws/getRepositoryCatalogData)
    """
    out = {
        "biocontainers_dockerhub": _fetch_dockerhub_count("biocontainers/multiqc/"),
        "biocontainers_quay": _fetch_quay_count("biocontainers/multiqc/"),
        "biocontainers_aws": _biocontainers_aws(),
    }
    out["biocontainers"] = sum(v for v in out.values() if v is not None)
    return out


if __name__ == "__main__":
    d = download_stats()
    # Print as a properly formatted JSON
    print(json.dumps(d, indent=2, sort_keys=True))

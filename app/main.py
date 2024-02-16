from os import getenv

from elasticapm.contrib.starlette import ElasticAPM, make_apm_client
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import PlainTextResponse
from github import Github

from . import __version__, models


def get_latest_release() -> models.LatestRelease:
    """Get the latest release from the database."""
    g = Github(getenv("GITHUB_TOKEN"))
    repo = g.get_repo("ewels/MultiQC")
    release = repo.get_latest_release()
    return models.LatestRelease(
        version=release.tag_name,
        release_date=release.published_at.date(),
        html_url=release.html_url,
    )


app = FastAPI(
    title="MultiQC API",
    description="MultiQC API service, providing run-time information about available updates.",
    version=__version__,
    license_info={
        "name": "Source code available under the MIT Licence",
        "url": "https://github.com/MultiQC/api.multiqc.info/blob/main/LICENSE",
    },
)
apm = make_apm_client()
app.add_middleware(ElasticAPM, client=apm)


@app.on_event("startup")
def on_startup():
    """Get the latest release of MultiQC server startup."""
    app.latest_release = get_latest_release()


@app.get("/")
async def index(background_tasks: BackgroundTasks):
    """
    Root endpoint for the API.

    Returns a list of available endpoints.
    """
    return {
        "message": "Welcome to the MultiQC service API",
        "available_endpoints": [
            {"path": route.path, "name": route.name} for route in app.routes if route.name != "swagger_ui_redirect"
        ],
    }


@app.get("/version")
async def version(
    background_tasks: BackgroundTasks,
    version_multiqc: str | None = None,
    version_python: str | None = None,
    operating_system: str | None = None,
    installation_method: str | None = None,
    ci_environment: str | None = None,
):
    """Endpoint for MultiQC that returns the latest release, plus bonus info."""
    extra = {
        "version_multiqc": version_multiqc,
        "version_python": version_python,
        "operating_system": operating_system,
        "installation_method": installation_method,
        "ci_environment": ci_environment,
    }
    return models.VersionResponse(latest_release=app.latest_release)


@app.get("/version.php", response_class=PlainTextResponse)
async def version_legacy(background_tasks: BackgroundTasks, v: str | None = None):
    """
    Legacy endpoint that mimics response from the old PHP script.

    Accessed by MultiQC versions 1.14 and earlier,
    after being redirected to by https://multiqc.info/version.php
    """
    extra = {
        "version_multiqc": v,
        "version_python": None,
        "operating_system": None,
        "installation_method": None,
        "ci_environment": None,
    }
    return app.latest_release.version

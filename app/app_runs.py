import logging
from contextlib import asynccontextmanager

from typing import cast

import datetime

import uvicorn

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRoute

from app import __version__, db

logger = logging.getLogger("multiqc_api")

logger.info("Starting MultiQC API run logger service")

# Add timestamp to the uvicorn logger
for h in logging.getLogger("uvicorn.access").handlers:
    h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="MultiQC API",
    description="MultiQC API service, providing run-time information about available "
    ""
    ""
    ""
    ""
    ""
    ""
    ""
    ""
    "updates.",
    version=__version__,
    license_info={
        "name": "Source code available under the MIT Licence",
        "url": "https://github.com/MultiQC/api.multiqc.info/blob/main/LICENSE",
    },
    lifespan=lifespan,
)

db.create_db_and_tables()


@app.get("/run")
async def run(
    background_tasks: BackgroundTasks,
    duration: str,
    modules: str,
    modules_failed: str,
    version_multiqc: str,
    version_python: str,
    operating_system: str = "",
    is_docker: str = "",
    is_singularity: str = "",
    is_conda: str = "",
    is_ci: str = "",
):
    """
    Log the modules run by MultiQC
    """
    timestamp = datetime.datetime.now().isoformat(timespec="microseconds")
    background_tasks.add_task(
        db.log_run,
        timestamp,
        duration=duration,
        modules=modules,
        modules_failed=modules_failed,
        version_multiqc=version_multiqc,
        version_python=version_python,
        operating_system=operating_system,
        is_docker=is_docker,
        is_singularity=is_singularity,
        is_conda=is_conda,
        is_ci=is_ci,
        user_ip=None,
    )


@app.get("/")
async def index(_: BackgroundTasks):
    """
    Root endpoint for the API.

    Returns a list of available endpoints.
    """
    routes = [cast(APIRoute, r) for r in app.routes]
    return {
        "message": "Welcome to the MultiQC runs logging service",
        "available_endpoints": [
            {"path": route.path, "name": route.name} for route in routes if route.name != "swagger_ui_redirect"
        ],
    }


@app.get("/health")
async def health():
    """
    Health check endpoint. Checks if the visits table contains records
    in the past 15 minutes.
    """
    try:
        visits = db.get_visit_stats(start=datetime.datetime.now() - datetime.timedelta(minutes=15))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    if not visits:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No recent visits found")
    return PlainTextResponse(content=str(len(visits)))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

import logging

import datetime
from typing import cast

import uvicorn

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRoute
from fastapi_utilities import repeat_every
from sqlalchemy.exc import ProgrammingError

from app import __version__, db

logger = logging.getLogger("multiqc_app_downloads")

logger.info("Starting MultiQC API download scraping service")

# Add timestamp to the uvicorn logger
for h in logging.getLogger("uvicorn.access").handlers:
    h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

app = FastAPI(
    title="MultiQC API",
    description="MultiQC API service, providing run-time information about available " "updates.",
    version=__version__,
    license_info={
        "name": "Source code available under the MIT Licence",
        "url": "https://github.com/MultiQC/api.multiqc.info/blob/main/LICENSE",
    },
)

db.create_db_and_tables()


@app.get("/")
async def index(_: BackgroundTasks):
    """
    Root endpoint for the API.
    Returns a list of available endpoints.
    """
    routes = [cast(APIRoute, r) for r in app.routes]
    return {
        "message": "Welcome to the MultiQC downloads scraping service",
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


@repeat_every(
    seconds=60 * 60 * 24,  # every day
    wait_first=True,
    logger=logger,
)
async def update_downloads():
    """
    Repeated task to update the daily download statistics
    """
    _update_download_stats()


@app.post("/update_downloads")
async def update_downloads_endpoint(background_tasks: BackgroundTasks):
    """
    Endpoint to manually update the daily download statistics
    """
    try:
        background_tasks.add_task(_update_download_stats)
        msg = "Queued updating the download stats in the DB"
        logger.info(msg)
        return PlainTextResponse(content=msg)
    except Exception as e:
        msg = f"Failed to update the download stats: {e}"
        raise HTTPException(status_code=status.INTERNAL_SERVER_ERROR, detail=msg)


def _update_download_stats():
    """
    Update the daily download statistics in the database
    """
    try:
        existing_downloads = db.get_download_stats()
    except ProgrammingError:
        logger.error("The table does not exist, will create and populate with historical data")
        existing_downloads = []
    if len(existing_downloads) == 0:  # first time, populate historical data
        logger.info("Collecting historical downloads data...")
        df = daily.collect_daily_download_stats()
        logger.info(f"Adding {len(df)} historical entries to the table...")
        db.insert_download_stats(df)
        logger.info(f"Successfully populated {len(df)} historical entries")
    else:  # recent days only
        n_days = 4
        logger.info(f"Updating downloads data for the last {n_days} days...")
        df = daily.collect_daily_download_stats(days=n_days)
        logger.info(f"Adding {len(df)} recent entries to the table. Will update existing " f"entries at the same date")
        db.insert_download_stats(df)
        logger.info(f"Successfully updated {len(df)} new daily download statistics")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

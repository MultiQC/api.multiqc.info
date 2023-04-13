from datetime import date

from pydantic import BaseModel, HttpUrl


class LatestRelease(BaseModel):
    """Model for the latest release."""

    version: str
    release_date: date
    html_url: HttpUrl


class BroadcastMessage(BaseModel):
    """Model for the broadcast message."""

    message: str
    level: str = "info"


class ModuleWarning(BaseModel):
    """Model for a module warning."""

    module: str
    message: str
    level: str


class VersionResponse(BaseModel):
    """Response model for the version endpoint."""

    latest_release: LatestRelease
    broadcast_messages: list[BroadcastMessage] = []
    module_warnings: list[ModuleWarning] = []

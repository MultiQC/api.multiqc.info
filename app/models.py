import datetime
from enum import Enum

from pydantic import BaseModel, HttpUrl


class LatestRelease(BaseModel):
    """Model for the latest release."""

    version: str
    release_date: datetime.date
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


class IntervalTypes(str, Enum):
    """Allowed intervals for the usage endpoint."""

    S = "second"
    T = "minute"
    H = "hour"
    D = "day"
    W = "week"
    M = "month"
    Y = "year"


class SortTypes(str, Enum):
    """Allowed sort types for the usage endpoint."""

    date_asc = "date_asc"
    date_desc = "date_desc"


class UsageCategory(str, Enum):
    """How to categorise the usage data."""

    version_multiqc = "version_multiqc"
    version_multiqc_simple = "version_multiqc_simple"
    version_python = "version_python"
    version_python_simple = "version_python_simple"
    operating_system = "operating_system"
    installation_method = "installation_method"
    ci_environment = "ci_environment"


usage_category_nicenames = dict(
    version_multiqc="MultiQC version",
    version_multiqc_simple="MultiQC version (simple)",
    version_python="Python version",
    version_python_simple="Python version (simple)",
    operating_system="Operating system",
    installation_method="Installation method",
    ci_environment="CI environment",
)

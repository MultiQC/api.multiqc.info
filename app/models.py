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


class IntervalType(str, Enum):
    """Allowed intervals for the usage endpoint."""

    s = "second"
    min = "minute"
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
    is_docker = "is_docker"
    is_singularity = "is_singularity"
    is_conda = "is_conda"
    is_ci = "is_ci"
    is_notebook = "is_notebook"
    interactive_function_name = "interactive_function_name"


usage_category_nice_names = dict(
    version_multiqc="MultiQC version",
    version_multiqc_simple="MultiQC version (simple)",
    version_python="Python version",
    version_python_simple="Python version (simple)",
    operating_system="Operating system",
    is_docker="Docker",
    is_singularity="Singularity",
    is_conda="Conda",
    is_ci="CI environment",
    is_notebook="Notebook",
    interactive_function_name="Interactive function",
)

"""api.multiqc.info: Providing run-time information about available updates."""

__version__ = "0.1.0.dev0"

from pathlib import Path

import os

import logging

from dotenv import load_dotenv

load_dotenv()

tmp_path = Path(os.getenv("TMPDIR", "/tmp"))
log_path = tmp_path / "multiqc_api.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path),
    ],
)
logging.debug(f"Logging to {log_path}")


def strtobool(val) -> bool:
    """
    Replaces deprecated https://docs.python.org/3.9/distutils/apiref.html#distutils.util.strtobool
    The deprecation recommendation is to re-implement the function https://peps.python.org/pep-0632/

    ------------------------------------------------------------

    Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val_str = str(val).lower()
    if val_str in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val_str in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r}")

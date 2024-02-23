"""api.multiqc.info: Providing run-time information about available updates."""

__version__ = "0.1.0.dev0"

from pathlib import Path

import os

import logging
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

load_dotenv()

tmp_path = Path(os.getenv("TMPDIR", "/tmp"))
log_path = tmp_path / "multiqc_api.log"
logging.basicConfig(
    level=logging.DEBUG if os.getenv("ENVIRONMENT") == "DEV" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(log_path, maxBytes=10_000, backupCount=10),
    ],
)
logging.debug(f"Logging to {log_path}")

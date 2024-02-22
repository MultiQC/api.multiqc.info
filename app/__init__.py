"""api.multiqc.info: Providing run-time information about available updates."""

__version__ = "0.1.0.dev0"

from pathlib import Path

import os

import logging

from dotenv import load_dotenv

load_dotenv()

log_path = Path(os.getenv("TMPDIR", "/tmp")) / "multiqc_api.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path),
    ],
)
logging.debug(f"Logging to {log_path}")

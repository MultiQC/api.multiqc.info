"""api.multiqc.info: Providing run-time information about available updates."""

__version__ = "0.1.0.dev0"

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from logzio.handler import LogzioHandler
from logging.handlers import RotatingFileHandler

load_dotenv()

tmp_path = Path(os.getenv("TMPDIR", "/tmp"))
log_path = tmp_path / "multiqc_api.log"

logger = logging.getLogger("multiqc_api")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = RotatingFileHandler(log_path, maxBytes=10_000, backupCount=10)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

logz_handler = LogzioHandler(os.environ['LOGZIO_TOKEN'])
logz_handler.setFormatter(formatter)
logz_handler.setLevel(logging.DEBUG)
logger.addHandler(logz_handler)

logger.info(f"Logging to {log_path}")

"""Centralized logging setup.

Each run mode (bot / dashboard / tui) writes to its own rotating file under
`logs/` AND to stdout. Unhandled main-thread exceptions are caught and logged
before the process exits, so a `chatterbot diagnose` bundle gathered after a
crash can show what actually killed it.

Call `setup_logging(mode)` exactly once at process boot, before any other
logging happens, so all subsequent `logging.getLogger(__name__)` calls inherit
the configured handlers.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

LOG_DIR = Path("logs")
MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
BACKUP_COUNT = 5             # keep up to 5 rotations
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(mode: str) -> Path:
    """Wire stdout + a per-mode rotating file handler. Returns the log path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{mode}.log"

    formatter = logging.Formatter(FORMAT)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Replace any prior handlers so re-init in tests doesn't duplicate.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)

    # Crash trap. KeyboardInterrupt stays interactive.
    crash_logger = logging.getLogger("chatterbot.crash")

    def _excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        crash_logger.error(
            "UNHANDLED EXCEPTION", exc_info=(exc_type, exc_value, exc_tb)
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook
    return log_path

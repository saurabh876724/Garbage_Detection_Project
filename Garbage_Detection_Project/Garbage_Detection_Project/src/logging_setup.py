"""Application logging setup (spec section 30).

One rotating file under logs/ for technical detail plus a quiet console
handler. User-facing code logs through get_logger(__name__) and shows
friendly messages itself - tracebacks never reach the operator.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import config

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: int = logging.INFO,
                  log_file: str = "app.log") -> logging.Logger:
    """Attach handlers once and return the root application logger."""
    global _configured
    config.ensure_dirs()
    root = logging.getLogger("garbage")
    if _configured:
        return root

    root.setLevel(level)
    formatter = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(config.LOG_DIR / log_file,
                                       maxBytes=2_000_000, backupCount=5,
                                       encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.WARNING)

    root.addHandler(file_handler)
    root.addHandler(console)
    root.propagate = False

    # Third-party libraries are noisy at INFO level.
    for noisy in ("ultralytics", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger, configuring handlers on first use."""
    if not _configured:
        setup_logging()
    return logging.getLogger(f"garbage.{name}")

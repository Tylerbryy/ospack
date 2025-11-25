"""Logging configuration for ospack."""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Get a logger with Rich formatting.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Use Rich handler for pretty output
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=False,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        # Set level from environment or default to INFO
        level = os.environ.get("OSPACK_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))

    return logger

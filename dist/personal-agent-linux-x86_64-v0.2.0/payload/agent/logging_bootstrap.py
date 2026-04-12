from __future__ import annotations

import logging
import sys
from typing import TextIO


_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging_if_needed(
    *,
    level: int = logging.INFO,
    stream: TextIO | None = None,
) -> bool:
    """Attach a stdout handler only when no explicit logging config exists."""

    root = logging.getLogger()
    if root.handlers:
        return False
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)
    return True


__all__ = ["configure_logging_if_needed"]

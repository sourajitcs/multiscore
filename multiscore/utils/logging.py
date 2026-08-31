from __future__ import annotations

import logging
import sys
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", stream=None) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        datefmt="%H:%M:%S",
        stream=stream or sys.stderr,
        force=True,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "multiscore")

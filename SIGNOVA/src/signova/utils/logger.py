"""
Structured logger setup for SIGNOVA.
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "signova", level: str = "INFO") -> logging.Logger:
    """Configure a clean, timestamped logger."""
    logger = logging.getLogger(name)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

import sys
import logging
from typing import Optional

from rag.config import settings


def setup_logger(
        name: str,
        level: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    if level is None:
        level = settings.log_level

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
       fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
       datefmt='%Y-%m-%d %H:%M:%S'
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger

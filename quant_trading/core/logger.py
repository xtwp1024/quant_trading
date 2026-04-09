import logging
import os
from rich.logging import RichHandler

def setup_logger(name: str = "Titan", level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )
    return logging.getLogger(name)

logger = setup_logger()

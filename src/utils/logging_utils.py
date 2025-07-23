import logging
import sys

_FMT = "[%(asctime)s] [%(levelname)-8s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str = "asd-scanner") -> logging.Logger:
    """Return a colourised console logger (INFO level default)."""
    logging.addLevelName(logging.INFO, "\033[1;32mINFO\033[0m")
    logging.addLevelName(logging.WARNING, "\033[1;33mWARN\033[0m")
    logging.addLevelName(logging.ERROR, "\033[1;31mERROR\033[0m")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_FMT, datefmt=_DATEFMT))

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

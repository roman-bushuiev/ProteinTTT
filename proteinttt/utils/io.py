import logging
import sys
from enum import Enum


def setup_logger(log_file_path=None, log_name='log', debug=False):
    """Setup a logger with a file handler and a stream handler.
    
    Copy from https://github.com/pluskal-lab/DreaMS/blob/4fbc05e6b264961e47906bafe6cd5f495a8cea54/dreams/utils/io.py#L38
    
    Args:
        log_file_path (str, optional): Path to the log file.
        log_name (str, optional): Name of the logger.
        debug (bool, optional): Whether to set the logger to debug level.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO if not debug else logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

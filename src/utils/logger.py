import logging
import colorlog
import os
from typing import Optional

def get_logger(name: str, log_to_file: bool = False, log_file_path: str = 'logs/app.log', log_level: int = logging.DEBUG) -> logging.Logger:
    """
    Configures and returns a logger instance.
    
    :param name: Name of the logger.
    :param log_to_file: Whether to log to a file.
    :param log_file_path: Path for the log file if log_to_file is True.
    :param log_level: Logging level.
    :return: Configured Logger instance.
    """
    log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    }
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(log_level)
        console_handler = logging.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=log_colors
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        if log_to_file:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            file_handler = logging.FileHandler(log_file_path)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    return logger

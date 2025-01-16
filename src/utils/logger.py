import logging
import colorlog
import os

def get_logger(name, log_to_file=False, log_file_path='logs/app.log', log_level=logging.DEBUG):
    """
    Configures and returns a logger instance.

    :param name: Name of the logger.
    :param log_to_file: Whether to log to a file (default=False).
    :param log_file_path: Path for the log file if `log_to_file=True`.
    :param log_level: Logging level (default=logging.DEBUG).
    """
    log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    }

    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # Ensure no duplicate handlers
        logger.setLevel(log_level)

        # Console handler with color
        console_handler = logging.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=log_colors
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Optional file handler
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

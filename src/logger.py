import logging
import colorlog

def get_logger(name):
    # Define log colors for different levels
    log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',  # Set INFO to green for general information messages
        'WARNING': 'yellow',  # Set WARNING to yellow for attention-worthy messages
        'ERROR': 'red',  # Set ERROR to red for critical issues
        'CRITICAL': 'bold_red'  # Set CRITICAL to bold red for fatal problems
    }

    # Create a logger
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Set log level
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

        # Create handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)  # Match the logger's level to capture all logs

        # Create a formatter that applies color based on log levels
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=log_colors
        )

        # Add formatter to the handler
        ch.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(ch)

    return logger

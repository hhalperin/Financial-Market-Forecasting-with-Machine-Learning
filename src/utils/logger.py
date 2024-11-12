import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Set log level
        logger.setLevel(logging.INFO)

        # Create handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add formatter to handler
        ch.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(ch)

    return logger

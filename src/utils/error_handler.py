import time
import logging
from functools import wraps

def handle_api_errors(func):
    """
    Decorator that retries an API call up to 3 times with a 5-second delay.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 3
        delay = 5  # seconds
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    logging.info(f"Retrying {func.__name__} (Attempt {attempt+2}/{retries})...")
                else:
                    logging.error(f"Failed after {retries} attempts.")
                    raise
    return wrapper

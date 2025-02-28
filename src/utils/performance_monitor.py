"""
Performance Monitor Module

Provides a decorator to measure function execution time and log a warning if it exceeds
a specified threshold.
"""

import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def profile_time(threshold: float = 0.5):
    """
    Decorator to measure the execution time of a function.
    Logs a warning if the function takes longer than 'threshold' seconds.
    
    :param threshold: Maximum acceptable execution time (in seconds) before logging a warning.
    :return: Decorated function with execution time profiling.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.monotonic()
            result = func(*args, **kwargs)
            elapsed = time.monotonic() - start_time
            if elapsed > threshold:
                logger.warning(
                    f"Performance Bottleneck: Function '{func.__name__}' took {elapsed:.3f} seconds, exceeding threshold of {threshold} seconds."
                )
            else:
                logger.info(f"Function '{func.__name__}' executed in {elapsed:.3f} seconds.")
            return result
        return wrapper
    return decorator

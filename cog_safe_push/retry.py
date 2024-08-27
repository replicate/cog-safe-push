import functools

from . import log


def retry(attempts=3):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    log.warning(f"Exception occurred: {e}")
                    if attempt < attempts:
                        log.warning(f"Retrying attempt {attempt}/{attempts}")
                    else:
                        log.warning(f"Giving up after {attempts} attempts")
                        raise
            return None

        return wrapper_retry

    return decorator_retry

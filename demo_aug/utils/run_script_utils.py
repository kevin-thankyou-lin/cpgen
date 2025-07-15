import logging
import random
import time
from typing import Any, Callable, Tuple, Type


def retry_on_exception(
    max_retries: int = 5,
    retry_delay: float = 1,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable:
    """
    Sample usage:

    @retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
    def my_function() -> None:
        # Your function code here
        pass

    Decorated function will return None if max_retries is reached.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for retry_count in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logging.warning(f"Retry {retry_count + 1} - Error: {e}")
                    time.sleep(retry_delay * (2**retry_count) + random.uniform(0, 0.1))
            logging.error(
                f"Max retries reached. Unable to complete {func} ({args}, {kwargs})."
            )
            return None

        return wrapper

    return decorator

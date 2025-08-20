import time
from sqlalchemy.exc import OperationalError

def retry_on_operational_error(fn, retries=3, delay=2):
    """
    Retry a database operation if an OperationalError occurs.
    Useful for ephemeral CI connections that sometimes drop SSL.
    """
    for i in range(retries):
        try:
            return fn()
        except OperationalError as e:
            print(f"DB operation failed, retry {i + 1}/{retries}: {e}")
            if i == retries - 1:
                raise
            time.sleep(delay)

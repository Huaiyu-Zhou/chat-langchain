# test_db.py
import time
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from backend.db import engine

def test_query():
    with engine.connect() as conn:
        r = conn.execute(text("SELECT EXTRACT(EPOCH FROM CURRENT_TIMESTAMP);"))
        print("DB OK:", r.scalar())

def retry_on_operational_error(fn, retries=3, delay=2):
    """
    Retry a database operation if an OperationalError occurs.
    Useful for CI ephemeral connections.
    """
    for i in range(retries):
        try:
            return fn()
        except OperationalError as e:
            print(f"DB operation failed, retry {i + 1}/{retries}: {e}")
            if i == retries - 1:
                raise
            time.sleep(delay)

# Run the test query with retries
retry_on_operational_error(test_query)

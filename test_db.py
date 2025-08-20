# test_db.py
from sqlalchemy import text
from backend.db import engine

with engine.connect() as conn:
    r = conn.execute(text("SELECT EXTRACT(EPOCH FROM CURRENT_TIMESTAMP);"))
    print("DB OK:", r.scalar())

# backend/db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

DATABASE_URL = os.getenv("RECORD_MANAGER_DB_URL")
if not DATABASE_URL:
    raise RuntimeError("RECORD_MANAGER_DB_URL environment variable is not set")

# Use NullPool for CI / short-lived jobs to avoid EOF on reused sockets.
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    connect_args={"sslmode": "require"},
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

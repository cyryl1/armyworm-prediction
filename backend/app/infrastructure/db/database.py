"""SQLAlchemy engine, session factory, and base model.

This module is the single place where the database connection is
configured.  Everything else depends on the ``SessionLocal`` factory
and the ``Base`` declarative base.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")

# Pin the psycopg2 driver (the one we install); newer SQLAlchemy defaults
# plain "postgresql://" URLs to psycopg v3. Also accept legacy "postgres://".
for _prefix in ("postgres://", "postgresql://"):
    if SQLALCHEMY_DATABASE_URL.startswith(_prefix):
        SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://" + SQLALCHEMY_DATABASE_URL[len(_prefix):]
        break

# Only use check_same_thread for SQLite
connect_args = (
    {"check_same_thread": False}
    if SQLALCHEMY_DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

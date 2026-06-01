"""
Database engine, session factory, and table initialisation.

Default:  SQLite (file trading_bot.db in the working directory)
Override: set DATABASE_URL env var — e.g. postgresql://user:pass@host/db
"""
from __future__ import annotations

import os
import logging

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./trading_bot.db")

# SQLite needs check_same_thread=False for async-adjacent usage
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,        # recycle stale connections
    echo=False,                # set True to log every SQL statement
)

# Enable WAL mode for SQLite — much better concurrent read performance
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_wal(dbapi_conn, _record):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        dbapi_conn.execute("PRAGMA synchronous=NORMAL")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables that don't exist yet.  Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database ready: {DATABASE_URL.split('?')[0]}")


def get_session() -> Session:
    """Return a new SQLAlchemy session.  Caller is responsible for closing it."""
    return SessionLocal()

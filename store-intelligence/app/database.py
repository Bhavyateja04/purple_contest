"""
Database setup and session management.
Uses SQLAlchemy with SQLite (configurable via DATABASE_URL env var).
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import (
    Column, String, Boolean, Float, Integer, Text,
    DateTime, Index, create_engine, text
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./store_intelligence.db")


def _make_engine():
    kwargs = {}
    if DATABASE_URL.startswith("sqlite"):
        kwargs = {
            "connect_args": {"check_same_thread": False},
            "poolclass": StaticPool,
        }
    return create_engine(DATABASE_URL, echo=False, **kwargs)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class EventRecord(Base):
    __tablename__ = "events"

    event_id = Column(String, primary_key=True, index=True)
    store_id = Column(String, nullable=False, index=True)
    camera_id = Column(String, nullable=False)
    visitor_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    zone_id = Column(String, nullable=True, index=True)
    dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, default=False)
    confidence = Column(Float, default=1.0)
    # metadata fields flattened
    queue_depth = Column(Integer, nullable=True)
    sku_zone = Column(String, nullable=True)
    session_seq = Column(Integer, default=0)
    # ingestion metadata
    ingested_at = Column(DateTime, nullable=False)


# Composite index for common query patterns
Index("ix_events_store_ts", EventRecord.store_id, EventRecord.timestamp)
Index("ix_events_store_type", EventRecord.store_id, EventRecord.event_type)
Index("ix_events_visitor", EventRecord.visitor_id, EventRecord.event_type)


def create_tables():
    import app.database as _self
    Base.metadata.create_all(bind=_self.engine)
    logger.info("Database tables created/verified")


def get_db() -> Generator[Session, None, None]:
    # Use module-level SessionLocal so tests can swap it out
    import app.database as _self
    db = _self.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_db_connection() -> bool:
    try:
        import app.database as _self
        with _self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB connection check failed: {e}")
        return False

"""Database engine and session management (SQLAlchemy 2.x).

The engine is created lazily (on first use), so importing this module never fails just because
credentials are missing — useful for tests, which point DATABASE_URL at SQLite instead.
"""

import logging
from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import get_database_url

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM table classes in database/models.py."""


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = get_database_url()
    # pool_pre_ping: transparently reconnect if PostgreSQL restarted.
    engine = create_engine(url, pool_pre_ping=True)
    logger.info("Database engine created for %s", engine.url.render_as_string(hide_password=True))
    return engine


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed afterwards."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()

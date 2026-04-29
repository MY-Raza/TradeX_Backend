from __future__ import annotations
import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

settings = get_settings()

# ── Convert any sync URL variant → asyncpg URL ────────────────────────────
def _make_async_url(url: str) -> str:
    url = url.replace("?sslmode=require", "").replace("&sslmode=require", "")

    replacements = [
        ("postgresql+psycopg2://", "postgresql+asyncpg://"),
        ("postgresql+psycopg://",  "postgresql+asyncpg://"),
        ("postgresql://",          "postgresql+asyncpg://"),
        ("postgres://",            "postgresql+asyncpg://"),
    ]
    for old, new in replacements:
        if url.startswith(old):
            return url.replace(old, new, 1)
    return url

_async_url = _make_async_url(settings.DATABASE_URL)
logging.warning(f"[DEBUG] Async URL being used: {_async_url}")

# ── Engine ─────────────────────────────────────────────────────────────────
engine = create_async_engine(
    _async_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
    echo=settings.DEBUG,
    future=True,
    connect_args={
        "ssl": False,
        "statement_cache_size": 0,
        "timeout": 30,
        "server_settings": {"application_name": "railway-backend"}
    },
)

# ── Session factory ────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ── Declarative Base ───────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── FastAPI dependency ─────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
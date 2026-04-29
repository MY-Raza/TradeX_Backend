from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────
    # Points to local socat tunnel which forwards to PC PostgreSQL via Tailscale
    DATABASE_URL: str = "postgresql+asyncpg://postgres:y2s57r29a@127.0.0.1:5433/TradeX"

    # ── SQLAlchemy pool ───────────────────────────────────────────────────
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_PRE_PING: bool = True

    # ── App ───────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    DEBUG: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
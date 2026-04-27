from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str  # No default — forces Railway to supply it

    # ── SQLAlchemy pool ───────────────────────────────────────────────────
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_PRE_PING: bool = True

    # ── App ───────────────────────────────────────────────────────────────
    APP_ENV: str = "production"
    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # ← Railway env vars are case-insensitive
        extra="ignore",         # ← ignore any extra vars Railway injects
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
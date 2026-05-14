from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
import os

from app.api.routes import strategies, models, data, backtest, sentiment, createstrategy, ai_route
from app.db.session import get_db


# ---------------------------------------------------------------------------
# Startup: add the three "last run" columns to strategy_registry if missing.
# The CSV / DB table was created without them; they are only written by
# backtest_service after a run — but every GET /backtest/strategies and
# GET /strategies call tries to SELECT them, which crashes until they exist.
# Using IF NOT EXISTS makes this completely safe to run on every deploy.
# ---------------------------------------------------------------------------

async def _ensure_strategy_columns() -> None:
    async for db in get_db():
        try:
            for col_def in [
                "ADD COLUMN IF NOT EXISTS last_pnl_pct DOUBLE PRECISION",
                "ADD COLUMN IF NOT EXISTS last_run_tp  DOUBLE PRECISION",
                "ADD COLUMN IF NOT EXISTS last_run_sl  DOUBLE PRECISION",
            ]:
                await db.execute(
                    text(f"ALTER TABLE strategies.strategy_registry {col_def}")
                )
            await db.commit()
            print("[startup] strategy_registry columns ensured OK")
        except Exception as exc:
            print(f"[startup] Could not alter strategy_registry: {exc}")
            await db.rollback()
        break  # get_db is a generator — only need one session


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _ensure_strategy_columns()
    yield

app = FastAPI(title="TradeX API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS – explicit origins + regex to cover all Vercel preview deployments
# Add EXTRA_ORIGINS env var on Railway for any additional domains:
#   e.g.  EXTRA_ORIGINS=https://my-custom-domain.com
# ---------------------------------------------------------------------------

_base_origins = [
    "https://tradex-rho-seven.vercel.app",   # production Vercel deployment
    "http://localhost:5173",                   # Vite dev server
    "http://localhost:3000",                   # alternative local dev
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

_extra = os.getenv("EXTRA_ORIGINS", "")
if _extra:
    _base_origins += [o.strip() for o in _extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_base_origins,
    # Covers ALL Vercel preview URLs: tradex-rho-seven-git-*.vercel.app, etc.
    allow_origin_regex=r"https://tradex-.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(strategies.router)
app.include_router(models.router)
app.include_router(data.router)
app.include_router(backtest.router)
app.include_router(sentiment.router)
app.include_router(createstrategy.router)
app.include_router(ai_route.router)


@app.get("/")
def root():
    return {"message": "TradeX Backend Running"}
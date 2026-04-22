"""
TradeX – Sentiment Router

Endpoints
---------
GET  /sentiment/coins                  → list[CoinOption]            Dropdown options
POST /sentiment/run                    → SentimentRunResponse         Full pipeline
GET  /sentiment/results/{coin}         → SentimentResultsResponse     Cached results
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.sentiment_schema import (
    CoinOption,
    SentimentResultsResponse,
    SentimentRunRequest,
    SentimentRunResponse,
)
from app.services import sentiment_service

router = APIRouter(prefix="/sentiment", tags=["Sentiment"])

DB = Annotated[AsyncSession, Depends(get_db)]


# ===========================================================================
# GET /sentiment/coins
# ===========================================================================

@router.get(
    "/coins",
    response_model=list[CoinOption],
    summary="List supported coins",
    description=(
        "Returns every coin that the sentiment pipeline supports. "
        "Use `id` as `coin` in the run request and results endpoint."
    ),
)
async def list_coins() -> list[CoinOption]:
    return await sentiment_service.get_supported_coins()


# ===========================================================================
# POST /sentiment/run
# ===========================================================================

@router.post(
    "/run",
    response_model=SentimentRunResponse,
    summary="Run sentiment pipeline",
    description=(
        "Scrapes Reddit for the latest posts and comments, runs FinBERT sentiment "
        "analysis filtered to the selected coin, stores results in four tables "
        "(`<coin>_posts_sentiment`, `<coin>_comments_sentiment`, "
        "`<coin>_posts_sentiment_hourly`, `<coin>_comments_sentiment_hourly`), "
        "and returns the full results including individual posts, hourly chart "
        "data, and overall stats."
    ),
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Unsupported coin supplied.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Scraper or FinBERT pipeline raised an exception.",
        },
    },
)
async def run_sentiment(
    req: SentimentRunRequest,
    db:  DB,
) -> SentimentRunResponse:
    return await sentiment_service.run_sentiment(db, req)


# ===========================================================================
# GET /sentiment/results/{coin}
# ===========================================================================

@router.get(
    "/results/{coin}",
    response_model=SentimentResultsResponse,
    summary="Get cached sentiment results",
    description=(
        "Returns previously analysed sentiment results for the given coin "
        "directly from the database without re-running the pipeline. "
        "Returns 404 if the pipeline has not been run yet for this coin."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Sentiment tables not found — run the pipeline first.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Unsupported coin.",
        },
    },
)
async def get_results(
    db:   DB,
    coin: str = Path(..., description="Coin id, e.g. 'btc' | 'eth' | 'sol'"),
) -> SentimentResultsResponse:
    return await sentiment_service.get_sentiment_results(db, coin)
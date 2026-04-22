"""
TradeX – Sentiment Service

Pipeline triggered by POST /sentiment/run
------------------------------------------
1. Run reddit_scrapper  → saves to reddit.reddit_posts / reddit.reddit_comments
2. Run sentiment_analysis.run_pipeline(coin)
   → saves to reddit.<coin>_posts_sentiment, reddit.<coin>_comments_sentiment,
             reddit.<coin>_posts_sentiment_hourly, reddit.<coin>_comments_sentiment_hourly
3. Read results back via SQLAlchemy Table() objects (sentiment_model.py)
4. Shape into SentimentResultsResponse and return

GET /sentiment/results/{coin}
------------------------------
- Reads the four tables and returns the same SentimentResultsResponse
  (no re-analysis; just serves cached DB results)

GET /sentiment/coins
---------------------
- Returns list[CoinOption] built from COIN_CONFIG
"""

from __future__ import annotations

import asyncio
import math
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import inspect, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.sentiment_model import (
    get_posts_sentiment_table,
    get_comments_sentiment_table,
    get_posts_hourly_table,
    get_comments_hourly_table,
)
from app.schemas.sentiment_schema import (
    COIN_CONFIG,
    CoinOption,
    HourlyPoint,
    OverallStats,
    SentimentComment,
    SentimentPost,
    SentimentResultsResponse,
    SentimentRunRequest,
    SentimentRunResponse,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA = "reddit"

LABEL_MAP = {
    1:  "Positive",
    0:  "Neutral",
    -1: "Negative",
}


# ─────────────────────────────────────────────────────────────────────────────
# Safe type coercions
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _safe_int(val) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Table existence check  (uses information_schema — avoids reflect overhead)
# ─────────────────────────────────────────────────────────────────────────────

async def _table_exists(db: AsyncSession, schema: str, table: str) -> bool:
    sql = text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.tables "
        "  WHERE table_schema = :schema AND table_name = :table"
        ")"
    )
    result = await db.execute(sql, {"schema": schema, "table": table})
    return bool(result.scalar())


# ─────────────────────────────────────────────────────────────────────────────
# Row shapers  (RowMapping → Pydantic schema)
# ─────────────────────────────────────────────────────────────────────────────

def _shape_posts(rows) -> list[SentimentPost]:
    out = []
    for r in rows:
        score = _safe_int(r.sentiment_score)
        out.append(SentimentPost(
            id         = str(r.post_id),
            title      = str(r.title or ""),
            author     = str(r.author or ""),
            upvotes    = _safe_int(r.estimated_upvotes if r.estimated_upvotes is not None else r.score),
            comments   = _safe_int(r.num_comments),
            sentiment  = LABEL_MAP.get(score, "Neutral"),
            score      = _safe_float(r.sentiment_score),
            confidence = _safe_float(r.sentiment_confidence),
            subreddit  = str(r.subreddit or ""),
            post_time  = str(r.post_time) if r.post_time else None,
        ))
    return out


def _shape_comments(rows) -> list[SentimentComment]:
    out = []
    for r in rows:
        score = _safe_int(r.sentiment_score)
        out.append(SentimentComment(
            id           = str(r.comment_id),
            text         = str(r.comment_text or "")[:300],   # truncate for frontend
            author       = str(r.comment_author or ""),
            upvotes      = _safe_int(r.comment_score),
            sentiment    = LABEL_MAP.get(score, "Neutral"),
            score        = _safe_float(r.sentiment_score),
            confidence   = _safe_float(r.sentiment_confidence),
            subreddit    = str(r.subreddit or ""),
            comment_time = str(r.comment_time) if r.comment_time else None,
        ))
    return out


def _shape_hourly(rows) -> list[HourlyPoint]:
    out = []
    for r in rows:
        out.append(HourlyPoint(
            hour          = str(r.time_window)[:16] if r.time_window else "",
            sentiment     = _safe_float(r.mean_sentiment),
            confidence    = _safe_float(r.sentiment_confidence_mean),
            std_sentiment = _safe_float(r.std_sentiment),
            post_count    = _safe_int(r.post_id_count),
        ))
    return out


def _overall_stats(
    posts:    list[SentimentPost],
    comments: list[SentimentComment],
) -> OverallStats:
    all_scores = [p.score for p in posts] + [c.score for c in comments]
    all_confs  = [p.confidence for p in posts] + [c.confidence for c in comments]

    n      = len(all_scores) or 1
    mean_s = sum(all_scores) / n
    std_s  = (sum((x - mean_s) ** 2 for x in all_scores) / n) ** 0.5
    mean_c = sum(all_confs) / n

    return OverallStats(
        mean_sentiment  = round(mean_s, 4),
        std_sentiment   = round(std_s,  4),
        confidence_mean = round(mean_c, 4),
        total_posts     = len(posts),
        total_comments  = len(comments),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DB fetch  (uses Table() objects from sentiment_model.py)
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_results(db: AsyncSession, coin: str) -> SentimentResultsResponse:
    """
    Read all four sentiment tables for the given coin using SQLAlchemy Table()
    objects and build the response. Raises HTTP 404 if any table is missing.
    """
    # Resolve Table() descriptors for this coin
    tbl_posts    = get_posts_sentiment_table(coin)
    tbl_comments = get_comments_sentiment_table(coin)
    tbl_ph       = get_posts_hourly_table(coin)
    tbl_ch       = get_comments_hourly_table(coin)

    table_map = {
        tbl_posts.name:    tbl_posts,
        tbl_comments.name: tbl_comments,
        tbl_ph.name:       tbl_ph,
        tbl_ch.name:       tbl_ch,
    }

    # Verify all four tables exist before querying
    for tbl_name in table_map:
        if not await _table_exists(db, SCHEMA, tbl_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Sentiment table '{SCHEMA}.{tbl_name}' not found. "
                    f"Run 'Analyze Sentiment' for coin='{coin}' first."
                ),
            )

    # Execute all four selects concurrently
    (
        posts_result,
        comments_result,
        ph_result,
        ch_result,
    ) = await asyncio.gather(
        db.execute(select(tbl_posts).order_by(   tbl_posts.c.post_time.desc())),
        db.execute(select(tbl_comments).order_by(tbl_comments.c.comment_time.desc())),
        db.execute(select(tbl_ph).order_by(      tbl_ph.c.time_window.asc())),
        db.execute(select(tbl_ch).order_by(      tbl_ch.c.time_window.asc())),
    )

    posts    = _shape_posts(   posts_result.mappings().all())
    comments = _shape_comments(comments_result.mappings().all())
    hourly_p = _shape_hourly(  ph_result.mappings().all())
    hourly_c = _shape_hourly(  ch_result.mappings().all())
    overall  = _overall_stats(posts, comments)

    return SentimentResultsResponse(
        coin            = coin,
        posts           = posts,
        comments        = comments,
        hourly_posts    = hourly_p,
        hourly_comments = hourly_c,
        overall         = overall,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scraper runner  (blocking – runs in thread pool)
# ─────────────────────────────────────────────────────────────────────────────

def _run_scraper() -> None:
    """
    Import and execute the reddit scraper synchronously.
    Designed to be called via asyncio.to_thread().

    The scraper executes at module level, so we import (or reload) it to run it.
    """
    import importlib
    import os
    import sys

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    module_path = "TradeX.sentiments.data.reddit_scrapper"
    if module_path in sys.modules:
        importlib.reload(sys.modules[module_path])
    else:
        importlib.import_module(module_path)


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment pipeline runner  (blocking – runs in thread pool)
# ─────────────────────────────────────────────────────────────────────────────

def _run_sentiment_pipeline(coin: str) -> None:
    """
    Execute the FinBERT sentiment pipeline for the given coin.
    Designed to be called via asyncio.to_thread().
    """
    from app.services.sentiment_analysis import run_pipeline   # lazy import
    run_pipeline(coin=coin, apply_coin_filter=True, save_to_database=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public service functions called by the router
# ─────────────────────────────────────────────────────────────────────────────

async def get_supported_coins() -> list[CoinOption]:
    """Return dropdown list of all supported coins."""
    return [
        CoinOption(id=coin_id, display=cfg["display"])
        for coin_id, cfg in COIN_CONFIG.items()
    ]


async def run_sentiment(
    db:  AsyncSession,
    req: SentimentRunRequest,
) -> SentimentRunResponse:
    """
    Full pipeline:
      1. Validate coin against COIN_CONFIG
      2. Run Reddit scraper in thread pool  (non-blocking)
      3. Run FinBERT pipeline for selected coin in thread pool  (non-blocking)
      4. Read results from DB via SQLAlchemy Table() objects
      5. Return SentimentRunResponse
    """
    coin = req.coin.lower()

    if coin not in COIN_CONFIG:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported coin '{coin}'. "
                f"Valid options: {list(COIN_CONFIG.keys())}"
            ),
        )

    # Step 1 — Scrape Reddit
    try:
        await asyncio.to_thread(_run_scraper)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reddit scraper failed: {exc}",
        )

    # Step 2 — Run FinBERT pipeline for selected coin
    try:
        await asyncio.to_thread(_run_sentiment_pipeline, coin)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment pipeline failed for coin='{coin}': {exc}",
        )

    # Step 3 — Read and return results
    results = await _fetch_results(db, coin)

    return SentimentRunResponse(
        coin    = coin,
        message = (
            f"Sentiment analysis complete for {COIN_CONFIG[coin]['display']}. "
            f"{results.overall.total_posts} posts and "
            f"{results.overall.total_comments} comments analysed."
        ),
        results = results,
    )


async def get_sentiment_results(
    db:   AsyncSession,
    coin: str,
) -> SentimentResultsResponse:
    """
    Return cached sentiment results from the DB without re-running the pipeline.
    Used by the frontend to populate the tab on revisit / coin switch.
    """
    coin = coin.lower()

    if coin not in COIN_CONFIG:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported coin '{coin}'. Valid: {list(COIN_CONFIG.keys())}",
        )

    return await _fetch_results(db, coin)
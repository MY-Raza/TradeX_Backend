"""
TradeX – ORM models for sentiment data sources.

Four dynamic tables are resolved at runtime, all in schema `reddit`:
  1. <coin>_posts_sentiment           – per-post FinBERT results
  2. <coin>_comments_sentiment        – per-comment FinBERT results
  3. <coin>_posts_sentiment_hourly    – hourly aggregated post sentiment
  4. <coin>_comments_sentiment_hourly – hourly aggregated comment sentiment

All use SQLAlchemy Table() so the coin name can be injected at call time,
consistent with get_price_table() / get_predictions_table() in backtest_model.py.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Text,
)

# ---------------------------------------------------------------------------
# Schema constant
# ---------------------------------------------------------------------------
SENTIMENT_SCHEMA = "reddit"


# ---------------------------------------------------------------------------
# 1. Per-post sentiment  →  reddit.<coin>_posts_sentiment
# ---------------------------------------------------------------------------

def get_posts_sentiment_table(coin: str) -> Table:
    """
    Return a Table descriptor for  reddit.<coin>_posts_sentiment.

    Columns mirror the output of sentiment_analysis.add_sentiment_to_df()
    applied to reddit_posts, plus the original post metadata columns
    written by reddit_scrapper.py.

    Args:
        coin: Lowercase coin id, e.g. "btc", "eth", "sol"

    Returns:
        SQLAlchemy Table object bound to schema=reddit
    """
    meta = MetaData(schema=SENTIMENT_SCHEMA)
    return Table(
        f"{coin}_posts_sentiment",
        meta,
        # ── Original scraper columns ──────────────────────────────────────
        Column("post_id",               String,                  nullable=True),
        Column("post_time",             DateTime(timezone=True), primary_key=True, nullable=False),
        Column("subreddit",             String,                  nullable=True),
        Column("title",                 Text,                    nullable=True),
        Column("author",                String,                  nullable=True),
        Column("score",                 Integer,                 nullable=True),
        Column("upvote_ratio",          Float,                   nullable=True),
        Column("estimated_upvotes",     Integer,                 nullable=True),
        Column("estimated_downvotes",   Integer,                 nullable=True),
        Column("num_comments",          Integer,                 nullable=True),
        Column("url",                   Text,                    nullable=True),
        # ── Cleaned text ──────────────────────────────────────────────────
        Column("cleaned_text",          Text,                    nullable=True),
        Column("is_valid",              Boolean,                 nullable=True),
        # ── Feature extraction columns (from data_cleaner.py) ────────────
        Column("emoji_count",           Integer,                 nullable=True),
        Column("caps_ratio",            Float,                   nullable=True),
        Column("punct_intensity",       Integer,                 nullable=True),
        Column("exclamation_count",     Integer,                 nullable=True),
        Column("question_count",        Integer,                 nullable=True),
        Column("spam_score",            Float,                   nullable=True),
        Column("token_count",           Integer,                 nullable=True),
        Column("content_hash",          String,                  nullable=True),
        Column("simhash",               String,                  nullable=True),   # uint64 stored as str
        Column("had_url",               Boolean,                 nullable=True),
        Column("had_bot_content",       Boolean,                 nullable=True),
        # ── FinBERT sentiment output ──────────────────────────────────────
        Column("sentiment_score",       SmallInteger,            nullable=True),   # -1 | 0 | 1
        Column("sentiment_confidence",  Float,                   nullable=True),   # [0, 1]
        Column("sentiment_label",       String,                  nullable=True),   # negative | neutral | positive
    )


# ---------------------------------------------------------------------------
# 2. Per-comment sentiment  →  reddit.<coin>_comments_sentiment
# ---------------------------------------------------------------------------

def get_comments_sentiment_table(coin: str) -> Table:
    """
    Return a Table descriptor for  reddit.<coin>_comments_sentiment.

    Columns mirror the output of sentiment_analysis.add_sentiment_to_df()
    applied to reddit_comments, plus the original comment metadata columns
    written by reddit_scrapper.py.

    Args:
        coin: Lowercase coin id, e.g. "btc", "eth", "sol"

    Returns:
        SQLAlchemy Table object bound to schema=reddit
    """
    meta = MetaData(schema=SENTIMENT_SCHEMA)
    return Table(
        f"{coin}_comments_sentiment",
        meta,
        # ── Original scraper columns ──────────────────────────────────────
        Column("comment_id",            String,                  nullable=True),
        Column("comment_time",          DateTime(timezone=True), primary_key=True, nullable=False),
        Column("post_id",               String,                  nullable=True),
        Column("subreddit",             String,                  nullable=True),
        Column("comment_text",          Text,                    nullable=True),
        Column("comment_author",        String,                  nullable=True),
        Column("comment_score",         Integer,                 nullable=True),
        # ── Cleaned text ──────────────────────────────────────────────────
        Column("cleaned_text",          Text,                    nullable=True),
        Column("is_valid",              Boolean,                 nullable=True),
        # ── Feature extraction columns (from data_cleaner.py) ────────────
        Column("emoji_count",           Integer,                 nullable=True),
        Column("caps_ratio",            Float,                   nullable=True),
        Column("punct_intensity",       Integer,                 nullable=True),
        Column("exclamation_count",     Integer,                 nullable=True),
        Column("question_count",        Integer,                 nullable=True),
        Column("spam_score",            Float,                   nullable=True),
        Column("token_count",           Integer,                 nullable=True),
        Column("content_hash",          String,                  nullable=True),
        Column("simhash",               String,                  nullable=True),
        Column("had_url",               Boolean,                 nullable=True),
        Column("had_bot_content",       Boolean,                 nullable=True),
        # ── FinBERT sentiment output ──────────────────────────────────────
        Column("sentiment_score",       SmallInteger,            nullable=True),
        Column("sentiment_confidence",  Float,                   nullable=True),
        Column("sentiment_label",       String,                  nullable=True),
    )


# ---------------------------------------------------------------------------
# 3. Hourly post aggregation  →  reddit.<coin>_posts_sentiment_hourly
# ---------------------------------------------------------------------------

def get_posts_hourly_table(coin: str) -> Table:
    """
    Return a Table descriptor for  reddit.<coin>_posts_sentiment_hourly.

    Columns mirror the output of sentiment_analysis.aggregate_sentiment_hourly()
    applied to the posts DataFrame.

    Args:
        coin: Lowercase coin id, e.g. "btc", "eth", "sol"

    Returns:
        SQLAlchemy Table object bound to schema=reddit
    """
    meta = MetaData(schema=SENTIMENT_SCHEMA)
    return Table(
        f"{coin}_posts_sentiment_hourly",
        meta,
        Column("time_window",               DateTime(timezone=True), primary_key=True, nullable=False),
        Column("mean_sentiment",            Float,   nullable=True),
        Column("std_sentiment",             Float,   nullable=True),
        Column("sentiment_confidence_mean", Float,   nullable=True),
        Column("emoji_count_mean",          Float,   nullable=True),
        Column("caps_ratio_mean",           Float,   nullable=True),
        Column("punct_intensity_mean",      Float,   nullable=True),
        Column("spam_score_mean",           Float,   nullable=True),
        Column("token_count_mean",          Float,   nullable=True),
        Column("post_id_count",             Integer, nullable=True),
    )


# ---------------------------------------------------------------------------
# 4. Hourly comment aggregation  →  reddit.<coin>_comments_sentiment_hourly
# ---------------------------------------------------------------------------

def get_comments_hourly_table(coin: str) -> Table:
    """
    Return a Table descriptor for  reddit.<coin>_comments_sentiment_hourly.

    Columns mirror the output of sentiment_analysis.aggregate_sentiment_hourly()
    applied to the comments DataFrame.

    Args:
        coin: Lowercase coin id, e.g. "btc", "eth", "sol"

    Returns:
        SQLAlchemy Table object bound to schema=reddit
    """
    meta = MetaData(schema=SENTIMENT_SCHEMA)
    return Table(
        f"{coin}_comments_sentiment_hourly",
        meta,
        Column("time_window",               DateTime(timezone=True), primary_key=True, nullable=False),
        Column("mean_sentiment",            Float,   nullable=True),
        Column("std_sentiment",             Float,   nullable=True),
        Column("sentiment_confidence_mean", Float,   nullable=True),
        Column("emoji_count_mean",          Float,   nullable=True),
        Column("caps_ratio_mean",           Float,   nullable=True),
        Column("punct_intensity_mean",      Float,   nullable=True),
        Column("spam_score_mean",           Float,   nullable=True),
        Column("token_count_mean",          Float,   nullable=True),
        Column("post_id_count",             Integer, nullable=True),
    )
"""
TradeX – Sentiment Pydantic Schemas

Endpoints served
----------------
POST /sentiment/run                → SentimentRunResponse        (scrape + analyse)
GET  /sentiment/results/{coin}     → SentimentResultsResponse    (fetch stored results)
GET  /sentiment/coins              → list[CoinOption]            (supported coins)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ===========================================================================
# Supported coins registry
# Centralised here so both the schema and service share the same source of truth.
# ===========================================================================

COIN_CONFIG: dict[str, dict] = {
    "btc": {
        "display":    "Bitcoin (BTC)",
        "pattern":    r"\b(btc|bitcoin)\b|\$btc",
        "native_subs": {"bitcoin", "bitcoinmarkets", "satoshistreetbets"},
    },
    "eth": {
        "display":    "Ethereum (ETH)",
        "pattern":    r"\b(eth|ethereum)\b|\$eth",
        "native_subs": {"ethereum", "ethtrader"},
    },
    "bnb": {
        "display":    "BNB",
        "pattern":    r"\b(bnb|binance\s*coin)\b|\$bnb",
        "native_subs": set(),
    },
    "sol": {
        "display":    "Solana (SOL)",
        "pattern":    r"\b(sol|solana)\b|\$sol",
        "native_subs": {"solana"},
    },
    "xrp": {
        "display":    "XRP / Ripple",
        "pattern":    r"\b(xrp|ripple)\b|\$xrp",
        "native_subs": {"ripple", "xrp"},
    },
    "ada": {
        "display":    "Cardano (ADA)",
        "pattern":    r"\b(ada|cardano)\b|\$ada",
        "native_subs": {"cardano"},
    },
    "doge": {
        "display":    "Dogecoin (DOGE)",
        "pattern":    r"\b(doge|dogecoin)\b|\$doge",
        "native_subs": {"dogecoin"},
    },
    "dot": {
        "display":    "Polkadot (DOT)",
        "pattern":    r"\b(dot|polkadot)\b|\$dot",
        "native_subs": {"polkadot"},
    },
    "avax": {
        "display":    "Avalanche (AVAX)",
        "pattern":    r"\b(avax|avalanche)\b|\$avax",
        "native_subs": {"avax"},
    },
    "link": {
        "display":    "Chainlink (LINK)",
        "pattern":    r"\b(link|chainlink)\b|\$link",
        "native_subs": set(),
    },
}


# ===========================================================================
# Coin dropdown option  (GET /sentiment/coins)
# ===========================================================================

class CoinOption(BaseModel):
    id:      str   # e.g. "btc"
    display: str   # e.g. "Bitcoin (BTC)"


# ===========================================================================
# Run request  (POST /sentiment/run)
# ===========================================================================

class SentimentRunRequest(BaseModel):
    coin: str = Field(
        ...,
        description="Coin id from COIN_CONFIG, e.g. 'btc' | 'eth' | 'sol'",
    )


# ===========================================================================
# Individual post / comment row returned to the frontend
# ===========================================================================

class SentimentPost(BaseModel):
    id:         str
    title:      str
    author:     str
    upvotes:    int
    comments:   int
    sentiment:  str    # "Positive" | "Neutral" | "Negative"
    score:      float  # numeric: -1 | 0 | 1
    confidence: float  # [0, 1]
    subreddit:  Optional[str] = None
    post_time:  Optional[str] = None


class SentimentComment(BaseModel):
    id:           str
    text:         str
    author:       str
    upvotes:      int
    sentiment:    str
    score:        float
    confidence:   float
    subreddit:    Optional[str] = None
    comment_time: Optional[str] = None


# ===========================================================================
# Hourly chart point
# ===========================================================================

class HourlyPoint(BaseModel):
    hour:          str
    sentiment:     float
    confidence:    float
    std_sentiment: Optional[float] = None
    post_count:    Optional[int]   = None


# ===========================================================================
# Overall analytics stats
# ===========================================================================

class OverallStats(BaseModel):
    mean_sentiment:  float
    std_sentiment:   float
    confidence_mean: float
    total_posts:     int
    total_comments:  int


# ===========================================================================
# Full results response  (GET /sentiment/results/{coin})
# ===========================================================================

class SentimentResultsResponse(BaseModel):
    coin:            str
    posts:           list[SentimentPost]
    comments:        list[SentimentComment]
    hourly_posts:    list[HourlyPoint]
    hourly_comments: list[HourlyPoint]
    overall:         OverallStats


# ===========================================================================
# Run response  (POST /sentiment/run)
# ===========================================================================

class SentimentRunResponse(BaseModel):
    coin:    str
    message: str
    results: SentimentResultsResponse
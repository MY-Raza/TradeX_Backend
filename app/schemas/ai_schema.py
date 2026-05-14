"""
TradeX – AI Chat Pydantic Schemas

Endpoints served
----------------
POST /ai/chat      → AIChatResponse
GET  /ai/history   → list[ChatMessage]
DELETE /ai/history → AIDeleteHistoryResponse
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ===========================================================================
# Chat turn
# ===========================================================================

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


# ===========================================================================
# Tool execution record – one entry per service call the AI made
# ===========================================================================

class ToolExecution(BaseModel):
    tool_name: str = Field(
        ...,
        description=(
            "Internal name of the service function called, "
            "e.g. 'get_strategies', 'run_backtest', 'get_sentiment_results'"
        ),
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the service function",
    )
    status: Literal["success", "error"] = "success"
    error: Optional[str] = Field(
        None,
        description="Error message if the tool call failed",
    )
    result_summary: Optional[str] = Field(
        None,
        description="Short human-readable description of what was returned",
    )


# ===========================================================================
# POST /ai/chat – request
# ===========================================================================

class AIChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language prompt from the user",
    )
    session_id: Optional[str] = Field(
        None,
        description=(
            "Opaque client-side session identifier. "
            "When supplied the server returns the conversation history "
            "scoped to that session. When omitted a new session is started."
        ),
    )


# ===========================================================================
# POST /ai/chat – response
# ===========================================================================

class AIChatResponse(BaseModel):
    session_id: str = Field(
        ...,
        description="Session identifier – echo back to maintain conversation history",
    )
    reply: str = Field(
        ...,
        description="AI-generated natural-language answer",
    )
    tools_executed: list[ToolExecution] = Field(
        default_factory=list,
        description="Ordered list of backend service calls the AI made",
    )
    data: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Structured payload from the last tool execution "
            "(e.g. BacktestResponse, PaginatedStrategies). "
            "Null when the AI answered purely from context."
        ),
    )


# ===========================================================================
# GET /ai/history – response item already defined as ChatMessage above
# ===========================================================================

class AIHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]
    total: int


# ===========================================================================
# DELETE /ai/history – response
# ===========================================================================

class AIDeleteHistoryResponse(BaseModel):
    session_id: str
    deleted: bool
    message: str


# ===========================================================================
# Internal – Gemini function-call argument schemas
# These mirror the JSON schemas sent to Gemini as tool declarations.
# Kept here so ai_model.py stays clean and schemas are version-controlled.
# ===========================================================================

class _GetStrategiesArgs(BaseModel):
    symbol: Optional[str] = None
    time_horizon: Optional[str] = None
    search: Optional[str] = None
    page: int = 1
    page_size: int = 20


class _GetStrategyDetailArgs(BaseModel):
    strategy_name: str


class _RunBacktestArgs(BaseModel):
    strategy_name: str
    exchange: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    starting_balance: float = 1000.0
    take_profit: float = 1.0
    stop_loss: float = 1.0
    buy_after_minutes: int = 0
    fee: float = 0.05
    leverage: float = 1.0
    slippage: float = 0.0


class _GetBacktestStrategiesArgs(BaseModel):
    pass  # no arguments – returns full dropdown list


class _GetStrategyRunsArgs(BaseModel):
    strategy_name: str


class _GetModelsArgs(BaseModel):
    model_type: Literal["ml", "dl"] = "ml"
    search: Optional[str] = None
    page: int = 1
    page_size: int = 20


class _GetModelDetailArgs(BaseModel):
    model_type: Literal["ml", "dl"]
    model_name: str


class _GetSentimentResultsArgs(BaseModel):
    coin: str


class _RunSentimentArgs(BaseModel):
    coin: str


class _GetOHLCVArgs(BaseModel):
    exchange: str
    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
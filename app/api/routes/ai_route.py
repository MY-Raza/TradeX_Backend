"""
TradeX – AI Chat Router

Endpoints
---------
POST   /ai/chat                          → AIChatResponse
GET    /ai/history/{session_id}          → AIHistoryResponse
DELETE /ai/history/{session_id}          → AIDeleteHistoryResponse
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.ai_schema import (
    AIChatRequest,
    AIChatResponse,
    AIDeleteHistoryResponse,
    AIHistoryResponse,
)
from app.services import ai_service

router = APIRouter(prefix="/ai", tags=["AI"])

DB = Annotated[AsyncSession, Depends(get_db)]


# ===========================================================================
# POST /ai/chat
# ===========================================================================

@router.post(
    "/chat",
    response_model=AIChatResponse,
    summary="Send a natural-language prompt to the TradeX AI",
    description=(
        "Accepts a natural-language message and an optional session_id. "
        "The AI analyses the prompt, calls the appropriate backend services "
        "(strategies, backtest, models, sentiment, OHLCV), and returns a "
        "structured response with a text reply, the tool execution trace, "
        "and the raw structured data from the last tool call.\n\n"
        "**Example prompts**\n"
        "- `Select the best strategy for BTC 1h and run the backtest from 2025-01-01 to 2025-12-31`\n"
        "- `What is the win rate of strategy sig_1h_btc_3 on Binance?`\n"
        "- `Show me the current BTC sentiment`\n"
        "- `List all ETH strategies sorted by timeframe`\n"
        "- `Compare ML models and show the top performer`"
    ),
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Gemini API is unreachable or returned an error.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error during AI orchestration.",
        },
    },
)
async def ai_chat(
    req: AIChatRequest,
    db: DB,
) -> AIChatResponse:
    return await ai_service.process_chat(
        db=db,
        user_message=req.message,
        session_id=req.session_id,
    )


# ===========================================================================
# GET /ai/history/{session_id}
# ===========================================================================

@router.get(
    "/history/{session_id}",
    response_model=AIHistoryResponse,
    summary="Retrieve conversation history for a session",
    description=(
        "Returns all text messages (user and assistant turns) for the given "
        "session. Tool-call parts are filtered out – only human-readable "
        "content is returned. Returns an empty list if the session does not "
        "exist or has no messages."
    ),
)
async def get_history(
    session_id: str = Path(..., description="Session id returned by POST /ai/chat"),
) -> AIHistoryResponse:
    messages = ai_service.get_session_messages(session_id)
    return AIHistoryResponse(
        session_id=session_id,
        messages=messages,
        total=len(messages),
    )


# ===========================================================================
# DELETE /ai/history/{session_id}
# ===========================================================================

@router.delete(
    "/history/{session_id}",
    response_model=AIDeleteHistoryResponse,
    summary="Clear conversation history for a session",
    description=(
        "Permanently removes all stored conversation turns for the given "
        "session. Returns `deleted: true` if the session existed, "
        "`deleted: false` if it was not found (idempotent)."
    ),
)
async def delete_history(
    session_id: str = Path(..., description="Session id to clear"),
) -> AIDeleteHistoryResponse:
    deleted = ai_service.delete_session(session_id)
    return AIDeleteHistoryResponse(
        session_id=session_id,
        deleted=deleted,
        message=(
            f"Session '{session_id}' cleared successfully."
            if deleted
            else f"Session '{session_id}' not found – nothing to delete."
        ),
    )
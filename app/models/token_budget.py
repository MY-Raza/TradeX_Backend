"""
TradeX – Token Budget Manager  (v3 – Strategy Generator Compression)
=====================================================================

WHAT CHANGED vs v2
-------------------
The only change in v3 is the extension of _HEAVY_ARRAY_KEYS to include
field names that appear in CreateStrategyResponse:

  "ledger"       – list[LedgerEntry]       (can be 100s of rows)
  "pnl_data"     – list[PnLPoint]          (one point per trade)
  "win_loss_data"– list[WinLossPoint]      (only 2 items, but included for consistency)

WHY: compress_tool_result() truncates any list whose key is in _HEAVY_ARRAY_KEYS
to _MAX_ARRAY_ITEMS entries (currently 3) and adds a __<key>_total__ sibling
so the AI knows the full size.  Without these additions, a strategy with 200
trades would store a 200-entry ledger in session history on every create_strategy
call, inflating the token count by thousands and triggering 413 errors.

The compressed version stored in _SESSION_STORE still contains:
  - ledger[0:3]           – first 3 trades for context
  - __ledger_total__      – total trade count (e.g. 187)
  - pnl_data[0:3]         – first 3 PnL points
  - __pnl_data_total__    – total PnL point count
  - summary               – full BacktestSummary (not a list, always kept)
  - strategy_id, message  – always kept (not lists)

The FULL uncompressed CreateStrategyResponse is always returned to the UI via
AIChatResponse.data.  Only the session-history copy is compressed.

NO OTHER CHANGES: All functions, constants, and logic from v2 are preserved
exactly as-is.  This file is a drop-in replacement for the v2 token_budget.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("tradex.ai.token_budget")

# ---------------------------------------------------------------------------
# Token budget constants
# ---------------------------------------------------------------------------

# Hard ceiling for input tokens sent to Groq (free tier: 12 000 TPM total).
# We reserve ~4 500 for output + framing overhead.
TOKEN_BUDGET: int = 7_500

# When the session store reaches this token count we proactively summarise.
# This prevents the store from growing to the point where even a fresh request
# exceeds the hard ceiling.
SOFT_COMPRESSION_THRESHOLD: int = 6_000

# Emergency ceiling used during 413 retry path (absolute minimum).
EMERGENCY_TOKEN_BUDGET: int = 4_500

# How many recent messages to ALWAYS keep (never summarised away).
# Covers: last user turn + assistant tool-call msg + tool result + final reply
# = 4 messages minimum.  We keep 6 to preserve one full extra exchange.
_MIN_RECENT_MESSAGES: int = 6

# How many items from a "heavy" array field to keep in compressed tool results.
_MAX_ARRAY_ITEMS: int = 3   # was 5 in v1; tighter = safer

# Conservative chars-per-token ratio (mixed JSON/English; ±15% accuracy).
_CHARS_PER_TOKEN: float = 3.8

# Tool results older than this many rounds are eligible for deep compression.
_STALE_TOOL_ROUNDS: int = 2


# ===========================================================================
# Token estimation
# ===========================================================================

def estimate_tokens(text: str) -> int:
    """
    Fast ≈token count via character heuristic.
    Accuracy: ±15 % – good enough for budget gating.
    """
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Estimate total token count for an OpenAI-format messages list.
    Accounts for content, tool_calls, and per-message overhead (~4 tokens).
    """
    total = 0
    for msg in messages:
        total += 4  # role + framing overhead
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += estimate_tokens(json.dumps(tool_calls, default=str))
    return total


# ===========================================================================
# Tool result compression
# ===========================================================================

# Fields known to carry large arrays. Trimmed aggressively in history copies.
#
# v3 ADDITIONS (marked with # NEW):
#   "ledger"        – list[LedgerEntry] from CreateStrategyResponse
#   "pnl_data"      – list[PnLPoint]    from CreateStrategyResponse / BacktestResponse
#   "win_loss_data" – list[WinLossPoint] from CreateStrategyResponse
#
# These were missing in v2, causing full strategy ledgers (100s of rows) to be
# stored in session history after every create_strategy call.
_HEAVY_ARRAY_KEYS: frozenset[str] = frozenset({
    # existing (v2)
    "trades", "win_loss_chart", "pnl_per_trade", "equity_curve", "candles",
    "ohlcv", "rows", "data", "posts", "results", "hourly_sentiment",
    "hourly", "per_post", "items", "records", "entries",
    # new (v3) – strategy generator response arrays
    "ledger",           # NEW: list[LedgerEntry] – often 100s of rows
    "pnl_data",         # NEW: list[PnLPoint]    – one per trade
    "win_loss_data",    # NEW: list[WinLossPoint] – always 2 items but consistent
    "ranked_strategies", # NEW: list in compare_strategies result (max 5, compress anyway)
})


def compress_tool_result(tool_name: str, data: dict[str, Any]) -> dict[str, Any]:
    """
    Return a token-reduced copy of a tool result dict for storage in history.

    Heavy array fields are truncated to _MAX_ARRAY_ITEMS entries.  A
    `__<key>_total__` sibling key tells the model the original count so it
    can still reason about the full dataset.

    The UNCOMPRESSED original is always returned to the UI via
    AIChatResponse.data – only the history copy is compressed.
    """
    return _compress_dict(data, depth=0)


def _compress_dict(obj: Any, depth: int) -> Any:
    """Recursive helper – depth-limited to guard against pathological nesting."""
    if depth > 3:
        return obj
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, list) and k in _HEAVY_ARRAY_KEYS:
                original_len = len(v)
                truncated = v[:_MAX_ARRAY_ITEMS]
                out[k] = [_compress_dict(item, depth + 1) for item in truncated]
                if original_len > _MAX_ARRAY_ITEMS:
                    out[f"__{k}_total__"] = original_len
                    out[f"__{k}_shown__"] = len(truncated)
            else:
                out[k] = _compress_dict(v, depth + 1)
        return out
    if isinstance(obj, list):
        return [_compress_dict(item, depth + 1) for item in obj]
    return obj


def make_ultra_compact_tool_summary(tool_name: str, content_json: str) -> str:
    """
    Build a 1-line summary of a tool result for use in session summaries.

    Called by build_summary_message() when compressing old turns.
    Returns a string like:
      "run_backtest: 142 trades, win_rate=62.3%, pnl=+14.50%"
    """
    try:
        data = json.loads(content_json)
    except (json.JSONDecodeError, TypeError):
        return f"{tool_name}: (unparseable result)"

    try:
        # ── Existing tool summaries (v2, unchanged) ──────────────────────
        if tool_name == "get_strategies":
            return f"get_strategies: total={data.get('total','?')} strategies"
        if tool_name == "get_strategy_detail":
            return f"get_strategy_detail: strategy='{data.get('name','?')}'"
        if tool_name == "get_backtest_strategies":
            count = data.get("count") or len(data.get("items", []))
            return f"get_backtest_strategies: {count} strategies available"
        if tool_name == "run_backtest":
            s = data.get("summary", {})
            return (
                f"run_backtest: {s.get('total_trades','?')} trades, "
                f"win_rate={s.get('win_rate','?')}%, "
                f"pnl={s.get('total_pnl_pct','?'):+.2f}%"
                if isinstance(s.get("total_pnl_pct"), (int, float))
                else f"run_backtest: {s.get('total_trades','?')} trades"
            )
        if tool_name == "get_strategy_runs":
            return f"get_strategy_runs: {data.get('count','?')} runs"
        if tool_name in ("get_sentiment_results", "run_sentiment"):
            overall = data.get("overall", {})
            return (
                f"{tool_name}: coin={data.get('coin','?')}, "
                f"mean={overall.get('mean_sentiment','?')}"
            )
        if tool_name == "get_ohlcv":
            return (
                f"get_ohlcv: {data.get('total_rows','?')} candles, "
                f"last_close={data.get('close','?')}"
            )
        if tool_name == "get_models":
            return f"get_models: total={data.get('total','?')} models"
        if tool_name == "get_model_detail":
            return (
                f"get_model_detail: model='{data.get('model_name','?')}', "
                f"win_rate={data.get('win_rate','?')}%"
            )

        # ── v3: Strategy generator tool summaries ────────────────────────
        if tool_name == "create_strategy":
            s = data.get("summary", {})
            return (
                f"create_strategy: id='{data.get('strategy_id','?')}', "
                f"{s.get('total_trades','?')} trades, "
                f"win_rate={s.get('win_rate','?')}%, "
                f"pnl={s.get('total_pnl_pct','?'):+.2f}%"
                if isinstance(s.get("total_pnl_pct"), (int, float))
                else f"create_strategy: id='{data.get('strategy_id','?')}'"
            )

        if tool_name == "compare_strategies":
            return (
                f"compare_strategies: generated={data.get('generated','?')}, "
                f"best='{data.get('best_strategy_id','?')}', "
                f"best_win_rate={data.get('best_win_rate','?')}%, "
                f"best_pnl={data.get('best_pnl_pct','?'):+.2f}%"
                if isinstance(data.get("best_pnl_pct"), (int, float))
                else f"compare_strategies: generated={data.get('generated','?')}"
            )

    except Exception:
        pass

    return f"{tool_name}: executed"


def deduplicate_tool_results(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove duplicate tool result messages from history (same tool_call_id).

    Duplicates can accumulate when a tool is called multiple times in the same
    session (e.g. get_backtest_strategies called twice across two rounds).
    Keeping only the most recent result saves tokens and avoids confusing the
    model with stale data.
    """
    seen_tool_call_ids: set[str] = set()
    result: list[dict[str, Any]] = []

    # Traverse in reverse to keep most recent; then reverse back
    for msg in reversed(history):
        if msg.get("role") == "tool":
            tcid = msg.get("tool_call_id") or msg.get("name", "")
            if tcid in seen_tool_call_ids:
                continue  # drop older duplicate
            seen_tool_call_ids.add(tcid)
        result.append(msg)

    result.reverse()
    return result


def trim_history(
    messages: list[dict[str, Any]],
    budget: int = TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    """
    Hard-trim a messages list to fit within *budget* estimated tokens.

    Strategy:
    1. Always keep the system prompt (index 0).
    2. Always keep at least _MIN_RECENT_MESSAGES from the tail.
    3. Drop the oldest non-system messages until within budget.

    This is a structural trim (drops entire messages) unlike compress_session()
    which summarises.  Used as a final safety net.
    """
    if not messages:
        return messages

    system_msg = messages[0]
    non_system = list(messages[1:])

    # Protect the most recent _MIN_RECENT_MESSAGES
    protected_tail = non_system[-_MIN_RECENT_MESSAGES:] if len(non_system) >= _MIN_RECENT_MESSAGES else non_system
    trimmable = non_system[:-_MIN_RECENT_MESSAGES] if len(non_system) >= _MIN_RECENT_MESSAGES else []

    # Drop from the oldest end until we fit
    while trimmable:
        candidate = [system_msg] + trimmable + protected_tail
        if estimate_messages_tokens(candidate) <= budget:
            return candidate
        trimmable.pop(0)

    # Only system + protected tail remains
    minimal = [system_msg] + protected_tail
    if estimate_messages_tokens(minimal) <= budget:
        return minimal

    # Nuclear: system only
    return [system_msg]


# ===========================================================================
# Session-level summarisation (proactive, mutates _SESSION_STORE in place)
# ===========================================================================

def build_summary_message(history: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build a compact role="user" summary message from a history list.

    The summary is built locally without an LLM call and covers:
    - Which tools were executed and their compact outcomes
    - The last user question and assistant answer (truncated to 400 chars each)

    This is injected at position [1] (right after the system prompt) to give
    the model context about what happened earlier without paying full token cost.
    """
    tool_summaries: list[str] = []
    last_user = ""
    last_assistant = ""

    for msg in history:
        role = msg.get("role")
        if role == "tool":
            name = msg.get("name", "tool")
            content = msg.get("content", "{}")
            tool_summaries.append(make_ultra_compact_tool_summary(name, content))
        elif role == "user":
            content = msg.get("content") or ""
            if isinstance(content, str) and content:
                last_user = content[-400:]
        elif role == "assistant":
            content = msg.get("content") or ""
            if isinstance(content, str) and content:
                last_assistant = content[-400:]

    parts = ["[PRIOR CONTEXT – older turns were compressed to save tokens]"]
    if tool_summaries:
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = [s for s in tool_summaries if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]
        parts.append("Tools used: " + " | ".join(unique))
    if last_user:
        parts.append(f"User asked: {last_user}")
    if last_assistant:
        parts.append(f"Assistant replied: {last_assistant}")

    return {"role": "user", "content": "\n".join(parts)}


def compress_session(
    session_store: dict[str, list[dict[str, Any]]],
    session_id: str,
    budget: int = SOFT_COMPRESSION_THRESHOLD,
) -> None:
    """
    Proactively compress a session IN PLACE when it exceeds *budget* tokens.

    This is called by _append_message() after every write.  It ensures the
    stored history never grows uncontrollably between requests, which was the
    root cause of repeated 413 errors even after single-retry logic was added.

    Strategy
    --------
    1. Estimate current token usage.
    2. If within budget → no-op (fast path).
    3. Build a summary from all messages older than _MIN_RECENT_MESSAGES.
    4. Replace the old messages with: system_msg + summary_msg + recent_tail.
    5. Write the compressed list back to session_store[session_id].

    The user-visible history (GET /ai/history) is stored separately in a
    _HISTORY_STORE dict (see ai_service.py) and is NOT compressed, so the
    user always sees their full conversation.
    """
    history = session_store.get(session_id)
    if not history:
        return

    # First pass: dedup (cheap, always run)
    history = deduplicate_tool_results(history)
    session_store[session_id] = history

    current_tokens = estimate_messages_tokens(history)
    if current_tokens <= budget:
        return  # fast path – no compression needed

    logger.info(
        "compress_session | session=%s | tokens=%d > soft_threshold=%d | compressing",
        session_id, current_tokens, budget,
    )

    system_msg = history[0]
    non_system = history[1:]

    # Identify the "old" portion to summarise
    recent_cutoff = max(0, len(non_system) - _MIN_RECENT_MESSAGES)
    old_messages = non_system[:recent_cutoff]
    recent_tail = non_system[recent_cutoff:]

    if not old_messages:
        # Nothing to summarise – all messages are "recent".
        # Force-trim the tail to fit.
        session_store[session_id] = trim_history(history, budget=budget)
        return

    # Build a summary of the old messages
    summary_msg = build_summary_message(old_messages)

    # Check if any existing summary message is already at index 1 –
    # if so, merge with it rather than stacking summaries.
    if (
        recent_tail
        and recent_tail[0].get("role") == "user"
        and recent_tail[0].get("content", "").startswith("[PRIOR CONTEXT")
    ):
        # Already has a summary; merge
        existing_summary = recent_tail[0]["content"]
        merged_content = summary_msg["content"] + "\n\n" + existing_summary
        summary_msg = {"role": "user", "content": merged_content[-1200:]}
        recent_tail = recent_tail[1:]

    compressed = [system_msg, summary_msg] + recent_tail
    after_tokens = estimate_messages_tokens(compressed)

    logger.info(
        "compress_session | session=%s | before=%d | after=%d tokens | "
        "summarised %d old messages",
        session_id, current_tokens, after_tokens, len(old_messages),
    )

    # If still over budget after summary injection, do a structural trim
    if after_tokens > budget:
        compressed = trim_history(compressed, budget=budget)

    session_store[session_id] = compressed


# ===========================================================================
# Emergency compression (used inside _call_groq_with_budget 413 retry path)
# ===========================================================================

def emergency_compress(
    messages: list[dict[str, Any]],
    budget: int = EMERGENCY_TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    """
    Maximally compress a messages list to fit within *budget*.

    This is called during the 413 retry path to ensure the retry payload will
    definitely fit, even if it means keeping very little history.

    Unlike compress_session() this does NOT mutate the session store – it
    returns a new (disposable) messages list for the retry call only.
    The session store is separately updated via compress_session().

    Steps
    -----
    1. Keep system prompt.
    2. Build a summary of everything.
    3. Keep only the last 2 messages (most recent user + assistant).
    4. If still over budget, drop the summary and keep only last 1 message.
    """
    if not messages:
        return messages

    system_msg = messages[0]
    non_system = messages[1:]

    summary_msg = build_summary_message(non_system)

    # Try: system + summary + last 3 messages
    for tail_size in (3, 2, 1):
        recent = non_system[-tail_size:] if len(non_system) >= tail_size else non_system
        candidate = [system_msg, summary_msg] + recent
        if estimate_messages_tokens(candidate) <= budget:
            logger.info(
                "emergency_compress | fit with tail=%d | tokens=%d",
                tail_size, estimate_messages_tokens(candidate),
            )
            return candidate

    # Absolute minimum: system + summary only
    minimal = [system_msg, summary_msg]
    if estimate_messages_tokens(minimal) <= budget:
        return minimal

    # Nuclear option: system + truncated summary
    max_summary_chars = int(budget * _CHARS_PER_TOKEN * 0.8)
    truncated_content = summary_msg["content"][:max_summary_chars]
    return [system_msg, {"role": "user", "content": truncated_content}]
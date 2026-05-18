"""
TradeX – Token Budget Manager
==============================

WHY THIS FILE EXISTS
--------------------
Groq's free tier enforces a **TPM (Tokens Per Minute) limit of 12 000** for
llama-3.3-70b-versatile.  A single request that carries a long conversation
history + large tool results can easily exceed this limit, producing:

    Error 413 – Request too large for model `llama-3.3-70b-versatile`
    TPM Limit: 12000  Requested: 24320

Root causes in the original code
---------------------------------
1. ``_MAX_HISTORY_MESSAGES = 50`` – each tool round appends 2–3 messages
   (assistant tool_call + tool result).  After a few backtests the history
   balloons past the token limit even though the message *count* is ≤ 50.
2. ``run_backtest`` returns a full trade ledger + chart data – JSON-dumped
   into the history as a ``role="tool"`` message, often 5 000–15 000 tokens
   by itself.
3. ``get_sentiment_results`` embeds every Reddit post; similar problem.
4. ``max_tokens=4096`` in ``call_groq()`` inflates the *requested* token
   count even when a short answer is expected – Groq counts requested output
   tokens against the TPM limit.
5. No pre-flight token estimate → the 413 is only discovered after the call.

This module provides
--------------------
- ``estimate_tokens(text)``          – fast ≈4-chars-per-token heuristic
- ``estimate_messages_tokens(msgs)`` – total tokens across a messages list
- ``compress_tool_result(name, data)``– strips chart arrays / raw posts from
                                        heavy payloads before they enter history
- ``trim_history(history, budget)``  – intelligent rolling trim that preserves
                                        the system prompt and recent context
- ``summarise_history(history)``     – condenses old turns into one summary
                                        message when trim alone isn't enough
- ``TOKEN_BUDGET``                   – safe ceiling (9 000) leaving headroom
                                        below the 12 000 TPM limit for output
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("tradex.ai.token_budget")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Groq free-tier TPM hard limit is 12 000.
# We target 9 000 for *input* tokens so the model has ≈3 000 for its output.
# Adjust TOKEN_BUDGET upward if you upgrade to a paid plan.
TOKEN_BUDGET: int = 9_000

# Absolute minimum context to retain after aggressive trimming.
# Keeps the system prompt + at least the last 2 user/assistant exchanges.
_MIN_MESSAGES_TO_KEEP: int = 5

# Tool results that include large array fields have those fields capped.
_MAX_ARRAY_ITEMS_IN_TOOL_RESULT: int = 5

# Characters-per-token ratio (conservative for mixed code/text/JSON content).
# tiktoken would be more accurate but adds a dependency; 3.8 is a safe floor.
_CHARS_PER_TOKEN: float = 3.8


# ===========================================================================
# Token estimation
# ===========================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in *text* using a character-count heuristic.

    Accuracy: ±10 % for English prose, ±20 % for dense JSON.  Good enough for
    budget-gating; we add a safety margin via TOKEN_BUDGET anyway.

    We deliberately avoid importing tiktoken to keep the dependency surface
    small and the startup time fast.
    """
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Estimate total token count for an OpenAI-format messages list.

    Accounts for:
    - ``content`` field (str or None)
    - ``tool_calls`` array (serialised to JSON for counting)
    - A small per-message overhead (role string + framing ≈ 4 tokens each)
    """
    total = 0
    for msg in messages:
        # Per-message overhead
        total += 4

        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)

        # Tool-call messages store args in a nested structure
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += estimate_tokens(json.dumps(tool_calls, default=str))

    return total


# ===========================================================================
# Tool result compression
# ===========================================================================

# Fields known to carry large arrays.  When a tool result contains these keys
# the array is replaced with a compact summary so the model still understands
# *what* was returned without ingesting thousands of tokens of raw data.
_HEAVY_ARRAY_KEYS: frozenset[str] = frozenset(
    {
        # run_backtest payloads
        "trades",
        "win_loss_chart",
        "pnl_per_trade",
        "equity_curve",
        "candles",
        # get_ohlcv
        "ohlcv",
        "rows",
        "data",
        # sentiment
        "posts",
        "results",
        "hourly_sentiment",
        "hourly",
        "per_post",
        # generic
        "items",
        "records",
        "entries",
    }
)


def compress_tool_result(tool_name: str, data: dict[str, Any]) -> dict[str, Any]:
    """
    Return a token-reduced copy of a tool result dict.

    Strategy
    --------
    1. Walk top-level and one level of nesting.
    2. For any list field whose key is in ``_HEAVY_ARRAY_KEYS``:
       - Keep only the first ``_MAX_ARRAY_ITEMS_IN_TOOL_RESULT`` items.
       - Add a ``"__truncated__": <original_count>`` sibling key so the model
         knows data was cut.
    3. Recursively apply to nested dicts (e.g. ``result.summary`` sub-objects).

    This keeps structured metadata (counts, win rates, PnL) visible to the
    model while stripping the raw ledger that only the UI needs.

    The *original* uncompressed data is still returned to the caller via
    ``last_structured_data`` in AIChatResponse – only the *history copy* is
    compressed.
    """
    return _compress_dict(data, depth=0)


def _compress_dict(obj: Any, depth: int) -> Any:
    """Recursive helper for compress_tool_result."""
    if depth > 3:          # guard against pathologically nested objects
        return obj
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, list) and k in _HEAVY_ARRAY_KEYS:
                original_len = len(v)
                truncated = v[:_MAX_ARRAY_ITEMS_IN_TOOL_RESULT]
                out[k] = [_compress_dict(item, depth + 1) for item in truncated]
                if original_len > _MAX_ARRAY_ITEMS_IN_TOOL_RESULT:
                    out[f"__{k}_total__"] = original_len
                    out[f"__{k}_shown__"] = len(truncated)
            else:
                out[k] = _compress_dict(v, depth + 1)
        return out
    if isinstance(obj, list):
        # Non-heavy-key lists: compress each element, don't truncate
        return [_compress_dict(item, depth + 1) for item in obj]
    return obj


# ===========================================================================
# History trimming
# ===========================================================================

def trim_history(
    history: list[dict[str, Any]],
    budget: int = TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    """
    Trim an OpenAI-format messages list so its estimated token count fits
    within *budget*.

    Algorithm
    ---------
    1. Always keep ``history[0]`` (system prompt) – it must never be dropped.
    2. Estimate total tokens.  If within budget, return as-is.
    3. Build a "mandatory tail" = last ``_MIN_MESSAGES_TO_KEEP`` non-system
       messages.  These are always retained to preserve immediate context.
    4. Remove messages from the *oldest* end (index 1 onwards, skipping system)
       until the estimate fits, or until only the mandatory tail remains.
    5. If even the minimum tail still exceeds budget, truncate *content* of
       heavy tool messages rather than dropping them entirely.

    Returns a new list (does not mutate the original).
    """
    if not history:
        return history

    # Fast path: already within budget
    current_tokens = estimate_messages_tokens(history)
    if current_tokens <= budget:
        return history

    logger.info(
        "trim_history | tokens=%d > budget=%d | trimming history (len=%d)",
        current_tokens, budget, len(history),
    )

    system_msg = history[0]          # always index 0 by convention
    non_system = list(history[1:])   # mutable working copy

    # Mandatory tail: keep the last N messages unconditionally
    mandatory_start = max(0, len(non_system) - _MIN_MESSAGES_TO_KEEP)
    mandatory_tail = non_system[mandatory_start:]
    trimmable = non_system[:mandatory_start]

    # Drop from the oldest end of the trimmable region
    while trimmable:
        candidate = [system_msg] + trimmable + mandatory_tail
        if estimate_messages_tokens(candidate) <= budget:
            break
        dropped = trimmable.pop(0)
        logger.debug(
            "trim_history | dropped message role=%s content_len=%s",
            dropped.get("role"),
            len(str(dropped.get("content") or "")),
        )

    trimmed = [system_msg] + trimmable + mandatory_tail
    after_tokens = estimate_messages_tokens(trimmed)

    logger.info(
        "trim_history | before=%d tokens | after=%d tokens | removed=%d messages",
        current_tokens, after_tokens, len(history) - len(trimmed),
    )

    # Last resort: truncate the content of the single largest tool message
    # if we're still over budget (protects against one giant tool result)
    if after_tokens > budget:
        trimmed = _truncate_largest_tool_message(trimmed, budget)

    return trimmed


def _truncate_largest_tool_message(
    history: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    """
    Find the tool message with the largest content string and hard-truncate it
    to fit within the remaining token budget.  Applied as a last resort when
    structural trimming isn't enough.
    """
    # Find the largest tool message (skip system / first message)
    largest_idx = -1
    largest_len = 0
    for i, msg in enumerate(history):
        if msg.get("role") == "tool":
            content = msg.get("content") or ""
            if isinstance(content, str) and len(content) > largest_len:
                largest_len = len(content)
                largest_idx = i

    if largest_idx == -1:
        return history   # nothing to truncate

    # How many chars can we keep in that message?
    other_tokens = estimate_messages_tokens(
        [m for i, m in enumerate(history) if i != largest_idx]
    )
    remaining_budget = budget - other_tokens
    max_chars = max(200, int(remaining_budget * _CHARS_PER_TOKEN))

    new_history = list(history)
    original_content = new_history[largest_idx].get("content", "")
    if isinstance(original_content, str) and len(original_content) > max_chars:
        truncated_content = (
            original_content[:max_chars]
            + "\n... [TRUNCATED – data too large for context window] ..."
        )
        new_history[largest_idx] = {
            **new_history[largest_idx],
            "content": truncated_content,
        }
        logger.warning(
            "_truncate_largest_tool_message | truncated tool result from %d to %d chars",
            len(original_content), max_chars,
        )

    return new_history


# ===========================================================================
# History summarisation
# ===========================================================================

def build_summary_message(history: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Produce a single ``role="user"`` message that summarises the conversation
    history provided.  This is injected *before* the current messages when the
    full history exceeds the token budget even after trimming.

    The summary is built locally (no LLM call) from structured data already
    in the history to avoid extra API calls.  It covers:
    - Which tools were called and their outcomes
    - The most recent user/assistant exchange
    """
    tool_calls_seen: list[str] = []
    last_user: str = ""
    last_assistant: str = ""

    for msg in history:
        role = msg.get("role")
        if role == "tool":
            name = msg.get("name", "unknown_tool")
            tool_calls_seen.append(name)
        elif role == "user":
            content = msg.get("content") or ""
            if isinstance(content, str) and content:
                last_user = content[-500:]   # last 500 chars of latest user turn
        elif role == "assistant":
            content = msg.get("content") or ""
            if isinstance(content, str) and content:
                last_assistant = content[-500:]

    parts: list[str] = ["[CONVERSATION SUMMARY – earlier context was compressed]"]
    if tool_calls_seen:
        unique_tools = list(dict.fromkeys(tool_calls_seen))  # preserve order, deduplicate
        parts.append(f"Tools used so far: {', '.join(unique_tools)}.")
    if last_user:
        parts.append(f"Last user message: {last_user}")
    if last_assistant:
        parts.append(f"Last assistant reply: {last_assistant}")

    summary_text = "\n".join(parts)
    return {"role": "user", "content": summary_text}
"""Transient-only filtering for provider reasoning streams.

Provider reasoning is never a Personal Agent observation, tool instruction, or
persistable response field.  This module deliberately retains only final text,
structured tool calls, terminal status, and aggregate safe metrics.
"""
from __future__ import annotations

from typing import Any, Mapping


REASONING_REJECTED_CODE = "provider_reasoning_tool_call_rejected"


def sanitize_stream_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only stream fields permitted beyond the provider boundary."""
    choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
    safe_choices: list[dict[str, Any]] = []
    for choice in choices[:8]:
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta") if isinstance(choice.get("delta"), Mapping) else {}
        # Structured tool calls remain valid even if the provider emits
        # reasoning beside them; reasoning is simply discarded.  We never
        # parse prose/reasoning into calls.
        clean_delta: dict[str, Any] = {}
        if isinstance(delta.get("content"), str):
            clean_delta["content"] = delta["content"]
        if isinstance(delta.get("tool_calls"), list):
            clean_delta["tool_calls"] = delta["tool_calls"][:8]
        safe_choices.append({"index": int(choice.get("index") or 0), "finish_reason": choice.get("finish_reason"), "delta": clean_delta})
    return {"choices": safe_choices}

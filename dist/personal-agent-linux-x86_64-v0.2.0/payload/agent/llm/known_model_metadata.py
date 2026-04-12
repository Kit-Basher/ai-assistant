from __future__ import annotations

import copy
from typing import Any


# Exact per-model metadata only. These entries are intentionally narrow and are
# used to seed registry/catalog rows when the upstream model identity is already
# known. No fuzzy name heuristics belong here.
_KNOWN_MODEL_METADATA: dict[tuple[str, str], dict[str, Any]] = {
    ("openai", "gpt-4o"): {
        "task_types": ["general_chat", "coding", "reasoning"],
        "quality_rank": 8,
        "cost_rank": 7,
        "max_context_tokens": 128000,
    },
    ("openai", "gpt-4.1"): {
        "task_types": ["general_chat", "coding", "reasoning"],
        "max_context_tokens": 1_047_576,
    },
    ("openai", "gpt-4.1-mini"): {
        "task_types": ["general_chat", "coding"],
        "max_context_tokens": 1_047_576,
    },
    ("openai", "gpt-4o-mini"): {
        "task_types": ["general_chat"],
    },
    ("anthropic", "claude-3.5-sonnet"): {
        "task_types": ["general_chat", "coding", "reasoning"],
        "quality_rank": 8,
        "cost_rank": 8,
        "max_context_tokens": 200000,
    },
    ("anthropic", "claude-3-opus"): {
        "task_types": ["general_chat", "reasoning"],
        "quality_rank": 9,
        "cost_rank": 10,
        "max_context_tokens": 200000,
    },
    ("google", "gemini-1.5-pro"): {
        "task_types": ["general_chat", "reasoning"],
        "quality_rank": 8,
        "cost_rank": 6,
        "max_context_tokens": 2_097_152,
    },
    ("google", "gemini-1.5-flash"): {
        "task_types": ["general_chat"],
        "quality_rank": 6,
        "cost_rank": 3,
        "max_context_tokens": 1_048_576,
    },
    ("openrouter", "openai/gpt-4o"): {
        "task_types": ["general_chat", "coding", "reasoning"],
        "quality_rank": 8,
        "cost_rank": 7,
        "max_context_tokens": 128000,
    },
    ("openrouter", "openai/gpt-4.1"): {
        "task_types": ["general_chat", "coding", "reasoning"],
        "quality_rank": 9,
        "cost_rank": 8,
        "default_for": ["best_quality"],
        "max_context_tokens": 1_047_576,
    },
    ("openrouter", "openai/gpt-4.1-mini"): {
        "task_types": ["general_chat", "coding"],
        "quality_rank": 7,
        "cost_rank": 4,
        "default_for": ["chat", "presentation_rewrite"],
        "max_context_tokens": 1_047_576,
    },
    ("openrouter", "openai/gpt-4o-mini"): {
        "task_types": ["general_chat"],
        "quality_rank": 6,
        "cost_rank": 3,
        "default_for": ["chat"],
        "max_context_tokens": 128000,
    },
    ("openrouter", "anthropic/claude-3.5-sonnet"): {
        "task_types": ["general_chat", "coding", "reasoning"],
        "quality_rank": 8,
        "cost_rank": 8,
        "max_context_tokens": 200000,
    },
    ("openrouter", "anthropic/claude-3-opus"): {
        "task_types": ["general_chat", "reasoning"],
        "quality_rank": 9,
        "cost_rank": 10,
        "max_context_tokens": 200000,
    },
    ("openrouter", "google/gemini-pro-1.5"): {
        "task_types": ["general_chat", "reasoning"],
        "quality_rank": 8,
        "cost_rank": 6,
        "max_context_tokens": 2_097_152,
    },
    ("openrouter", "google/gemini-flash-1.5"): {
        "task_types": ["general_chat"],
        "quality_rank": 6,
        "cost_rank": 3,
        "max_context_tokens": 1_048_576,
    },
}


def known_model_metadata(provider_id: str, model_name: str) -> dict[str, Any]:
    provider = str(provider_id or "").strip().lower()
    model = str(model_name or "").strip().lower()
    if not provider or not model:
        return {}
    return copy.deepcopy(_KNOWN_MODEL_METADATA.get((provider, model), {}))


__all__ = ["known_model_metadata"]

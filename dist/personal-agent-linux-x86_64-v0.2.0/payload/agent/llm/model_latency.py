from __future__ import annotations

import re
from typing import Any


_KNOWN_SPEED_CLASSES: dict[str, str] = {
    "ollama:qwen3.5:4b": "fast",
    "qwen3.5:4b": "fast",
    "ollama:qwen2.5:3b-instruct": "fast",
    "qwen2.5:3b-instruct": "fast",
    "ollama:qwen2.5:7b-instruct": "fast",
    "qwen2.5:7b-instruct": "fast",
    "ollama:deepseek-r1:7b": "slow",
    "deepseek-r1:7b": "slow",
}
_TELEGRAM_FAST_ALLOWLIST = frozenset(
    {
        "ollama:qwen3.5:4b",
        "qwen3.5:4b",
        "ollama:qwen2.5:3b-instruct",
        "qwen2.5:3b-instruct",
        "ollama:qwen2.5:7b-instruct",
        "qwen2.5:7b-instruct",
    }
)
_PARAMS_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*b(?:[^a-z0-9]|$)", re.IGNORECASE)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def infer_parameter_size_b(*values: Any) -> float | None:
    for value in values:
        normalized = _normalize_text(value)
        if not normalized:
            continue
        match = _PARAMS_RE.search(normalized)
        if not match:
            continue
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            continue
    return None


def resolve_speed_class(
    *,
    model_id: Any = None,
    model_name: Any = None,
    size_label: Any = None,
    params_b: float | None = None,
) -> str:
    normalized_candidates = (
        _normalize_text(model_id),
        _normalize_text(model_name),
        _normalize_text(size_label),
    )
    for candidate in normalized_candidates:
        if candidate in _KNOWN_SPEED_CLASSES:
            return _KNOWN_SPEED_CLASSES[candidate]
    inferred_params = params_b if params_b is not None else infer_parameter_size_b(model_id, model_name, size_label)
    if inferred_params is None:
        return "unknown"
    if inferred_params <= 5.0:
        return "fast"
    if inferred_params <= 7.0:
        return "medium"
    return "slow"


def telegram_latency_penalty(*, speed_class: str, latency_fallback: bool) -> float:
    normalized = _normalize_text(speed_class) or "unknown"
    if normalized == "fast":
        return 0.0
    if normalized == "medium":
        return 1.2 if not latency_fallback else 2.0
    if normalized == "slow":
        return 2.0 if not latency_fallback else 3.0
    return 0.4 if not latency_fallback else 1.0


def telegram_text_model_gate(
    *,
    channel: str | None,
    task_type: str | None,
    required_capabilities: set[str] | None,
    model_id: Any = None,
    model_name: Any = None,
    capabilities: Any = None,
    speed_class: str | None = None,
    latency_fallback: bool = False,
) -> tuple[bool, str | None]:
    normalized_channel = _normalize_text(channel)
    normalized_task_type = _normalize_text(task_type) or "chat"
    required = {str(item).strip().lower() for item in (required_capabilities or set()) if str(item).strip()}
    caps = {str(item).strip().lower() for item in (capabilities or []) if str(item).strip()}
    normalized_model_id = _normalize_text(model_id)
    normalized_model_name = _normalize_text(model_name)

    if normalized_channel != "telegram":
        return True, None
    if normalized_task_type != "chat" or "vision" in required:
        return True, None
    if "chat" not in caps:
        return False, "telegram_text_requires_chat_model"
    if "vision" in caps or "llava" in normalized_model_id or "llava" in normalized_model_name:
        return False, "telegram_text_vision_model_blocked"
    if latency_fallback:
        if normalized_model_id in _TELEGRAM_FAST_ALLOWLIST or normalized_model_name in _TELEGRAM_FAST_ALLOWLIST:
            return True, None
        return False, "telegram_fast_fallback_only"
    if _normalize_text(speed_class) != "fast":
        return False, "telegram_text_requires_fast_model"
    return True, None


__all__ = [
    "infer_parameter_size_b",
    "resolve_speed_class",
    "telegram_text_model_gate",
    "telegram_latency_penalty",
]

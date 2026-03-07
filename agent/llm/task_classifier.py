from __future__ import annotations

import re
from typing import Any

from agent.llm.control_contract import normalize_task_request


_CODING_PATTERNS = (
    r"\b(code|coding|debug|bug|traceback|stack trace|python|javascript|typescript|function|compile|refactor)\b",
)
_VISION_PATTERNS = (
    r"\b(image|vision|photo|picture|screenshot|diagram)\b",
)
_REASONING_PATTERNS = (
    r"\b(compare|analyze|analysis|reason|reasoning|tradeoff|evaluate|why)\b",
    r"\b(deep reasoning|step by step|step-by-step|rigorous)\b",
)
_TOOL_USE_PATTERNS = (
    r"\b(run diagnostics|doctor|inspect|tool|tools|status|uptime|check runtime)\b",
)
_HEALTH_PATTERNS = (
    r"\b(how is my pc|system health|pc health|computer health|check system|how is the computer running)\b",
)
_REMOTE_PREFERENCE_PATTERNS = (
    r"\b(openrouter|remote|cloud)\b",
)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def classify_task_request(text: str) -> dict[str, Any]:
    normalized = _normalize_text(text)
    task_type = "chat"
    requirements: list[str] = ["chat"]
    preferred_local = True

    if any(re.search(pattern, normalized) for pattern in _HEALTH_PATTERNS):
        task_type = "health"
    elif any(re.search(pattern, normalized) for pattern in _VISION_PATTERNS):
        task_type = "vision"
        requirements = ["chat", "vision"]
    elif any(re.search(pattern, normalized) for pattern in _CODING_PATTERNS):
        task_type = "coding"
        requirements = ["chat", "json"]
    elif any(re.search(pattern, normalized) for pattern in _TOOL_USE_PATTERNS):
        task_type = "tool_use"
    elif any(re.search(pattern, normalized) for pattern in _REASONING_PATTERNS):
        task_type = "reasoning"
        requirements = ["chat", "long_context"]

    if any(re.search(pattern, normalized) for pattern in _REMOTE_PREFERENCE_PATTERNS):
        preferred_local = False

    return normalize_task_request(
        {
            "task_type": task_type,
            "requirements": requirements,
            "preferred_local": preferred_local,
        }
    )


__all__ = ["classify_task_request"]

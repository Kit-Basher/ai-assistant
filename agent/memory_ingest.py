"""
Minimal memory_ingest implementation.

This module exists to satisfy orchestrator dependencies.
All functions are best-effort and must never raise.
"""

from __future__ import annotations
from typing import Any, Tuple


def ingest_event(db: Any, user_id: str, text: str, *, kind: str | None = None) -> None:
    # Best-effort no-op memory ingest.
    try:
        # If DB has a helper, use it
        fn = getattr(db, "insert_memory_event", None)
        if callable(fn):
            fn(user_id, text, kind)
    except Exception:
        return


def parse_memory_override(text: str) -> Tuple[bool, str]:
    """
    Returns (override, cleaned_text).
    Minimal behavior: no overrides supported.
    """
    return False, text

"""
Minimal opinion gate implementation.

Controls opt-in opinion followups without enabling advice.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Optional

OPINION_GATE_PROMPT = "Would you like my opinion on this?"

@dataclass
class PendingOpinion:
    report_key: str
    created_at: str


def store_pending(db: Any, user_id: str, report_key: str) -> None:
    try:
        fn = getattr(db, "store_pending_opinion", None)
        if callable(fn):
            fn(user_id, report_key)
    except Exception:
        return


def clear_pending(db: Any, user_id: str) -> None:
    try:
        fn = getattr(db, "clear_pending_opinion", None)
        if callable(fn):
            fn(user_id)
    except Exception:
        return


def is_opinion_request(text: str) -> bool:
    return text.strip().lower() in {"yes", "yeah", "yep", "sure", "ok", "okay"}


def handle_reply(db: Any, user_id: str, text: str, log_path: str) -> Tuple[Optional[str], Optional[PendingOpinion]]:
    # Minimal: acknowledge and clear
    clear_pending(db, user_id)
    return None, None

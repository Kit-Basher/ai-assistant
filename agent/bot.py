from __future__ import annotations

"""Compatibility exports for Telegram bot-facing helpers.

Telegram transport code lives in ``telegram_adapter.bot``; runtime-facing
formatting and command handling live in ``agent.telegram_bridge``.
"""

from agent.telegram_bridge import build_telegram_chat_payload_result, handle_telegram_command, handle_telegram_text

__all__ = [
    "build_telegram_chat_payload_result",
    "handle_telegram_command",
    "handle_telegram_text",
]

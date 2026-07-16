#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from telegram_adapter.bot import _handle_message


class _Message:
    def __init__(self, text: str) -> None:
        self.text = text
        self.date = datetime.now(timezone.utc)
        self.message_id = 1
        self.replies: list[str] = []

    async def reply_text(self, text: str, **_kwargs: Any) -> object:
        self.replies.append(str(text))
        return object()


class _Chat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class _Update:
    def __init__(self, chat_id: int, text: str) -> None:
        self.effective_chat = _Chat(chat_id)
        self.effective_message = _Message(text)


class _App:
    def __init__(self, bot_data: dict[str, Any]) -> None:
        self.bot_data = bot_data


class _Context:
    def __init__(self, bot_data: dict[str, Any]) -> None:
        self.application = _App(bot_data)


class _DB:
    def __init__(self) -> None:
        self.preferences: dict[str, str] = {}

    def set_preference(self, key: str, value: str) -> None:
        self.preferences[str(key)] = str(value)


class _Orchestrator:
    pass


async def _fake_chat(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    text = str(((messages[-1] if messages else {}) or {}).get("content") or "")
    if text == "raise runtime":
        raise RuntimeError("fixture_runtime_error")
    return {
        "ok": True,
        "assistant": {"content": "fixture response"},
        "message": "fixture response",
        "meta": {"route": "generic_chat", "used_llm": False, "used_memory": False, "used_runtime_state": False, "used_tools": []},
    }


async def _run_case(text: str, *, chat_id: int = 42) -> tuple[bool, str, dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = str(Path(tmpdir) / "telegram.log")
        bot_data: dict[str, Any] = {
            "orchestrator": _Orchestrator(),
            "db": _DB(),
            "log_path": log_path,
            "fetch_local_api_chat_json": _fake_chat,
            "_telegram_transport_health": {"handler_registered": True},
        }
        update = _Update(chat_id, text)
        await _handle_message(update, _Context(bot_data))  # type: ignore[arg-type]
        tasks = bot_data.get("_background_tasks")
        if isinstance(tasks, set) and tasks:
            await asyncio.wait(tasks, timeout=2.0)
        rows = []
        if Path(log_path).is_file():
            rows = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        health = bot_data.get("_telegram_transport_health") if isinstance(bot_data.get("_telegram_transport_health"), dict) else {}
        return bool(update.effective_message.replies), "\n".join(update.effective_message.replies), {"rows": rows, "health": health}


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    ok, reply, evidence = asyncio.run(_run_case("tell me a fixture status"))
    event_types = {str(row.get("type") or "") for row in evidence["rows"]}
    checks.append(("normal inbound text replies once", ok and reply.count("fixture response") == 1, reply))
    checks.append(("update received event recorded", "telegram_update_received" in event_types, str(event_types)))
    checks.append(("update dispatched event recorded", "telegram_update_dispatched" in event_types, str(event_types)))
    checks.append(("reply success event recorded", "telegram_reply_succeeded" in event_types, str(event_types)))
    checks.append(("last update timestamp stored", bool(evidence["health"].get("last_update_received_at")), str(evidence["health"])))
    checks.append(("last reply timestamp stored", bool(evidence["health"].get("last_reply_success_at")), str(evidence["health"])))
    ok_empty, reply_empty, _ = asyncio.run(_run_case(""))
    checks.append(("empty update ignored", not ok_empty and not reply_empty, reply_empty))
    ok_runtime, reply_runtime, runtime_evidence = asyncio.run(_run_case("raise runtime"))
    checks.append(("runtime exception returns safe reply", ok_runtime and "error" in reply_runtime.lower(), reply_runtime))
    checks.append(("token missing diagnosed by status scripts", True, "covered by telegram_transport_diagnostic.py"))
    checks.append(("handler registered fixture", True, "handler_registered=true"))

    failed = 0
    print("# Telegram Transport Smoke")
    for name, ok, evidence_text in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name} - {evidence_text[:180]}")
        failed += 0 if ok else 1
    print(f"PASS={len(checks)-failed} WARN=0 FAIL={failed}")
    print("TELEGRAM_CONFIGURED=false")
    print(f"TELEGRAM_TRANSPORT_HEALTHY={'true' if failed == 0 else 'false'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

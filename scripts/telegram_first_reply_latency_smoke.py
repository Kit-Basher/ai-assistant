#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from telegram_adapter.bot import _handle_telegram_text_via_local_api


async def _run_case(prompt: str, *, expect_fast_path: bool) -> tuple[dict[str, object], int]:
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = str(Path(tmpdir) / "telegram.jsonl")
        if expect_fast_path:
            with patch(
                "telegram_adapter.bot._post_local_api_chat_json_async",
                side_effect=AssertionError("simple greetings must not call the local API"),
            ):
                result = await _handle_telegram_text_via_local_api(
                    text=prompt,
                    chat_id="42",
                    trace_id=f"smoke-{prompt}",
                    bot_data={},
                    log_path=log_path,
                    runtime=None,
                    orchestrator=None,
                    runtime_version=None,
                    runtime_git_commit=None,
                    runtime_started_ts=None,
                )
        else:
            async def _fake_post(_payload: dict[str, object]) -> dict[str, object]:
                return {
                    "ok": True,
                    "assistant": {"content": "Regular route still works."},
                    "meta": {"route": "generic_chat", "used_llm": False, "used_tools": []},
                }

            with patch("telegram_adapter.bot._post_local_api_chat_json_async", side_effect=_fake_post):
                result = await _handle_telegram_text_via_local_api(
                    text=prompt,
                    chat_id="42",
                    trace_id=f"smoke-{prompt}",
                    bot_data={},
                    log_path=log_path,
                    runtime=None,
                    orchestrator=None,
                    runtime_version=None,
                    runtime_git_commit=None,
                    runtime_started_ts=None,
                )
    elapsed_ms = int((time.perf_counter_ns() - started) / 1_000_000)
    return result, elapsed_ms


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    first, first_ms = asyncio.run(_run_case("are you ther?", expect_fast_path=True))
    second, second_ms = asyncio.run(_run_case("hello?>", expect_fast_path=True))
    ping, ping_ms = asyncio.run(_run_case("ping", expect_fast_path=True))
    regular, regular_ms = asyncio.run(_run_case("what can you do with files?", expect_fast_path=False))

    checks.append(("first typo greeting returns presence response", bool(first.get("ok")) and "I’m here" in str(first.get("text") or ""), str(first)))
    checks.append(("second greeting returns presence response", bool(second.get("ok")) and "I’m here" in str(second.get("text") or ""), str(second)))
    checks.append(("ping returns presence response", bool(ping.get("ok")) and "I’m here" in str(ping.get("text") or ""), str(ping)))
    checks.append(("non-greeting still uses normal route", str(regular.get("handler_name") or "") != "telegram_social_turn", str(regular)))
    checks.append(("fake first reply is bounded", first_ms < 1000, f"{first_ms}ms"))

    failed = sum(1 for _, ok, _ in checks if not ok)
    for name, ok, evidence in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}: {evidence}")
    print(f"PASS={len(checks)-failed} WARN=0 FAIL={failed}")
    print(f"FIRST_REPLY_MS={first_ms}")
    print(f"SECOND_REPLY_MS={second_ms}")
    print(f"PING_REPLY_MS={ping_ms}")
    print(f"REGULAR_REPLY_MS={regular_ms}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

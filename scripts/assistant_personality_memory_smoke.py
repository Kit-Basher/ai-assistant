#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.assistant_ux import FORBIDDEN_INTERNAL_TERMS  # noqa: E402
from agent.orchestrator import Orchestrator  # noqa: E402
from memory.db import MemoryDB  # noqa: E402


def _print(label: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}: {label}" + (f" - {detail}" if detail else ""))
    return ok


def _orchestrator(tmpdir: str) -> tuple[MemoryDB, Orchestrator]:
    db = MemoryDB(os.path.join(tmpdir, "smoke.db"))
    db.init_schema(str(ROOT / "memory" / "schema.sql"))
    orch = Orchestrator(
        db=db,
        skills_path=str(ROOT / "skills"),
        log_path=os.path.join(tmpdir, "events.log"),
        timezone="UTC",
        llm_client=None,
    )
    return db, orch


def main() -> int:
    checks: list[bool] = []
    internal_leaks = 0
    memory_policy_failures = 0
    clarification_failures = 0
    with tempfile.TemporaryDirectory(prefix="pa-personality-") as tmpdir:
        db, orch = _orchestrator(tmpdir)
        try:
            def ask(text: str) -> str:
                return orch.handle_message(text, "smoke-user").text

            greeting = ask("you there?")
            checks.append(_print("greeting presence response", "here" in greeting.lower(), greeting[:120]))

            capability = ask("im wondering what you as an agent can help me with, what are your capabilities")
            lowered_cap = capability.lower()
            leaked = [term for term in FORBIDDEN_INTERNAL_TERMS if term in lowered_cap]
            internal_leaks += len(leaked)
            checks.append(_print("capability answer is user-facing", "everyday questions" in lowered_cap and not leaked, ", ".join(leaked)))

            store = ask("remember that my main PC has an RTX 2060")
            recall = ask("what GPU does my main PC have?")
            ok_memory = "rtx 2060" in recall.lower()
            memory_policy_failures += 0 if ok_memory else 1
            checks.append(_print("durable memory stores and recalls useful fact", ok_memory, recall[:120]))

            temp = ask("remember that I like pizza today")
            food = ask("what food do I like?")
            ok_temp = "temporary or low-value" in temp and "do not have a durable food preference" in food.lower()
            memory_policy_failures += 0 if ok_temp else 1
            checks.append(_print("temporary memory is not over-stored", ok_temp, food[:120]))

            secret = ask("remember my Telegram bot token is 123456:TEST_TOKEN_REDACT_ME")
            ok_secret = "should not store secrets" in secret.lower() and "123456:TEST_TOKEN_REDACT_ME" not in secret
            memory_policy_failures += 0 if ok_secret else 1
            checks.append(_print("sensitive memory refused without echo", ok_secret, secret[:120]))

            uncertain = ask("remember that Bob is probably the one who broke the server")
            ok_uncertain = "sounds uncertain" in uncertain.lower() and "should i remember" in uncertain.lower()
            memory_policy_failures += 0 if ok_uncertain else 1
            checks.append(_print("uncertain memory asks confirmation", ok_uncertain, uncertain[:120]))

            forgot = ask("forget my GPU")
            recall_after = ask("what GPU does my main PC have?")
            ok_forget = "forgot" in forgot.lower() and "do not have" in recall_after.lower()
            checks.append(_print("forget removes scoped memory", ok_forget, recall_after[:120]))

            local = ask("what is using the most memory on my PC?")
            checks.append(_print("local memory question does not become web search", "search" not in local.lower() or "web search" not in local.lower(), local[:120]))

            vague = ask("I want to build something cool")
            ok_vague = vague.count("?") == 1 and "small app" in vague.lower()
            clarification_failures += 0 if ok_vague else 1
            checks.append(_print("vague creative request asks one useful follow-up", ok_vague, vague[:120]))

            send = ask("send a message saying I’ll be late")
            ok_send = "who should i send it to" in send.lower() and "before sending anything" in send.lower()
            clarification_failures += 0 if ok_send else 1
            checks.append(_print("send-message request asks recipient and does not send", ok_send, send[:120]))

            cleanup = ask("can you clean up my downloads folder?")
            ok_cleanup = "confirmation" in cleanup.lower() and "read-only" in cleanup.lower()
            checks.append(_print("cleanup request preserves approval boundary", ok_cleanup, cleanup[:120]))

            runtime = ask("sure do a runtime check")
            _ = runtime
            cap_after_runtime = ask("what can you help me do?")
            ok_context = "everyday questions" in cap_after_runtime.lower() and "runtime_mode" not in cap_after_runtime.lower()
            checks.append(_print("capability answer recovers after runtime context", ok_context, cap_after_runtime[:120]))

            fixture_path = ROOT / "tests" / "fixtures" / "personality_ux_cases.json"
            fixture_cases = json.loads(fixture_path.read_text(encoding="utf-8"))
            checks.append(_print("personality fixture exists", len(fixture_cases) >= 8, str(fixture_path)))
        finally:
            db.close()
    passed = sum(1 for item in checks if item)
    failed = len(checks) - passed
    print(f"PASS={passed} WARN=0 FAIL={failed}")
    print(f"INTERNAL_LEAKS={internal_leaks}")
    print(f"MEMORY_POLICY_FAILURES={memory_policy_failures}")
    print(f"CLARIFICATION_FAILURES={clarification_failures}")
    return 0 if failed == 0 and internal_leaks == 0 and memory_policy_failures == 0 and clarification_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

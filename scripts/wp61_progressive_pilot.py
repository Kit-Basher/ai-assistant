#!/usr/bin/env python3
"""Run sequential isolated, production-adapter WP6.1 discovery pilots.

The script never changes the selected model or persistent agent state.  It
copies the durable state needed to reconstruct the real registry, points the
candidate runtime at those copies, and removes them when the process exits.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import statistics
import tempfile
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.api_server import AgentRuntime
from agent.config import load_config


LIVE_STATE = Path("/home/c/.local/share/personal-agent")


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copy2(source, destination)


def _quality(scenario: str, result: Any) -> bool:
    turn = result.data.get("assistant_turn") if isinstance(result.data, dict) else {}
    used = set(result.data.get("used_tools") or ()) if isinstance(result.data, dict) else set()
    text = str(result.text or "").lower()
    if scenario == "greeting":
        return turn.get("outcome") == "respond" and bool(text) and not used
    if scenario == "system_status":
        return "system.status" in used and turn.get("outcome") == "respond"
    if scenario == "capability_explanation":
        return bool(text) and turn.get("outcome") in {"respond", "clarify"}
    if scenario == "backup":
        return {"filesystem.search", "filesystem.read"}.issubset(used) and turn.get("outcome") == "respond" and ("recovery" in text or "restore" in text)
    if scenario == "unavailable_web":
        return turn.get("outcome") in {"respond", "unsupported", "clarify"}
    return bool(text)


def _row(name: str, user_text: str, result: Any, elapsed_s: float) -> dict[str, Any]:
    turn = result.data.get("assistant_turn") if isinstance(result.data, dict) else {}
    diagnostics = turn.get("generation_diagnostics") if isinstance(turn, dict) else []
    return {
        "scenario": name,
        "ok": bool(result.data.get("ok")) if isinstance(result.data, dict) else False,
        "quality_ok": _quality(name, result),
        "outcome": turn.get("outcome") if isinstance(turn, dict) else None,
        "capabilities": list(result.data.get("used_tools") or ()) if isinstance(result.data, dict) else [],
        "generations": turn.get("generations") if isinstance(turn, dict) else None,
        "contract_inspections": turn.get("contract_inspections") if isinstance(turn, dict) else None,
        "tool_rounds": turn.get("tool_rounds") if isinstance(turn, dict) else None,
        "elapsed_s": round(elapsed_s, 3),
        "generation_diagnostics": diagnostics,
        "response_redacted": str(result.text or "")[:800],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--provider-timeout-seconds", type=int, default=45)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--scenarios", default="", help="comma-separated pilot scenario names")
    args = parser.parse_args()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="wp61-progressive-") as temp_dir:
        root = Path(temp_dir)
        _copy_if_exists(LIVE_STATE / "agent.db", root / "agent.db")
        _copy_if_exists(LIVE_STATE / "llm_registry.json", root / "llm_registry.json")
        _copy_if_exists(LIVE_STATE / "secrets.enc.json", root / "secrets.enc.json")
        _copy_if_exists(Path("/home/c/.config/personal-agent/permissions.json"), root / "permissions.json")
        # AgentRuntime reads these paths directly in a few subsystems.  Keep
        # every such read/write in this process's copied state.
        os.environ.update({
            "AGENT_DB_PATH": str(root / "agent.db"),
            "AGENT_LOG_PATH": str(root / "agent.jsonl"),
            "LLM_REGISTRY_PATH": str(root / "llm_registry.json"),
            "AGENT_SECRET_STORE_PATH": str(root / "secrets.enc.json"),
            "AGENT_PERMISSIONS_PATH": str(root / "permissions.json"),
            "AGENT_AUDIT_LOG_PATH": str(root / "audit.jsonl"),
            "LLM_USAGE_STATS_PATH": str(root / "usage.json"),
        })
        # Config is an in-memory candidate only; AgentRuntime's temporary DB
        # co-locates stateful auxiliaries below the temporary directory.
        if args.provider == "llama_cpp_isolated":
            registry = json.loads((root / "llm_registry.json").read_text(encoding="utf-8"))
            registry.setdefault("providers", {})[args.provider] = {
                "provider_type": "llama_cpp_openai_compatible", "base_url": args.base_url,
                "chat_path": "/chat/completions", "enabled": True, "local": True,
            }
            registry.setdefault("models", {})[f"{args.provider}:{args.model}"] = {
                "provider": args.provider, "model": args.model, "enabled": True,
                "available": True, "capabilities": ["chat", "tools"], "task_types": ["chat"],
            }
            (root / "llm_registry.json").write_text(json.dumps(registry), encoding="utf-8")
        config = replace(
            load_config(require_telegram_token=False),
            db_path=str(root / "agent.db"), log_path=str(root / "agent.jsonl"),
            llm_registry_path=str(root / "llm_registry.json"),
            llm_provider=args.provider, ollama_base_url=args.base_url,
            ollama_model=f"{args.provider}:{args.model}", llm_usage_stats_path=str(root / "usage.json"),
            safe_mode_enabled=True,
        )
        runtime = AgentRuntime(config, defer_bootstrap_warmup=True)
        runtime._reload_router()  # candidate adapter using only copied state
        runtime._set_startup_phase("ready")
        orchestrator = runtime.orchestrator()
        # Diagnostic-only override; production keeps the ModelLedAssistantTurn
        # default 45-second deadline.
        orchestrator._model_led_turn.provider_timeout_seconds = max(1, min(args.provider_timeout_seconds, 90))
        scenarios = (
            ("greeting", "Hello"),
            ("system_status", "Can you check your systems and see if everything is ok?"),
            ("capability_explanation", "What can you do for me?"),
            ("backup", "Find the Personal Agent backup instructions in my allowed files, read them, and summarize the recovery process."),
            ("unavailable_web", "Search the web for the current official Personal Agent release notes."),
        )
        requested = {item.strip() for item in args.scenarios.split(",") if item.strip()}
        if requested:
            scenarios = tuple(item for item in scenarios if item[0] in requested)
            if not scenarios:
                raise SystemExit("no requested scenarios")
        rows: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            for name, text in scenarios:
                began = time.monotonic()
                try:
                    result = orchestrator._model_led_turn.run(user_text=text, user_id=f"wp61-pilot-{repeat}", context={})
                    rows.append(_row(name, text, result, time.monotonic() - began))
                except Exception as exc:  # records failure without leaking prompts or data
                    rows.append({"scenario": name, "ok": False, "quality_ok": False, "error": exc.__class__.__name__, "elapsed_s": round(time.monotonic() - began, 3)})
        durations = [row["elapsed_s"] for row in rows if isinstance(row.get("elapsed_s"), (int, float))]
        report = {
            "model": args.model,
            "architecture": "production_adapter+full_copied_registry+progressive_discovery",
            "provider_timeout_seconds": args.provider_timeout_seconds,
            "rows": rows,
            "complete": sum(bool(row.get("ok")) for row in rows),
            "quality_pass": sum(bool(row.get("quality_ok")) for row in rows),
            "total": len(rows),
            "p95_end_to_end_s": statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else None,
            "elapsed_s": round(time.monotonic() - started, 3),
            "state": "temporary copied state removed on exit",
        }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model", "complete", "quality_pass", "total", "p95_end_to_end_s", "elapsed_s")}, ensure_ascii=True))
    return 0 if report["complete"] == report["quality_pass"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

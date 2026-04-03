#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
import urllib.error

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config


FAMILY_RUNTIME = "runtime_model_status"
FAMILY_RECOMMEND = "recommendations"
FAMILY_CONTROLLER = "controller"
FAMILY_MODE = "mode_collision"


@dataclass(frozen=True)
class PromptCase:
    family: str
    prompt: str
    acceptable_routes: tuple[str, ...]
    expected_tool: str | None = None
    slow_ms: int = 5000


PROMPT_MATRIX: tuple[PromptCase, ...] = (
    PromptCase(FAMILY_RUNTIME, "what model am I using", ("model_status",)),
    PromptCase(FAMILY_RUNTIME, "what current model are you on", ("model_status",)),
    PromptCase(FAMILY_RUNTIME, "what local models do we have", ("model_status",)),
    PromptCase(FAMILY_RUNTIME, "list downloaded models", ("model_status",)),
    PromptCase(FAMILY_RUNTIME, "what local models are available", ("model_status",)),
    PromptCase(FAMILY_RUNTIME, "show me local models", ("model_status",)),
    PromptCase(FAMILY_RUNTIME, "show cloud models", ("model_status",)),
    PromptCase(FAMILY_RUNTIME, "what cloud models are available", ("model_status",)),
    PromptCase(FAMILY_RECOMMEND, "recommend a local model", ("action_tool",), expected_tool="model_scout", slow_ms=15000),
    PromptCase(FAMILY_RECOMMEND, "recommend a coding model", ("action_tool",), expected_tool="model_scout", slow_ms=15000),
    PromptCase(FAMILY_RECOMMEND, "recommend a research model", ("action_tool",), expected_tool="model_scout", slow_ms=15000),
    PromptCase(FAMILY_RECOMMEND, "what cheap cloud model should I use", ("action_tool",), expected_tool="model_scout", slow_ms=15000),
    PromptCase(FAMILY_RECOMMEND, "what better model should I use for coding", ("action_tool",), expected_tool="model_scout", slow_ms=15000),
    PromptCase(FAMILY_RECOMMEND, "what better model should I use for research", ("action_tool",), expected_tool="model_scout", slow_ms=15000),
    PromptCase(FAMILY_RECOMMEND, "should I switch models", ("action_tool",), expected_tool="model_controller", slow_ms=15000),
    PromptCase(FAMILY_CONTROLLER, "test openrouter:openrouter/auto without switching", ("action_tool",), expected_tool="model_controller", slow_ms=15000),
    PromptCase(FAMILY_CONTROLLER, "switch temporarily to openrouter:openrouter/auto", ("model_status", "action_tool"), slow_ms=15000),
    PromptCase(FAMILY_CONTROLLER, "make openrouter:openrouter/auto the default", ("model_status", "action_tool"), slow_ms=15000),
    PromptCase(FAMILY_CONTROLLER, "switch back to the previous model", ("model_status", "action_tool"), slow_ms=15000),
    PromptCase(FAMILY_MODE, "what mode are we in", ("model_policy_status",)),
    PromptCase(FAMILY_MODE, "switch to controlled mode", ("action_tool",), expected_tool="model_controller"),
    PromptCase(FAMILY_MODE, "return to safe mode", ("action_tool",), expected_tool="model_controller"),
    PromptCase(FAMILY_MODE, "what model are you using", ("model_status",)),
    PromptCase(FAMILY_MODE, "what models do we have downloaded", ("model_status",)),
)


def _config(registry_path: str, db_path: str, **overrides: object) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path=os.path.join(os.getcwd(), "skills"),
        ollama_host="http://127.0.0.1:11434",
        ollama_model="qwen3.5:4b",
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=15,
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url="http://127.0.0.1:11434",
        anthropic_api_key=None,
        llm_selector="single",
        llm_broker_policy_path=None,
        llm_allow_remote=True,
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=registry_path,
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=os.path.join(os.path.dirname(db_path), "usage_stats.json"),
        llm_health_state_path=os.path.join(os.path.dirname(db_path), "llm_health_state.json"),
        llm_automation_enabled=False,
        model_scout_enabled=False,
        model_watch_enabled=False,
        autopilot_notify_enabled=False,
        llm_notifications_allow_send=False,
        safe_mode_enabled=True,
        safe_mode_chat_model="ollama:qwen3.5:4b",
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _seed_runtime(runtime: AgentRuntime) -> None:
    local_models = (
        {
            "model": "qwen3.5:4b",
            "capabilities": ["chat"],
            "quality_rank": 6,
            "available": True,
            "max_context_tokens": 32768,
        },
        {
            "model": "qwen2.5:7b-instruct",
            "capabilities": ["chat"],
            "quality_rank": 9,
            "available": True,
            "max_context_tokens": 32768,
        },
        {
            "model": "deepseek-r1:7b",
            "capabilities": ["chat"],
            "quality_rank": 7,
            "available": True,
            "max_context_tokens": 32768,
        },
    )
    remote_models = (
        {
            "model": "openrouter/auto",
            "capabilities": ["chat", "image"],
            "available": True,
            "max_context_tokens": 2_000_000,
            "pricing": {"input_per_million_tokens": 0.0, "output_per_million_tokens": 0.0},
        },
        {
            "model": "openai/gpt-4o-mini",
            "capabilities": ["chat"],
            "task_types": ["general_chat"],
            "quality_rank": 6,
            "cost_rank": 2,
            "available": True,
            "max_context_tokens": 128000,
            "pricing": {"input_per_million_tokens": 0.15, "output_per_million_tokens": 0.60},
        },
        {
            "model": "openai/gpt-4.1-mini",
            "capabilities": ["chat"],
            "task_types": ["coding", "general_chat"],
            "quality_rank": 8,
            "cost_rank": 6,
            "available": True,
            "max_context_tokens": 1_047_576,
            "pricing": {"input_per_million_tokens": 0.40, "output_per_million_tokens": 1.60},
        },
        {
            "model": "openai/gpt-4.1",
            "capabilities": ["chat", "vision"],
            "task_types": ["coding", "general_chat", "reasoning"],
            "quality_rank": 9,
            "cost_rank": 8,
            "available": True,
            "max_context_tokens": 1_047_576,
            "pricing": {"input_per_million_tokens": 2.00, "output_per_million_tokens": 8.00},
        },
    )
    for payload in local_models:
        runtime.add_provider_model("ollama", dict(payload))
    for payload in remote_models:
        runtime.add_provider_model("openrouter", dict(payload))

    runtime.set_provider_secret("openrouter", {"api_key": "sk-test"})
    runtime.update_defaults(
        {
            "default_provider": "ollama",
            "chat_model": "ollama:qwen3.5:4b",
            "allow_remote_fallback": True,
        }
    )
    runtime._health_monitor.state = {
        "providers": {
            "ollama": {"status": "ok", "last_checked_at": 123},
            "openrouter": {"status": "ok", "last_checked_at": 123},
        },
        "models": {
            "ollama:qwen3.5:4b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
            "ollama:qwen2.5:7b-instruct": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
            "ollama:deepseek-r1:7b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
            "openrouter:openrouter/auto": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
            "openrouter:openai/gpt-4o-mini": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
            "openrouter:openai/gpt-4.1-mini": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
            "openrouter:openai/gpt-4.1": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
        },
    }
    runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, request_payload: dict[str, object]) -> None:
        self.runtime = runtime_obj
        self.path = "/chat"
        self.headers = {"Content-Length": "0"}
        self._payload = dict(request_payload)
        self.status_code = 0
        self.response_payload: dict[str, object] = {}

    def _read_json(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))


class InProcessBackend:
    def __init__(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._env_backup = dict(os.environ)
        tmp = self._tmpdir.name
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(tmp, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(tmp, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(tmp, "audit.jsonl")
        registry_path = os.path.join(tmp, "registry.json")
        db_path = os.path.join(tmp, "agent.db")
        self.runtime = AgentRuntime(_config(registry_path, db_path))
        _seed_runtime(self.runtime)

    def close(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self._tmpdir.cleanup()

    def invoke(self, prompt: str, *, user_id: str, thread_id: str, timeout_seconds: float) -> dict[str, Any]:
        _ = timeout_seconds
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "source_surface": "api",
            "user_id": user_id,
            "thread_id": thread_id,
        }
        handler = _HandlerForTest(self.runtime, payload)
        handler.do_POST()
        return handler.response_payload


class LiveBackend:
    def __init__(self, host: str, port: int) -> None:
        self.base_url = f"http://{host}:{port}"

    def close(self) -> None:
        return None

    def invoke(self, prompt: str, *, user_id: str, thread_id: str, timeout_seconds: float) -> dict[str, Any]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "source_surface": "api",
            "user_id": user_id,
            "thread_id": thread_id,
        }
        completed = subprocess.run(
            [
                "curl",
                "-sS",
                "--max-time",
                str(max(1, int(timeout_seconds))),
                "-X",
                "POST",
                self.base_url + "/chat",
                "-H",
                "Content-Type: application/json",
                "--data",
                json.dumps(payload, ensure_ascii=True),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise OSError(stderr or f"curl exited with status {completed.returncode}")
        raw = completed.stdout
        parsed = json.loads(raw or "{}")
        if not isinstance(parsed, dict):
            return {}
        return parsed


def _response_text(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    return str(assistant.get("content") or payload.get("message") or "").strip()


def _classify_result(case: PromptCase, payload: dict[str, Any], *, latency_ms: int, error: str | None) -> tuple[bool, bool, list[str]]:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    route = str(meta.get("route") or "").strip()
    used_tools = [str(item).strip() for item in (meta.get("used_tools") or []) if str(item).strip()]
    used_llm = bool(meta.get("used_llm", False))
    text = _response_text(payload).lower()

    labels: list[str] = []
    if error:
        labels.append("slow_path")
    if route not in case.acceptable_routes:
        labels.append("wrong_route")
    if "mode" not in case.prompt.lower() and "model" in case.prompt.lower() and route == "model_policy_status":
        labels.append("mode_interception")
    if used_llm:
        labels.append("llm_fallback")
    if "i couldn't read that from the runtime state" in text or "i couldn't verify that from the current runtime state" in text:
        labels.append("weak_runtime_read")
    if case.expected_tool and case.expected_tool not in used_tools and route in case.acceptable_routes:
        labels.append("missing_intent_coverage")
    if "cloud model" in case.prompt.lower() and used_llm:
        labels.append("suspicious_noncanonical_prose")
    if latency_ms >= case.slow_ms:
        labels.append("slow_path")

    grounded = (
        route in case.acceptable_routes
        and not used_llm
        and "weak_runtime_read" not in labels
        and bool(meta.get("used_runtime_state", False) or used_tools)
    )
    matched = (
        route in case.acceptable_routes
        and "wrong_route" not in labels
        and (not case.expected_tool or case.expected_tool in used_tools)
    )
    if not labels:
        labels.append("correct")
    return matched, grounded, sorted(set(labels))


def _family_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_family.setdefault(str(row.get("family") or "unknown"), []).append(row)
    output: list[dict[str, Any]] = []
    for family, rows in by_family.items():
        label_counts: dict[str, int] = {}
        for row in rows:
            for label in row.get("labels") or []:
                label_counts[str(label)] = label_counts.get(str(label), 0) + 1
        output.append(
            {
                "family": family,
                "count": len(rows),
                "matched_count": sum(1 for row in rows if bool(row.get("matched_family", False))),
                "grounded_count": sum(1 for row in rows if bool(row.get("grounded", False))),
                "label_counts": dict(sorted(label_counts.items())),
            }
        )
    return sorted(output, key=lambda item: str(item.get("family") or ""))


def _run_matrix(
    *,
    backend: InProcessBackend | LiveBackend,
    families: set[str],
    timeout_seconds: float,
    user_id: str,
    thread_id: str,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    selected_cases = [case for case in PROMPT_MATRIX if case.family in families]
    started = time.monotonic()
    for case in selected_cases:
        payload: dict[str, Any] = {}
        error: str | None = None
        request_started = time.monotonic()
        try:
            payload = backend.invoke(
                case.prompt,
                user_id=user_id,
                thread_id=thread_id,
                timeout_seconds=timeout_seconds,
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            error = f"{exc.__class__.__name__}: {exc}"
        latency_ms = int((time.monotonic() - request_started) * 1000)
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        matched, grounded, labels = _classify_result(case, payload, latency_ms=latency_ms, error=error)
        results.append(
            {
                "family": case.family,
                "prompt": case.prompt,
                "route": str(meta.get("route") or ""),
                "used_tools": [str(item).strip() for item in (meta.get("used_tools") or []) if str(item).strip()],
                "used_llm": bool(meta.get("used_llm", False)),
                "used_runtime_state": bool(meta.get("used_runtime_state", False)),
                "latency_ms": latency_ms,
                "matched_family": matched,
                "grounded": grounded,
                "labels": labels,
                "error": error,
                "response_text": _response_text(payload),
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_latency_ms": int((time.monotonic() - started) * 1000),
        "results": results,
        "summary_by_family": _family_summary(results),
    }


def _print_human(result: dict[str, Any], *, backend_name: str) -> None:
    print(f"Backend: {backend_name}")
    print()
    print("prompt | route | tools | llm | latency_ms | labels")
    print("--- | --- | --- | --- | ---: | ---")
    for row in result.get("results") if isinstance(result.get("results"), list) else []:
        tools = ",".join(str(item) for item in (row.get("used_tools") or [])) or "-"
        labels = ",".join(str(item) for item in (row.get("labels") or [])) or "-"
        print(
            f"{row.get('prompt')} | {row.get('route') or '-'} | {tools} | "
            f"{'yes' if bool(row.get('used_llm')) else 'no'} | {int(row.get('latency_ms') or 0)} | {labels}"
        )
    print()
    print("summary_by_family:")
    for item in result.get("summary_by_family") if isinstance(result.get("summary_by_family"), list) else []:
        print(
            f"- {item.get('family')}: matched={item.get('matched_count')}/{item.get('count')} "
            f"grounded={item.get('grounded_count')}/{item.get('count')} labels={json.dumps(item.get('label_counts') or {}, ensure_ascii=True, sort_keys=True)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a compact assistant prompt matrix against the Personal Agent chat path.")
    parser.add_argument("--backend", choices=("inprocess", "live"), default="inprocess")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument(
        "--family",
        action="append",
        choices=(FAMILY_RUNTIME, FAMILY_RECOMMEND, FAMILY_CONTROLLER, FAMILY_MODE),
        help="Repeat to limit the run to one or more prompt families.",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--user-id", default="matrix:user")
    parser.add_argument("--thread-id", default="matrix:user:thread")
    args = parser.parse_args()

    families = set(args.family or (FAMILY_RUNTIME, FAMILY_RECOMMEND, FAMILY_CONTROLLER, FAMILY_MODE))
    backend_name = str(args.backend)
    backend: InProcessBackend | LiveBackend
    if backend_name == "live":
        backend = LiveBackend(args.host, int(args.port))
    else:
        backend = InProcessBackend()

    try:
        result = _run_matrix(
            backend=backend,
            families=families,
            timeout_seconds=max(1.0, float(args.timeout)),
            user_id=str(args.user_id),
            thread_id=str(args.thread_id),
        )
    finally:
        backend.close()

    _print_human(result, backend_name=backend_name)
    if args.json_out:
        with open(str(args.json_out), "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=True, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

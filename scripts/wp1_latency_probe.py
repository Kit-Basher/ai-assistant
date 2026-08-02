from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any
import urllib.request
import sys


ROOT = Path(os.environ.get("WP1_LATENCY_CODE_ROOT", "") or Path(__file__).resolve().parents[1]).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ROUTING_PROMPTS = (
    "u here?",
    "show every local language engine ready for use",
    "check the machine service health picture",
    "recap the task we had underway",
    "what can you actually do?",
)


def _distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"samples": 0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    return {
        "samples": len(ordered),
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "max_ms": round(ordered[-1], 3),
    }


def _direct_routing(samples: int) -> dict[str, Any]:
    # Import only after redirecting every mutable runtime path into a temporary
    # directory. This lets the same probe run against either a baseline release
    # PYTHONPATH or the candidate checkout without touching user state.
    with tempfile.TemporaryDirectory(prefix="personal-agent-wp1-latency-") as raw:
        root = Path(raw)
        env = {
            "AGENT_DB_PATH": str(root / "agent.db"),
            "AGENT_LOG_PATH": str(root / "agent.jsonl"),
            "AGENT_AUDIT_LOG_PATH": str(root / "audit.jsonl"),
            "AGENT_SECRET_STORE_PATH": str(root / "secrets.enc.json"),
            "LLM_REGISTRY_PATH": str(root / "llm_registry.json"),
            "LLM_USAGE_STATS_PATH": str(root / "usage.json"),
            "LLM_HEALTH_STATE_PATH": str(root / "health.json"),
            "MODEL_SCOUT_STATE_PATH": str(root / "scout.json"),
            "AUTOPILOT_NOTIFY_STORE_PATH": str(root / "notifications.json"),
            "PERCEPTION_ROOTS": str(root),
            "LLM_AUTOMATION_ENABLED": "0",
            "AGENT_SAFE_MODE": "1",
            "TELEGRAM_ENABLED": "0",
        }
        previous = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            (root / "llm_registry.json").write_text("{}\n", encoding="utf-8")
            from agent.api_server import AgentRuntime
            from agent.config import load_config

            runtime = AgentRuntime(load_config(require_telegram_token=False))
            orchestrator = runtime.orchestrator()
            orchestrator.preview_conversation_request("latency", ROUTING_PROMPTS[0], thread_id="latency:warm")
            values: list[float] = []
            for index in range(samples):
                prompt = ROUTING_PROMPTS[index % len(ROUTING_PROMPTS)]
                started = time.perf_counter()
                orchestrator.preview_conversation_request(
                    f"latency-{index}",
                    prompt,
                    thread_id=f"latency-{index}:thread",
                )
                values.append((time.perf_counter() - started) * 1000.0)
            return _distribution(values)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _api_chat(base_url: str, samples: int) -> dict[str, Any]:
    values: list[float] = []
    inference_counts: list[int] = []
    for index in range(samples):
        prompt = ROUTING_PROMPTS[index % len(ROUTING_PROMPTS)]
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": prompt}],
                "user_id": f"wp1-latency-{index}",
                "thread_id": f"wp1-latency-{index}:thread",
                "source_surface": "api",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        values.append((time.perf_counter() - started) * 1000.0)
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        inference_counts.append(1 if bool(meta.get("used_llm")) else 0)
    return {**_distribution(values), "model_generations": sum(inference_counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeatable Work Package 1 routing/chat latency probe.")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--api-url", default="")
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args()
    samples = max(5, int(args.samples))
    report: dict[str, Any] = {
        "label": str(args.label),
        "routing": _direct_routing(samples),
        "method": "warm in-process unified preview; deterministic real HTTP /chat when --api-url is supplied",
    }
    if str(args.api_url).strip():
        report["end_to_end_chat"] = _api_chat(str(args.api_url), samples)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

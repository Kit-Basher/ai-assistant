#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LIVE_STATE = Path.home() / ".local" / "share" / "personal-agent"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(base: str, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> tuple[int, dict[str, Any], float]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"}, method=method)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            text = response.read(4 * 1024 * 1024).decode("utf-8")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {"_text": text}
            return int(response.status), parsed if isinstance(parsed, dict) else {}, (time.monotonic() - started) * 1000
    except urllib.error.HTTPError as exc:
        parsed = json.loads(exc.read().decode("utf-8") or "{}")
        return int(exc.code), parsed if isinstance(parsed, dict) else {}, (time.monotonic() - started) * 1000


def chat(base: str, text: str, suffix: str) -> tuple[int, dict[str, Any], float]:
    return request(base, "POST", "/chat", {
        "user_id": f"wp45-{suffix}",
        "session_id": f"wp45-{suffix}",
        "thread_id": f"wp45-{suffix}",
        "source_surface": "webui",
        "messages": [{"role": "user", "content": text}],
    })


def wait_ready(base: str, timeout: float = 40.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            _status, last, _elapsed = request(base, "GET", "/version", timeout=2.0)
            if last:
                return last
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"candidate_not_listening:{last}")


def start(port: int, state: Path, log: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update({
        "AGENT_API_HOST": "127.0.0.1",
        "AGENT_API_PORT": str(port),
        "AGENT_DB_PATH": str(state / "agent.db"),
        "AGENT_LOG_PATH": str(state / "agent.jsonl"),
        "LLM_REGISTRY_PATH": str(state / "llm_registry.json"),
        "AGENT_SECRET_STORE_PATH": str(state / "secrets.enc.json"),
        "AGENT_PERMISSIONS_PATH": str(state / "permissions.json"),
        "AGENT_AUDIT_LOG_PATH": str(state / "audit.jsonl"),
        "AGENT_PACK_STORE_PATH": str(state / "packs"),
        "AGENT_MODEL_MANAGER_STATE_PATH": str(state / "model_manager_state.json"),
        "LLM_USAGE_STATS_PATH": str(state / "llm_usage_stats.json"),
        "LLM_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "OLLAMA_MODEL": "Gemma:latest",
        "AGENT_SAFE_MODE": "1",
        "LLM_ALLOW_REMOTE": "0",
        "TELEGRAM_ENABLED": "0",
        "TELEGRAM_REQUIRED": "0",
        "PERSONAL_AGENT_INSTANCE": "dev",
        "PERSONAL_AGENT_RUNTIME_ROOT": str(ROOT),
        "AGENT_WEBUI_DIST_PATH": str(ROOT / "agent" / "webui" / "dist"),
        "PERSONAL_AGENT_GIT_COMMIT_OVERRIDE": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "PYTHONUNBUFFERED": "1",
    })
    handle = log.open("a", encoding="utf-8")
    try:
        return subprocess.Popen([sys.executable, "-m", "agent.api_server", "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    finally:
        handle.close()


def stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="build/reports/wp45-isolated-candidate.json")
    parser.add_argument("--latency-samples", type=int, default=20)
    args = parser.parse_args()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="personal-agent-wp45-") as raw:
        temp = Path(raw)
        state = temp / "state"
        state.mkdir()
        for name in ("agent.db", "llm_registry.json", "secrets.enc.json", "llm_usage_stats.json", "model_manager_state.json"):
            source = LIVE_STATE / name
            if source.is_file():
                shutil.copy2(source, state / name)
        (state / "packs").mkdir(exist_ok=True)
        registry_path = state / "llm_registry.json"
        before_registry = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        before_defaults = json.loads(registry_path.read_text(encoding="utf-8")).get("defaults", {})
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        log = temp / "candidate.log"
        proc = start(port, state, log)
        try:
            version = wait_ready(base)
            status, truth, elapsed = request(base, "GET", "/llm/models/truth")
            results.append({"name": "canonical_inventory", "passed": status == 200 and truth.get("counts", {}).get("physically_installed") == 9, "elapsed_ms": elapsed})
            results.append({"name": "selection_truth", "passed": truth.get("selection", {}).get("effective_model") == "ollama:Gemma:latest"})
            results.append({"name": "recommendation_truth", "passed": truth.get("recommendation", {}).get("default_general_assistant") == "qwen2.5:3b-instruct"})
            for index, (name, text, expected) in enumerate((
                ("presence", "u here?", "here"),
                ("model_status", "wht model r u using rn?", "gemma"),
                ("installed_inventory", "show models actually on this box pls", "qwen"),
                ("recommendation_chat", "which installed model is best for this assistant and why?", "qwen2.5:3b"),
                ("scout_chat", "what did model scout find?", "qwen2.5:3b"),
                ("why_selected", "why is Gemma still selected?", "still selected"),
                ("unavailable_models", "show unavailable or stale models", "not ready"),
                ("system_status", "give me a system status check", "ready"),
            )):
                status, body, elapsed = chat(base, text, f"case-{index}")
                message = str(body.get("message") or "").lower()
                results.append({"name": name, "passed": status == 200 and expected in message, "elapsed_ms": elapsed, "route": (body.get("meta") or {}).get("route"), "message": message[:300]})
            status, preview, _ = chat(base, "make ollama:qwen2.5:3b-instruct my default", "switch-deny")
            status2, denied, _ = chat(base, "no cancel that", "switch-deny")
            results.append({"name": "switch_preview_denial", "passed": status == 200 and status2 == 200 and any(term in str(denied.get("message") or "").lower() for term in ("cancel", "denied", "not"))})
            status, test_preview, _ = chat(
                base,
                "test qwen2.5 3b for me but do not change my default",
                "test-deny",
            )
            status2, test_denied, _ = chat(base, "no, cancel that test", "test-deny")
            test_message = str(test_preview.get("message") or "").lower()
            results.append({
                "name": "human_spaced_model_test_preview_denial",
                "passed": (
                    status == 200
                    and status2 == 200
                    and "ollama:qwen2.5:3b-instruct" in test_message
                    and "couldn't find that model" not in test_message
                    and any(term in str(test_denied.get("message") or "").lower() for term in ("cancel", "denied", "not"))
                ),
            })
            status, generic, generic_ms = chat(base, "In one short sentence, explain why leaves look green.", "generic")
            results.append({"name": "generic_model_chat", "passed": status == 200 and bool(str(generic.get("message") or "").strip()), "elapsed_ms": generic_ms, "model": (generic.get("meta") or {}).get("model")})
            status, ui, ui_ms = request(base, "GET", "/")
            results.append({"name": "web_ui", "passed": status == 200 and "Personal Agent" in str(ui.get("_text") or ""), "elapsed_ms": ui_ms})
            stop(proc)
            proc = start(port, state, log)
            wait_ready(base)
            status, restarted_truth, _ = request(base, "GET", "/llm/models/truth")
            results.append({"name": "restart_reconstruction", "passed": status == 200 and restarted_truth.get("counts", {}).get("physically_installed") == 9})
            after_registry = hashlib.sha256(registry_path.read_bytes()).hexdigest()
            after_defaults = json.loads(registry_path.read_text(encoding="utf-8")).get("defaults", {})
            results.append({"name": "zero_model_mutation", "passed": before_registry == after_registry and before_defaults == after_defaults})
            latency_output = temp / "latency.json"
            latency = subprocess.run([
                sys.executable, "scripts/wp45_latency_probe.py", "--base-url", base,
                "--label", "isolated-candidate", "--samples", str(args.latency_samples), "--output", str(latency_output),
            ], cwd=ROOT, check=False, capture_output=True, text=True, timeout=120)
            latency_report = json.loads(latency_output.read_text(encoding="utf-8")) if latency_output.is_file() else {}
            presence_p95 = ((latency_report.get("routes") or {}).get("presence") or {}).get("body_complete", {}).get("p95_ms")
            status_p95 = ((latency_report.get("routes") or {}).get("system_status") or {}).get("body_complete", {}).get("p95_ms")
            results.append({"name": "latency_budgets", "passed": latency.returncode == 0 and float(presence_p95 or 1e9) < 250 and float(status_p95 or 1e9) < 1000, "presence_p95_ms": presence_p95, "system_status_p95_ms": status_p95})
        finally:
            stop(proc)
        port_closed = True
        with socket.socket() as sock:
            port_closed = sock.connect_ex(("127.0.0.1", port)) != 0
        results.append({"name": "temporary_port_closed", "passed": port_closed})
        report = {
            "contract": "personal-agent.wp45-isolated-candidate.v1",
            "version": version,
            "results": results,
            "summary": {"passed": sum(bool(row.get("passed")) for row in results), "failed": sum(not bool(row.get("passed")) for row in results), "total": len(results)},
            "temporary_cleanup": {"port_closed": port_closed, "temporary_directory_removed_on_exit": True},
        }
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

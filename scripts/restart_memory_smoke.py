#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.llm.registry import load_registry_document


DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("RESTART_MEMORY_SMOKE_TIMEOUT_SECONDS", "60"))
DEFAULT_PROMPT_USER_ID = "restart-memory-smoke"
DEFAULT_PROMPT_THREAD_ID = "restart-memory-smoke:thread"
DEFAULT_RESTART_CYCLES = int(os.environ.get("RESTART_MEMORY_SMOKE_CYCLES", "2"))


def _first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _normalized(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _json_from_response(body: str) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
            ok = status < 400
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = int(getattr(exc, "code", 500))
        ok = False
    except urllib.error.URLError as exc:
        return {"ok": False, "status": 0, "payload": {}, "raw": "", "error": f"transport error: {exc.reason}"}
    except Exception as exc:  # pragma: no cover - defensive live-smoke guard
        return {"ok": False, "status": 0, "payload": {}, "raw": "", "error": f"transport error: {exc}"}
    payload_json = _json_from_response(raw)
    return {"ok": ok, "status": status, "payload": payload_json, "raw": raw}


def _post_chat(
    base_url: str,
    prompt: str,
    *,
    user_id: str,
    thread_id: str,
    trace_id: str,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": str(prompt or "")}],
        "purpose": "chat",
        "task_type": "chat",
        "source_surface": "operator_smoke",
        "user_id": user_id,
        "thread_id": thread_id,
        "trace_id": trace_id,
    }
    result = _request_json(base_url, "POST", "/chat", payload, timeout=timeout)
    payload_json = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    assistant = payload_json.get("assistant") if isinstance(payload_json.get("assistant"), dict) else {}
    meta = payload_json.get("meta") if isinstance(payload_json.get("meta"), dict) else {}
    text = str(assistant.get("content") or payload_json.get("message") or payload_json.get("error") or "").strip()
    return {
        "result": result,
        "ok": bool(result.get("ok")),
        "status": int(result.get("status") or 0),
        "text": text,
        "first_line": _first_line(text),
        "route": str(meta.get("route") or "").strip().lower() or "unknown",
        "trace_id": trace_id,
    }


def _wait_for_ready(base_url: str, *, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout)
    last: dict[str, Any] = {"ok": False, "status": 0, "payload": {}, "raw": ""}
    while time.monotonic() < deadline:
        last = _request_json(base_url, "GET", "/ready", timeout=5.0)
        if int(last.get("status") or 0) > 0:
            return last
        time.sleep(0.5)
    return last


def _await_chat_ok(base_url: str, prompt: str, *, user_id: str, thread_id: str, trace_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout)
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _post_chat(
            base_url,
            prompt,
            user_id=user_id,
            thread_id=thread_id,
            trace_id=trace_id,
            timeout=min(20.0, max(5.0, float(timeout))),
        )
        if bool(last.get("ok")) and int(last.get("status") or 0) < 400:
            return last
        time.sleep(0.5)
    return last


def _kill_process(proc: subprocess.Popen[str], *, timeout: float) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=timeout)
        except Exception:
            pass


def _start_server(
    *,
    port: int,
    workdir: Path,
    log_path: Path,
    db_path: Path,
    registry_path: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(
        {
            "AGENT_API_HOST": "127.0.0.1",
            "AGENT_API_PORT": str(int(port)),
            "AGENT_DB_PATH": str(db_path),
            "AGENT_LOG_PATH": str(log_path),
            "LLM_REGISTRY_PATH": str(registry_path),
            "AGENT_SECRET_STORE_PATH": str(workdir / "secrets.enc.json"),
            "AGENT_PERMISSIONS_PATH": str(workdir / "permissions.json"),
            "AGENT_AUDIT_LOG_PATH": str(workdir / "audit.jsonl"),
            "LLM_PROVIDER": "none",
            "ALLOW_CLOUD": "true",
            "PREFER_LOCAL": "true",
            "AGENT_TIMEZONE": "UTC",
            "TELEGRAM_ENABLED": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if extra_env:
        env.update(extra_env)
    log_handle = log_path.open("a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agent.api_server", "--host", "127.0.0.1", "--port", str(int(port))],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_handle.close()
    return proc


def _tail_log(log_path: Path, *, limit_lines: int = 120) -> str:
    if not log_path.is_file():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit_lines:])


def _current_default_model(defaults: dict[str, Any]) -> str:
    model = str(
        defaults.get("chat_model")
        or defaults.get("default_model")
        or defaults.get("resolved_default_model")
        or ""
    ).strip()
    return model


def _assert_contains_any(text: str, tokens: tuple[str, ...], *, label: str) -> None:
    lowered = _normalized(text)
    if not any(token in lowered for token in tokens):
        raise RuntimeError(f"{label} did not mention any of: {', '.join(tokens)}\nfirst_line={_first_line(text)}")


def _write_registry_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = load_registry_document(None)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Restart a temp Personal Agent API and verify that memory survives repeated restarts."
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Per phase timeout in seconds.")
    parser.add_argument(
        "--cycles",
        type=int,
        default=DEFAULT_RESTART_CYCLES,
        help="How many restart cycles to verify before exiting.",
    )
    parser.add_argument("--no-cleanup", action="store_true", help="Leave the temp server running if the smoke passes.")
    args = parser.parse_args(argv)

    if int(args.cycles) < 1:
        raise SystemExit("--cycles must be >= 1")

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    user_id = DEFAULT_PROMPT_USER_ID
    thread_id = DEFAULT_PROMPT_THREAD_ID
    preference_seed = "remember this: I prefer concise answers."
    seed_topic = "we are testing the assistant viability gate"
    model_target = "ollama:llama3"
    probe_turn = "what are we doing?"
    preference_probe = "what do you remember about me?"
    interruption_turn = "what is the runtime status?"
    resume_turn = "what were we working on before?"
    return_turn = "what are we doing right now?"

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        registry_path = workdir / "registry.json"
        db_path = workdir / "agent.db"
        log_path = workdir / "api.log"
        _write_registry_file(registry_path)

        proc = _start_server(
            port=port,
            workdir=workdir,
            log_path=log_path,
            db_path=db_path,
            registry_path=registry_path,
        )
        try:
            ready = _wait_for_ready(base_url, timeout=float(args.timeout))
            if int(ready.get("status") or 0) <= 0:
                raise SystemExit(f"server did not become reachable on {base_url}\n{_tail_log(log_path)}")

            defaults_set = _request_json(
                base_url,
                "PUT",
                "/defaults",
                {"default_provider": "ollama", "default_model": model_target},
                timeout=float(args.timeout),
            )
            if not bool(defaults_set.get("ok")):
                raise SystemExit(f"default model update failed: {defaults_set}\n{_tail_log(log_path)}")
            defaults_after_set = _request_json(base_url, "GET", "/defaults", timeout=float(args.timeout))
            defaults_after_payload = defaults_after_set.get("payload") if isinstance(defaults_after_set.get("payload"), dict) else {}
            if _current_default_model(defaults_after_payload) != model_target:
                raise SystemExit(
                    f"default model did not persist after initial update: {defaults_after_payload}\n{_tail_log(log_path)}"
                )

            preference_seed_result = _await_chat_ok(
                base_url,
                preference_seed,
                user_id=user_id,
                thread_id=thread_id,
                trace_id=f"restart-memory-preference-seed-{int(time.time())}",
                timeout=float(args.timeout),
            )
            if not preference_seed_result.get("ok"):
                raise SystemExit(f"preference seed turn failed: {preference_seed_result}\n{_tail_log(log_path)}")

            seed = _await_chat_ok(
                base_url,
                seed_topic,
                user_id=user_id,
                thread_id=thread_id,
                trace_id=f"restart-memory-seed-{int(time.time())}",
                timeout=float(args.timeout),
            )
            if not seed.get("ok"):
                raise SystemExit(f"seed turn failed: {seed}\n{_tail_log(log_path)}")

            preference_recall = _await_chat_ok(
                base_url,
                preference_probe,
                user_id=user_id,
                thread_id=thread_id,
                trace_id=f"restart-memory-preference-recall-{int(time.time())}",
                timeout=float(args.timeout),
            )
            if not preference_recall.get("ok"):
                raise SystemExit(f"pre-restart preference recall failed: {preference_recall}\n{_tail_log(log_path)}")
            _assert_contains_any(
                str(preference_recall.get("text") or ""),
                ("concise", "concise answers", "concise replies"),
                label="pre-restart preference recall",
            )

            recall = _await_chat_ok(
                base_url,
                probe_turn,
                user_id=user_id,
                thread_id=thread_id,
                trace_id=f"restart-memory-recall-{int(time.time())}",
                timeout=float(args.timeout),
            )
            if not recall.get("ok"):
                raise SystemExit(f"pre-restart recall failed: {recall}\n{_tail_log(log_path)}")
            _assert_contains_any(
                str(recall.get("text") or ""),
                ("assistant_viability_gate", "assistant viability gate", "what we are doing", "testing the assistant viability gate"),
                label="pre-restart recall",
            )

            interruption = _await_chat_ok(
                base_url,
                interruption_turn,
                user_id=user_id,
                thread_id=thread_id,
                trace_id=f"restart-memory-interruption-{int(time.time())}",
                timeout=float(args.timeout),
            )
            if not interruption.get("ok"):
                raise SystemExit(f"interruption turn failed: {interruption}\n{_tail_log(log_path)}")

            for cycle in range(1, int(args.cycles) + 1):
                _kill_process(proc, timeout=20.0)

                proc = _start_server(
                    port=port,
                    workdir=workdir,
                    log_path=log_path,
                    db_path=db_path,
                    registry_path=registry_path,
                )
                ready_after = _wait_for_ready(base_url, timeout=float(args.timeout))
                if int(ready_after.get("status") or 0) <= 0:
                    raise SystemExit(
                        f"server did not come back after restart cycle {cycle} on {base_url}\n{_tail_log(log_path)}"
                    )

                defaults_after_restart = _request_json(base_url, "GET", "/defaults", timeout=float(args.timeout))
                defaults_after_restart_payload = (
                    defaults_after_restart.get("payload")
                    if isinstance(defaults_after_restart.get("payload"), dict)
                    else {}
                )
                if _current_default_model(defaults_after_restart_payload) != model_target:
                    raise SystemExit(
                        f"default model did not survive restart cycle {cycle}: {defaults_after_restart_payload}\n{_tail_log(log_path)}"
                    )

                preference_after_restart = _await_chat_ok(
                    base_url,
                    preference_probe,
                    user_id=user_id,
                    thread_id=thread_id,
                    trace_id=f"restart-memory-preference-after-{cycle}-{int(time.time())}",
                    timeout=float(args.timeout),
                )
                if not preference_after_restart.get("ok"):
                    raise SystemExit(
                        f"post-restart preference recall failed in cycle {cycle}: {preference_after_restart}\n{_tail_log(log_path)}"
                    )
                _assert_contains_any(
                    str(preference_after_restart.get("text") or ""),
                    ("concise", "concise answers", "concise replies"),
                    label=f"post-restart preference recall cycle {cycle}",
                )

                resumed = _await_chat_ok(
                    base_url,
                    resume_turn,
                    user_id=user_id,
                    thread_id=thread_id,
                    trace_id=f"restart-memory-resume-{cycle}-{int(time.time())}",
                    timeout=float(args.timeout),
                )
                if not resumed.get("ok"):
                    raise SystemExit(
                        f"post-restart resume turn failed in cycle {cycle}: {resumed}\n{_tail_log(log_path)}"
                    )
                _assert_contains_any(
                    str(resumed.get("text") or ""),
                    ("assistant_viability_gate", "assistant viability gate", "working on", "testing", "memory", "current task"),
                    label=f"post-restart resume cycle {cycle}",
                )

                returned = _await_chat_ok(
                    base_url,
                    return_turn,
                    user_id=user_id,
                    thread_id=thread_id,
                    trace_id=f"restart-memory-return-{cycle}-{int(time.time())}",
                    timeout=float(args.timeout),
                )
                if not returned.get("ok"):
                    raise SystemExit(
                        f"post-restart return turn failed in cycle {cycle}: {returned}\n{_tail_log(log_path)}"
                    )
                _assert_contains_any(
                    str(returned.get("text") or ""),
                    ("assistant_viability_gate", "assistant viability gate", "what we are doing", "working on", "current task"),
                    label=f"post-restart return cycle {cycle}",
                )

                if cycle < int(args.cycles):
                    status_turn = _await_chat_ok(
                        base_url,
                        interruption_turn,
                        user_id=user_id,
                        thread_id=thread_id,
                        trace_id=f"restart-memory-status-{cycle}-{int(time.time())}",
                        timeout=float(args.timeout),
                    )
                    if not status_turn.get("ok"):
                        raise SystemExit(
                            f"status turn failed between restart cycles {cycle}: {status_turn}\n{_tail_log(log_path)}"
                        )

            print(f"base_url: {base_url}")
            print(f"restart_cycles: {int(args.cycles)}")
            print(f"restart_point: {seed_topic}")
            print(f"default_model: {model_target}")
            print(f"pre_restart_preference: {_first_line(str(preference_recall.get('text') or ''))}")
            print(f"pre_restart_recall: {_first_line(str(recall.get('text') or ''))}")
            print(f"last_post_restart_resume: {_first_line(str(resumed.get('text') or ''))}")
            print(f"last_post_restart_return: {_first_line(str(returned.get('text') or ''))}")
            print("restart-safe memory smoke passed")
            return 0
        finally:
            if not bool(args.no_cleanup):
                _kill_process(proc, timeout=20.0)


if __name__ == "__main__":
    raise SystemExit(main())

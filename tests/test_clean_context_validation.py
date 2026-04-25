from __future__ import annotations

import json
import os
import subprocess
import tempfile
import shutil
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.intent.assessment import IntentAssessment, IntentCandidate
from agent.public_chat import build_no_llm_public_message


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: Path, *, env: dict[str, str], args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *(args or [])],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _config(registry_path: str, db_path: str, *, skills_path: str, **overrides: object) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path=skills_path,
        ollama_host="http://127.0.0.1:11434",
        ollama_model="llama3",
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
        model_scout_state_path=os.path.join(os.path.dirname(db_path), "model_scout_state.json"),
        autopilot_notify_store_path=os.path.join(os.path.dirname(db_path), "llm_notifications.json"),
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _request_json(base_url: str, method: str, path: str, payload: dict[str, object] | None = None) -> tuple[int, dict[str, object], str]:
    headers = {"Accept": "application/json, text/html"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            raw = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw.strip().startswith("{") else {}
            if not isinstance(parsed, dict):
                parsed = {}
            return int(getattr(response, "status", 200)), parsed, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw) if raw.strip().startswith("{") else {}
        if not isinstance(parsed, dict):
            parsed = {}
        return int(getattr(exc, "code", 500)), parsed, raw


def _request_text(base_url: str, path: str) -> tuple[int, str]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers={"Accept": "text/html,application/json"})
    with urllib.request.urlopen(request, timeout=5.0) as response:
        return int(getattr(response, "status", 200)), response.read().decode("utf-8", errors="replace")


def _assistant_text(payload: dict[str, object]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    return str(assistant.get("content") or payload.get("message") or payload.get("error") or "").strip()


def _assistant_warnings(text: str) -> list[str]:
    lowered = str(text or "").lower()
    warnings: list[str] = []
    if not str(text or "").strip():
        warnings.append("empty assistant reply")
    if text.lstrip().startswith("{") or text.lstrip().startswith("["):
        warnings.append("raw json reply")
    if any(
        token in lowered
        for token in (
            "trace_id:",
            "route_reason:",
            "selection_policy",
            "runtime_payload",
            "runtime_state_failure_reason",
            "setup_type:",
            "operator_only:",
            "source_surface:",
            "thread_id:",
            "user_id:",
            "local_observations",
        )
    ):
        warnings.append("internal text leak")
    if len(str(text).split()) < 3:
        warnings.append("too short to read as assistant output")
    return warnings


def _write_fake_systemctl(bin_dir: Path, *, logs_dir: Path) -> Path:
    systemctl_path = bin_dir / "systemctl"
    systemctl_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n"
        "state_dir=\"${SYSTEMCTL_STATE_DIR:?}\"\n"
        "mkdir -p \"$state_dir\"\n"
        "enabled_flag=\"$state_dir/enabled.flag\"\n"
        "active_flag=\"$state_dir/active.flag\"\n"
        "case \"${1-}\" in\n"
        "  --user)\n"
        "    case \"${2-}\" in\n"
        "      show-environment)\n"
        "        exit 0\n"
        "        ;;\n"
        "      is-active)\n"
        "        if [ -f \"$active_flag\" ]; then exit 0; fi\n"
        "        exit 3\n"
        "        ;;\n"
        "      is-enabled)\n"
        "        if [ -f \"$enabled_flag\" ]; then exit 0; fi\n"
        "        exit 3\n"
        "        ;;\n"
        "      enable)\n"
        "        touch \"$enabled_flag\"\n"
        "        exit 0\n"
        "        ;;\n"
        "      start)\n"
        "        touch \"$active_flag\"\n"
        "        exit 0\n"
        "        ;;\n"
        "      stop)\n"
        "        rm -f \"$active_flag\"\n"
        "        exit 0\n"
        "        ;;\n"
        "      disable)\n"
        "        rm -f \"$enabled_flag\" \"$active_flag\"\n"
        "        exit 0\n"
        "        ;;\n"
        "      daemon-reload)\n"
        "        exit 0\n"
        "        ;;\n"
        "    esac\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    systemctl_path.chmod(0o755)
    (logs_dir / "systemctl.log").write_text("", encoding="utf-8")
    return systemctl_path


def _write_fake_curl(bin_dir: Path, *, logs_dir: Path, ready_after: int = 1) -> Path:
    curl_path = bin_dir / "curl"
    curl_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "count=0\n"
        "if [ -f \"$CURL_COUNT\" ]; then count=$(cat \"$CURL_COUNT\"); fi\n"
        "count=$((count + 1))\n"
        "printf '%s' \"$count\" > \"$CURL_COUNT\"\n"
        f"if [ \"$count\" -lt {ready_after} ]; then exit 1; fi\n"
        "printf '{\"ready\": true, \"summary\": \"Ready.\"}'\n",
        encoding="utf-8",
    )
    curl_path.chmod(0o755)
    (logs_dir / "curl-count.txt").write_text("", encoding="utf-8")
    return curl_path


def _write_fake_xdg_open(bin_dir: Path, *, logs_dir: Path) -> Path:
    xdg_open_path = bin_dir / "xdg-open"
    xdg_open_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"$XDG_OPEN_LOG\"\n",
        encoding="utf-8",
    )
    xdg_open_path.chmod(0o755)
    (logs_dir / "xdg-open.log").write_text("", encoding="utf-8")
    return xdg_open_path


def _write_fake_python(bin_dir: Path) -> Path:
    python_path = bin_dir / "python3"
    python_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "case \"${1-}\" in\n"
        "  -c)\n"
        "    case \"${FAKE_PYTHON_VERSION:-3.11.8}\" in\n"
        "      3.11*|3.12*|3.13*|4.*)\n"
        "        exit 0\n"
        "        ;;\n"
        "      *)\n"
        "        exit 1\n"
        "        ;;\n"
        "    esac\n"
        "    ;;\n"
        "  -m)\n"
        "    case \"${2-}\" in\n"
        "      venv)\n"
        "        target=\"${3-}\"\n"
        "        mkdir -p \"$target/bin\"\n"
        "        cat > \"$target/bin/python\" <<'PY'\n"
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [ \"${1-}\" = \"-m\" ] && [ \"${2-}\" = \"pip\" ]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"${1-}\" = \"-m\" ] && [ \"${2-}\" = \"agent\" ]; then\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
        "PY\n"
        "        chmod +x \"$target/bin/python\"\n"
        "        ln -sf \"$target/bin/python\" \"$target/bin/pip\"\n"
        "        exit 0\n"
        "        ;;\n"
        "      pip)\n"
        "        exit 0\n"
        "        ;;\n"
        "      agent)\n"
        "        exit 0\n"
        "        ;;\n"
        "      doctor)\n"
        "        exit 0\n"
        "        ;;\n"
        "    esac\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    return python_path


class _MemoryHandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object] | None = None) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.status_code = 0
        self.content_type = ""
        self.body = b""
        self._payload = payload or {}

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        self.status_code = status
        self.content_type = "application/json"
        self.body = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        cache_control: str | None = None,
    ) -> None:
        _ = cache_control
        self.status_code = status
        self.content_type = content_type
        self.body = body

    def _read_json(self) -> dict[str, object]:
        return dict(self._payload)


def _serve(runtime: AgentRuntime) -> tuple[ThreadingHTTPServer, str, threading.Thread]:
    class _HandlerForTest(APIServerHandler):
        pass

    _HandlerForTest.runtime = runtime
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HandlerForTest)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}", thread


def _memory_request_json(
    runtime: AgentRuntime,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object], str]:
    handler = _MemoryHandlerForTest(runtime, path, payload)
    if method.upper() == "GET":
        handler.do_GET()
    elif method.upper() == "POST":
        handler.do_POST()
    else:  # pragma: no cover - defensive
        raise AssertionError(f"unsupported method: {method}")
    raw = handler.body.decode("utf-8", errors="replace")
    parsed = json.loads(raw) if raw.strip().startswith("{") else {}
    if not isinstance(parsed, dict):
        parsed = {}
    return int(handler.status_code), parsed, raw


def _memory_request_text(runtime: AgentRuntime, path: str) -> tuple[int, str]:
    handler = _MemoryHandlerForTest(runtime, path)
    handler.do_GET()
    return int(handler.status_code), handler.body.decode("utf-8", errors="replace")


class TestCleanContextValidation(unittest.TestCase):
    def test_release_bundle_clean_context_install_launch_chat_relaunch_and_uninstall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_out = root / "bundle-out"
            install_root = root / "install-root"
            home = root / "home"
            bin_dir = root / "bin"
            logs_dir = root / "logs"
            state_dir = root / "systemctl-state"
            for path in (home, bin_dir, logs_dir, state_dir):
                path.mkdir(parents=True, exist_ok=True)

            systemctl_path = _write_fake_systemctl(bin_dir, logs_dir=logs_dir)
            curl_path = _write_fake_curl(bin_dir, logs_dir=logs_dir, ready_after=2)
            xdg_open_path = _write_fake_xdg_open(bin_dir, logs_dir=logs_dir)
            python_path = _write_fake_python(bin_dir)

            build = _run_script(
                REPO_ROOT / "scripts" / "build_release_bundle.sh",
                env=os.environ.copy(),
                args=["--outdir", str(bundle_out), "--clean"],
            )
            self.assertEqual(0, build.returncode, build.stderr)
            bundle_dir = Path(build.stdout.strip().splitlines()[0])

            install_env = os.environ.copy()
            install_env.update(
                {
                    "HOME": str(home),
                    "XDG_DATA_HOME": str(home / ".local" / "share"),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "SYSTEMCTL_LOG": str(logs_dir / "systemctl.log"),
                    "SYSTEMCTL_STATE_DIR": str(state_dir),
                    "CURL_COUNT": str(logs_dir / "curl-count.txt"),
                    "XDG_OPEN_LOG": str(logs_dir / "xdg-open.log"),
                    "AGENT_BUNDLE_INSTALL_PYTHON": str(python_path),
                    "AGENT_BUNDLE_INSTALL_SYSTEMCTL": str(systemctl_path),
                    "AGENT_BUNDLE_INSTALL_CURL": str(curl_path),
                    "AGENT_BUNDLE_INSTALL_XDG_OPEN": str(xdg_open_path),
                }
            )

            install = _run_script(bundle_dir / "install.sh", env=install_env, args=["--install-root", str(install_root)])
            self.assertEqual(0, install.returncode, install.stderr)

            runtime_root = install_root / "runtime"
            current_root = runtime_root / "current"
            release_root = current_root.resolve()
            launcher_path = install_root / "bin" / "personal-agent-webui"
            uninstall_path = install_root / "bin" / "personal-agent-uninstall"
            service_path = home / ".config" / "systemd" / "user" / "personal-agent-api.service"
            desktop_path = home / ".local" / "share" / "applications" / "personal-agent.desktop"
            icon_path = home / ".local" / "share" / "icons" / "hicolor" / "scalable" / "apps" / "personal-agent.svg"

            self.assertTrue(launcher_path.is_symlink())
            self.assertTrue(uninstall_path.is_symlink())
            self.assertTrue(current_root.is_symlink())
            self.assertTrue((release_root / ".venv" / "bin" / "python").exists())
            self.assertTrue(service_path.is_file())
            self.assertTrue(desktop_path.is_file())
            self.assertTrue(icon_path.is_file())
            service_text = service_path.read_text(encoding="utf-8")
            self.assertIn(str(current_root), service_text)
            self.assertIn(str(current_root / ".venv" / "bin" / "python"), service_text)
            self.assertIn(str(current_root / "agent" / "webui" / "dist"), service_text)
            self.assertNotIn(str(REPO_ROOT), service_text)

            db_path = install_root / "agent.db"
            runtime = AgentRuntime(
                _config(
                    str(install_root / "llm_registry.json"),
                    str(db_path),
                    skills_path=str(release_root / "skills"),
                    log_path=str(install_root / "agent.log"),
                )
            )
            orchestrator = runtime.orchestrator()
            ready_payload = {
                "ok": True,
                "ready": True,
                "phase": "ready",
                "startup_phase": "ready",
                "runtime_mode": "READY",
                "summary": "Ready to chat.",
            }

            with (
                mock.patch.object(runtime, "ready_status", return_value=ready_payload),
                mock.patch.object(runtime, "chat_route_decision", return_value={"route": "generic_chat"}),
                mock.patch.object(runtime, "should_use_assistant_frontdoor", return_value=False),
                mock.patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None),
                mock.patch.object(orchestrator, "_llm_chat_available", return_value=False),
                mock.patch.object(runtime, "consume_clarify_recovery_choice", return_value=(False, {})),
                mock.patch.object(runtime, "consume_binary_clarification_choice", return_value=(False, {})),
                mock.patch.object(runtime, "consume_intent_choice", return_value=(False, {})),
                mock.patch.object(runtime, "consume_thread_integrity_choice", return_value=(False, {})),
                mock.patch(
                    "agent.api_server.detect_low_confidence",
                    return_value=SimpleNamespace(is_low_confidence=False, reason="none", debug={"norm": "hello"}),
                ),
                mock.patch(
                    "agent.api_server.classify_ambiguity",
                    return_value=SimpleNamespace(ambiguous=False, reason="none"),
                ),
                mock.patch(
                    "agent.api_server.assess_intent_deterministic",
                    return_value=IntentAssessment(
                        decision="proceed",
                        confidence=1.0,
                        candidates=[IntentCandidate(intent="chat", score=1.0, reason="clean-context", details={})],
                        next_question=None,
                        debug={"source": "clean-context"},
                    ),
                ),
            ):
                server = None
                thread = None
                base_url = None
                use_network = True
                try:
                    server, base_url, thread = _serve(runtime)
                except PermissionError:
                    use_network = False
                try:
                    request_json = _request_json if use_network else _memory_request_json
                    request_text = _request_text if use_network else _memory_request_text
                    request_base = str(base_url or "")
                    webui_url = f"{base_url.rstrip('/')}/" if use_network and base_url else "http://127.0.0.1:8765/"
                    launcher_env = install_env | {
                        "AGENT_WEBUI_URL": webui_url,
                        "AGENT_LAUNCHER_SYSTEMCTL": str(systemctl_path),
                        "AGENT_LAUNCHER_XDG_OPEN": str(xdg_open_path),
                        "AGENT_LAUNCHER_WAIT_SECONDS": "5",
                        "AGENT_LAUNCHER_POLL_SECONDS": "0",
                        "AGENT_LAUNCHER_OPEN_BROWSER": "0",
                    }

                    root_status, root_html = request_text(request_base, "/") if use_network else request_text(runtime, "/")
                    self.assertEqual(200, root_status)
                    self.assertIn("id=\"root\"", root_html)
                    self.assertIn("personal-agent-webui", root_html)

                    ready_status, ready_payload_live, ready_raw = (
                        request_json(request_base, "GET", "/ready")
                        if use_network
                        else request_json(runtime, "GET", "/ready")
                    )
                    self.assertEqual(200, ready_status)
                    self.assertTrue(bool(ready_payload_live.get("ready")))
                    self.assertIn("Ready", str(ready_payload_live.get("summary") or ""))
                    self.assertNotIn("{", str(ready_payload_live.get("summary") or "").strip())
                    self.assertNotIn("trace_id:", ready_raw.lower())

                    first_launch = _run_script(launcher_path, env=launcher_env)
                    self.assertEqual(0, first_launch.returncode, first_launch.stderr)
                    self.assertIn("start personal-agent-api.service", (logs_dir / "systemctl.log").read_text(encoding="utf-8"))
                    self.assertIn("browser auto-open disabled", first_launch.stderr)
                    self.assertFalse((logs_dir / "xdg-open.log").read_text(encoding="utf-8").strip())

                    first_payload = {
                        "messages": [{"role": "user", "content": "tell me about a bicycle"}],
                        "session_id": "clean-context-session",
                        "thread_id": "clean-context-thread",
                        "user_id": "clean-context-user",
                        "source_surface": "webui",
                        "purpose": "chat",
                        "task_type": "chat",
                    }
                    first_status, first_body, first_raw = (
                        request_json(request_base, "POST", "/chat", first_payload)
                        if use_network
                        else request_json(runtime, "POST", "/chat", first_payload)
                    )
                    self.assertEqual(200, first_status)
                    self.assertTrue(bool(first_body.get("ok")))
                    first_text = _assistant_text(first_body)
                    self.assertIn(
                        first_text,
                        {
                            build_no_llm_public_message(),
                            build_no_llm_public_message(runtime_ready=True),
                        },
                    )
                    self.assertEqual(first_text, str(first_body.get("message") or "").strip())
                    self.assertEqual([], _assistant_warnings(first_text))
                    self.assertNotIn("runtime_payload", first_raw.lower())
                    self.assertNotIn("trace_id:", first_raw.lower())

                    followup_payload = {
                        "messages": [{"role": "user", "content": "what should I do first?"}],
                        "session_id": "clean-context-session",
                        "thread_id": "clean-context-thread",
                        "user_id": "clean-context-user",
                        "source_surface": "webui",
                        "purpose": "chat",
                        "task_type": "chat",
                    }
                    followup_status, followup_body, followup_raw = (
                        request_json(request_base, "POST", "/chat", followup_payload)
                        if use_network
                        else request_json(runtime, "POST", "/chat", followup_payload)
                    )
                    self.assertEqual(200, followup_status)
                    self.assertTrue(bool(followup_body.get("ok")))
                    followup_text = _assistant_text(followup_body)
                    self.assertEqual(build_no_llm_public_message(), followup_text)
                    self.assertEqual(followup_text, str(followup_body.get("message") or "").strip())
                    self.assertEqual([], _assistant_warnings(followup_text))
                    self.assertNotIn("runtime_payload", followup_raw.lower())

                    second_launch = _run_script(launcher_path, env=launcher_env)
                    self.assertEqual(0, second_launch.returncode, second_launch.stderr)
                    self.assertIn("browser auto-open disabled", second_launch.stderr)
                    self.assertFalse((logs_dir / "xdg-open.log").read_text(encoding="utf-8").strip())

                    second_followup_payload = {
                        "messages": [{"role": "user", "content": "what should I do first?"}],
                        "session_id": "clean-context-session",
                        "thread_id": "clean-context-thread",
                        "user_id": "clean-context-user",
                        "source_surface": "webui",
                        "purpose": "chat",
                        "task_type": "chat",
                    }
                    second_followup_status, second_followup_body, second_followup_raw = (
                        request_json(request_base, "POST", "/chat", second_followup_payload)
                        if use_network
                        else request_json(runtime, "POST", "/chat", second_followup_payload)
                    )
                    self.assertEqual(200, second_followup_status)
                    self.assertTrue(bool(second_followup_body.get("ok")))
                    second_followup_text = _assistant_text(second_followup_body)
                    self.assertEqual(build_no_llm_public_message(), second_followup_text)
                    self.assertEqual([], _assistant_warnings(second_followup_text))
                    self.assertNotIn("runtime_payload", second_followup_raw.lower())

                finally:
                    if server is not None and thread is not None:
                        server.shutdown()
                        thread.join(timeout=5)
                        server.server_close()

            preserve_state = _run_script(bundle_dir / "uninstall.sh", env=install_env, args=["--install-root", str(install_root)])
            self.assertEqual(0, preserve_state.returncode, preserve_state.stderr)
            self.assertTrue(install_root.exists())
            self.assertFalse((install_root / "runtime").exists())
            self.assertFalse((install_root / "bin").exists())
            self.assertFalse(desktop_path.exists())
            self.assertFalse(service_path.exists())

            remove_state = _run_script(
                bundle_dir / "uninstall.sh",
                env=install_env,
                args=["--install-root", str(install_root), "--remove-state"],
            )
            self.assertEqual(0, remove_state.returncode, remove_state.stderr)
            self.assertFalse(install_root.exists())

            for path in (bundle_out, home, bin_dir, logs_dir, state_dir):
                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

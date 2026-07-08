from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]


def _load_first_run_smoke():
    path = ROOT / "scripts" / "first_run_smoke.py"
    spec = importlib.util.spec_from_file_location("first_run_smoke_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contains_any_is_case_insensitive() -> None:
    module = _load_first_run_smoke()

    assert module._contains_any("Web search is not set up.", ("search is not configured", "WEB SEARCH"))
    assert not module._contains_any("Telegram is optional.", ("search", "backup"))


def test_first_run_smoke_accepts_safe_fresh_state(monkeypatch, tmp_path) -> None:
    module = _load_first_run_smoke()
    home = tmp_path / "home"
    state = home / ".local/share/personal-agent"
    config = home / ".config/personal-agent"
    state.mkdir(parents=True)
    config.mkdir(parents=True)
    log_path = tmp_path / "api.log"
    log_path.write_text("first-run test log\n", encoding="utf-8")

    @contextmanager
    def fake_isolated_api(timeout: float) -> Iterator[tuple[str, Path, Path]]:
        del timeout
        yield "http://127.0.0.1:9999", home, log_path

    def fake_request_json(method: str, base_url: str, path: str, **kwargs: Any) -> dict[str, Any]:
        del method, base_url, kwargs
        if path == "/ready":
            return {"http_status": 200, "ready": False, "runtime_mode": "BOOTSTRAP_REQUIRED", "state_label": "Degraded", "chat_usable": False}
        if path == "/state":
            return {"http_status": 200, "ok": True, "runtime_mode": "BOOTSTRAP_REQUIRED", "state_label": "Degraded"}
        if path == "/version":
            return {"http_status": 200, "runtime_instance": "dev", "git_commit": "testcommit"}
        if path == "/telegram/status":
            return {"http_status": 200, "configured": False, "effective_state": "disabled_optional", "reason": "not_configured"}
        if path == "/search/status":
            return {"http_status": 200, "enabled": False, "search_state": "never_configured", "reason": "search_disabled"}
        raise AssertionError(path)

    def fake_request_text(method: str, base_url: str, path: str, **kwargs: Any) -> tuple[int, str]:
        del method, base_url, path, kwargs
        return 200, "<html>Personal Agent</html>"

    def fake_post_chat(base_url: str, message: str, *, thread_id: str, timeout: float = 20.0) -> dict[str, Any]:
        del base_url, thread_id, timeout
        responses = {
            "what is dots.tts?": "Web search is not set up. I can set up local SearXNG after confirmation. I did not search.",
            "what do you remember about me?": "I do not have much saved about you yet.",
            "install htop": "Plan Mode v2\nAction type: package.install\nThis mutates the local system. Say yes to confirm.",
            "make a support bundle": "Support bundle preview. It will create a redacted support bundle and exclude raw tokens.",
            "back up the assistant": "Backup assistant preview. Secrets must remain redacted. This requires explicit confirmation.",
            "restore from backup": "Restore from backup preview. Restore v1 is confirmation-gated and uses a safety snapshot.",
            "clean old backup files": "Cleanup old Personal Agent files preview. I did not delete anything.",
        }
        return {"response": responses[message]}

    monkeypatch.setattr(module, "_isolated_api", fake_isolated_api)
    monkeypatch.setattr(module, "_request_json", fake_request_json)
    monkeypatch.setattr(module, "_request_text", fake_request_text)
    monkeypatch.setattr(module, "_post_chat", fake_post_chat)
    monkeypatch.setattr(module, "_git_status_short", lambda: "")

    checks = module.run(timeout=1.0)

    assert all(check.ok for check in checks), [(check.name, check.evidence) for check in checks if not check.ok]
    assert any(check.name == "real user state not targeted" for check in checks)
    assert any(check.name == "Plan Mode gates package install" for check in checks)


def test_first_run_smoke_rejects_false_podman_missing(monkeypatch, tmp_path) -> None:
    module = _load_first_run_smoke()
    home = tmp_path / "home"
    (home / ".local/share/personal-agent").mkdir(parents=True)
    (home / ".config/personal-agent").mkdir(parents=True)
    log_path = tmp_path / "api.log"
    log_path.write_text("", encoding="utf-8")

    @contextmanager
    def fake_isolated_api(timeout: float) -> Iterator[tuple[str, Path, Path]]:
        del timeout
        yield "http://127.0.0.1:9999", home, log_path

    def fake_request_json(method: str, base_url: str, path: str, **kwargs: Any) -> dict[str, Any]:
        del method, base_url, kwargs
        payloads = {
            "/ready": {"http_status": 200, "runtime_mode": "BOOTSTRAP_REQUIRED"},
            "/state": {"http_status": 200, "ok": True},
            "/version": {"http_status": 200, "runtime_instance": "dev", "git_commit": "testcommit"},
            "/telegram/status": {"http_status": 200, "configured": False, "effective_state": "disabled_optional"},
            "/search/status": {"http_status": 200, "enabled": False, "search_state": "never_configured", "reason": "search_disabled"},
        }
        return payloads[path]

    def fake_post_chat(base_url: str, message: str, *, thread_id: str, timeout: float = 20.0) -> dict[str, Any]:
        del base_url, thread_id, timeout
        if message == "what is dots.tts?":
            return {"response": "Search is not configured. This machine is missing Podman."}
        return {"response": "ok"}

    monkeypatch.setattr(module, "_isolated_api", fake_isolated_api)
    monkeypatch.setattr(module, "_request_json", fake_request_json)
    monkeypatch.setattr(module, "_request_text", lambda *args, **kwargs: (200, "<html></html>"))
    monkeypatch.setattr(module, "_post_chat", fake_post_chat)
    monkeypatch.setattr(module, "_git_status_short", lambda: "")

    checks = module.run(timeout=1.0)
    failures = [check for check in checks if not check.ok]

    assert any(check.name == "search setup guidance is safe" for check in failures)

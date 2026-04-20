#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import configparser
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STABLE_SERVICE = "personal-agent-api.service"
DEV_SERVICE = "personal-agent-api-dev.service"
STABLE_PORT = 8765
STABLE_URL = "http://127.0.0.1:8765"
DATA_HOME = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
STABLE_LAUNCHER = DATA_HOME / "personal-agent" / "bin" / "personal-agent-webui"
STABLE_DESKTOP = DATA_HOME / "applications" / "personal-agent.desktop"


def _log(message: str) -> None:
    print(f"[split-smoke] {message}", flush=True)


def _run_systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _service_state(unit: str) -> dict[str, str]:
    result = _run_systemctl("show", unit, "-p", "ActiveState", "-p", "MainPID", "-p", "UnitFileState")
    if result.returncode != 0:
        return {}
    state: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        state[key.strip()] = value.strip()
    return state


def _wait_for(predicate: Callable[[], bool], *, timeout_seconds: float = 15.0, poll_seconds: float = 0.25) -> bool:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(float(poll_seconds))
    return False


def _fetch_ready(url: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{url.rstrip('/')}/ready", method="GET")
    with urllib.request.urlopen(request, timeout=5.0) as response:
        raw = response.read().decode("utf-8", errors="replace")
    payload = json.loads(raw or "{}")
    return payload if isinstance(payload, dict) else {}


def _desktop_entry() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(STABLE_DESKTOP, encoding="utf-8")
    return parser


def _launcher_summary() -> tuple[str, str]:
    launcher_target = str(STABLE_LAUNCHER)
    try:
        launcher_target = str(STABLE_LAUNCHER.resolve())
    except Exception:
        launcher_target = str(STABLE_LAUNCHER)
    return str(STABLE_LAUNCHER), launcher_target


def _assert_stable_launcher_points_at_stable_runtime() -> None:
    if not STABLE_DESKTOP.is_file():
        raise RuntimeError(f"desktop entry missing: {STABLE_DESKTOP}")
    if not STABLE_LAUNCHER.exists():
        raise RuntimeError(f"launcher missing: {STABLE_LAUNCHER}")

    parser = _desktop_entry()
    if "Desktop Entry" not in parser:
        raise RuntimeError("desktop entry missing [Desktop Entry] block")
    entry = parser["Desktop Entry"]
    launcher_path, launcher_target = _launcher_summary()
    if str(entry.get("Exec") or "") != launcher_path:
        raise RuntimeError(f"desktop Exec does not point at stable launcher: {entry.get('Exec')}")
    if str(entry.get("TryExec") or "") != launcher_path:
        raise RuntimeError(f"desktop TryExec does not point at stable launcher: {entry.get('TryExec')}")
    if not str(launcher_target).endswith("/runtime/current/bin/personal-agent-webui"):
        raise RuntimeError(f"stable launcher does not resolve to runtime/current: {launcher_target}")

    launcher_text = Path(launcher_target).read_text(encoding="utf-8")
    if 'AGENT_LAUNCHER_SERVICE_NAME="personal-agent-api.service"' not in launcher_text:
        raise RuntimeError("stable launcher is not wired to the stable service")
    if 'AGENT_WEBUI_URL="http://127.0.0.1:8765/"' not in launcher_text:
        raise RuntimeError("stable launcher is not wired to the stable stable URL")


def _summary() -> list[str]:
    launcher_path, launcher_target = _launcher_summary()
    try:
        runtime_root = Path(launcher_target).resolve().parents[1]
    except Exception:
        runtime_root = Path("unknown")
    return [
        f"runtime_instance: stable",
        f"runtime_root: {runtime_root}",
        f"service_name: {STABLE_SERVICE}",
        f"launcher_target: {launcher_path}",
        f"launcher_resolved_target: {launcher_target}",
        f"api_base_url: {STABLE_URL}",
        f"api_port: {STABLE_PORT}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that the stable daily-driver stays healthy when the dev checkout process dies."
    )
    parser.add_argument("--stable-service", default=STABLE_SERVICE)
    parser.add_argument("--dev-service", default=DEV_SERVICE)
    parser.add_argument("--stable-url", default=STABLE_URL)
    parser.add_argument("--no-cleanup", action="store_true", help="Leave the dev service in its post-smoke state.")
    args = parser.parse_args(argv)

    stable_service = str(args.stable_service)
    dev_service = str(args.dev_service)
    stable_url = str(args.stable_url)
    dev_was_active = False
    dev_main_pid = 0

    stable_state = _service_state(stable_service)
    if str(stable_state.get("ActiveState") or "").strip() != "active":
        raise SystemExit(f"{stable_service} is not active")

    _assert_stable_launcher_points_at_stable_runtime()

    dev_state_before = _service_state(dev_service)
    dev_was_active = str(dev_state_before.get("ActiveState") or "").strip() == "active"
    _log(f"starting {dev_service}")
    start = _run_systemctl("start", dev_service)
    if start.returncode != 0:
        raise SystemExit(start.stderr.strip() or f"failed to start {dev_service}")
    try:
        if not _wait_for(lambda: str(_service_state(dev_service).get("ActiveState") or "") == "active"):
            raise SystemExit(f"{dev_service} did not become active")

        dev_state = _service_state(dev_service)
        dev_main_pid = int(str(dev_state.get("MainPID") or "0") or 0)
        if dev_main_pid <= 0:
            raise SystemExit(f"{dev_service} has no main pid")

        _log(f"signalling {dev_service} pid={dev_main_pid}")
        try:
            os.kill(dev_main_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        if not _wait_for(
            lambda: (
                int(str(_service_state(dev_service).get("MainPID") or "0") or 0) != dev_main_pid
                or str(_service_state(dev_service).get("ActiveState") or "").strip() != "active"
            ),
            timeout_seconds=20.0,
        ):
            raise SystemExit(f"{dev_service} did not stop or restart after process death")

        stable_state_after = _service_state(stable_service)
        if str(stable_state_after.get("ActiveState") or "").strip() != "active":
            raise SystemExit(f"{stable_service} stopped after dev process death")

        ready = _fetch_ready(stable_url)
        if not bool(ready.get("ready", False)):
            raise SystemExit(f"{stable_service} /ready is not healthy: {ready}")

        _assert_stable_launcher_points_at_stable_runtime()
    finally:
        if not bool(args.no_cleanup):
            if dev_was_active:
                _log(f"restoring {dev_service}")
                _run_systemctl("restart", dev_service)
            else:
                _run_systemctl("stop", dev_service)

    for line in _summary():
        _log(line)
    _log("stable service stayed healthy while the dev checkout process died")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

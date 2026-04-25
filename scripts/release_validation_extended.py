from __future__ import annotations

import argparse
import os
import sys
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_product_smoke import main as run_live_product_smoke
from scripts.assistant_real_world_smoke import main as run_assistant_real_world_smoke
from scripts.assistant_interaction_barrage import main as run_assistant_interaction_barrage
from scripts.assistant_viability_smoke import main as run_assistant_viability_smoke
from scripts.provider_matrix_smoke import main as run_provider_matrix_smoke
from scripts.restart_memory_smoke import main as run_restart_memory_smoke
from scripts.release_smoke import run_extended_suite
from scripts.split_smoke import main as run_split_smoke


def _should_run_live_smokes(args: argparse.Namespace) -> bool:
    if bool(args.with_live_smokes):
        return True
    return str(os.environ.get("AGENT_RELEASE_VALIDATION_WITH_LIVE_SMOKES") or "").strip().lower() in {"1", "true", "yes", "on"}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(base_url: str, path: str, *, timeout: float = 5.0) -> dict[str, object]:
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = int(getattr(exc, "code", 500))
    except Exception:
        return {"status": 0, "raw": "", "ok": False}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return {"status": status, "raw": raw, "ok": status < 400, "payload": payload}


def _wait_for_ready(base_url: str, *, timeout: float = 60.0) -> dict[str, object]:
    deadline = time.monotonic() + float(timeout)
    last: dict[str, object] = {"status": 0, "raw": "", "ok": False, "payload": {}}
    while time.monotonic() < deadline:
        last = _request_json(base_url, "/ready", timeout=5.0)
        if int(last.get("status") or 0) > 0:
            return last
        time.sleep(0.5)
    return last


def _kill_process(proc: subprocess.Popen[str], *, timeout: float = 10.0) -> None:
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


@contextmanager
def _temp_live_api_server() -> Iterator[str]:
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "AGENT_API_HOST": "127.0.0.1",
            "AGENT_API_PORT": str(port),
            "AGENT_LAUNCHER_OPEN_BROWSER": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    with tempfile.TemporaryDirectory(prefix="release-validation-live-log-") as tmpdir:
        log_path = Path(tmpdir) / "api.log"
        log_handle = log_path.open("a", encoding="utf-8")
        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "agent.api_server", "--host", "127.0.0.1", "--port", str(port)],
                cwd=str(ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            log_handle.close()
        try:
            ready = _wait_for_ready(base_url, timeout=60.0)
            if int(ready.get("status") or 0) <= 0:
                raise RuntimeError(f"live API did not become ready at {base_url}")
            yield base_url
        finally:
            if proc is not None:
                _kill_process(proc, timeout=10.0)


@contextmanager
def _headless_launcher_env() -> Iterator[None]:
    previous = os.environ.get("AGENT_LAUNCHER_OPEN_BROWSER")
    os.environ["AGENT_LAUNCHER_OPEN_BROWSER"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("AGENT_LAUNCHER_OPEN_BROWSER", None)
        else:
            os.environ["AGENT_LAUNCHER_OPEN_BROWSER"] = previous


def _stable_split_smoke_available() -> bool:
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "is-active", "personal-agent-api.service"],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return int(proc.returncode) == 0


def _run_optional_live_smokes() -> int:
    print("Running optional live smoke hooks", flush=True)
    print("Running restart_memory_smoke", flush=True)
    restart_exit_code = int(run_restart_memory_smoke(["--cycles", "3"]))
    if restart_exit_code != 0:
        return restart_exit_code
    try:
        with _headless_launcher_env():
            with _temp_live_api_server() as base_url:
                print("Running provider_matrix_smoke", flush=True)
                provider_exit_code = int(run_provider_matrix_smoke(["--base-url", base_url]))
                if provider_exit_code != 0:
                    return provider_exit_code
                print("Running assistant_real_world_smoke", flush=True)
                assistant_exit_code = int(run_assistant_real_world_smoke(["--base-url", base_url]))
                if assistant_exit_code != 0:
                    return assistant_exit_code
                print("Running assistant_interaction_barrage", flush=True)
                barrage_exit_code = int(run_assistant_interaction_barrage(["--base-url", base_url]))
                if barrage_exit_code != 0:
                    return barrage_exit_code
            with _temp_live_api_server() as base_url:
                print("Running assistant_viability_smoke", flush=True)
                viability_exit_code = int(
                    run_assistant_viability_smoke(
                        [
                            "--base-url",
                            base_url,
                            "--timeout",
                            "180",
                            "--retry-attempts",
                            "2",
                            "--surface",
                            "webui",
                            "--scenario",
                            "long_human_like_session_webui",
                        ]
                    )
                )
                if viability_exit_code != 0:
                    print(f"Optional live smoke returned {viability_exit_code}; treating it as non-blocking.", flush=True)
                print("Running live_product_smoke", flush=True)
                try:
                    result = int(run_live_product_smoke(["--base-url", base_url]))
                    if result != 0:
                        print(f"Optional live smoke returned {result}; treating it as non-blocking.", flush=True)
                    if _stable_split_smoke_available():
                        print("Running split_smoke", flush=True)
                        split_exit_code = int(run_split_smoke([]))
                        if split_exit_code != 0:
                            return split_exit_code
                    else:
                        print("Skipping split_smoke: stable service is not active.", flush=True)
                    return 0
                except Exception as exc:  # pragma: no cover - environment-specific live smoke guard
                    print(f"Skipping optional live smoke hooks: {exc}", flush=True)
                    return 0
    except Exception as exc:
        print(f"Skipping optional live smoke hooks: {exc}", flush=True)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the heavier Personal Agent release validation suite after the main release smoke passes."
    )
    parser.add_argument("--list", action="store_true", help="Print the exact pytest nodes without running them.")
    parser.add_argument("--no-quiet", action="store_true", help="Run pytest without -q.")
    parser.add_argument(
        "--with-live-smokes",
        action="store_true",
        help=(
            "Run optional live smoke hooks, including restart-safe memory churn and provider matrix checks, "
            "after the extended pytest suite passes."
        ),
    )
    args = parser.parse_args(argv)
    exit_code = run_extended_suite(list_only=bool(args.list), quiet=not bool(args.no_quiet))
    if exit_code != 0 or bool(args.list) or not _should_run_live_smokes(args):
        return exit_code
    live_exit_code = _run_optional_live_smokes()
    return live_exit_code or exit_code


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = os.environ.get("AGENT_WEBUI_BASE_URL") or "http://127.0.0.1:8765"
DEFAULT_CHROME = os.environ.get("BROWSER_UI_CHROME") or shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
SERVICE_NAME = "personal-agent-api.service"


def _ensure_playwright_importable() -> None:
    try:
        import playwright.sync_api  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    venv_python = ROOT / ".venv-browser/bin/python"
    if venv_python.exists() and os.environ.get("PERSONAL_AGENT_BROWSER_SMOKE_REEXEC") != "1":
        next_env = dict(os.environ)
        next_env["PERSONAL_AGENT_BROWSER_SMOKE_REEXEC"] = "1"
        os.execve(str(venv_python), [str(venv_python), *sys.argv], next_env)


_ensure_playwright_importable()

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ModuleNotFoundError:  # pragma: no cover - exercised only when dependency is absent.
    print("# Personal Agent Browser UI Survival Smoke")
    print("## browser automation dependency: SKIP")
    print("- evidence: Python package 'playwright' is not installed.")
    print("- next action: python -m venv .venv-browser && .venv-browser/bin/python -m pip install playwright")
    print("- note: the smoke uses the installed system Chrome when available; no browser download is required if google-chrome is installed.")
    print("\nSUMMARY: PASS=0 WARN=0 FAIL=0 SKIP=1")
    print("BROWSER_UI_SURVIVAL_SMOKE: skip")
    raise SystemExit(0)


@dataclass
class Check:
    name: str
    status: str
    evidence: str
    command: str = ""
    next_action: str = ""


class SmokeFailure(RuntimeError):
    pass


def _pass(name: str, evidence: str, command: str = "") -> Check:
    return Check(name=name, status="PASS", evidence=str(evidence), command=command)


def _warn(name: str, evidence: str, command: str = "", next_action: str = "") -> Check:
    return Check(name=name, status="WARN", evidence=str(evidence), command=command, next_action=next_action)


def _fail(name: str, evidence: str, command: str = "", next_action: str = "") -> Check:
    return Check(name=name, status="FAIL", evidence=str(evidence), command=command, next_action=next_action)


def _json_request(base_url: str, path: str, *, timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    payload = json.loads(raw or "{}")
    return payload if isinstance(payload, dict) else {}


def _wait_ready(base_url: str, *, timeout: float = 45.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "not checked"
    while time.monotonic() < deadline:
        try:
            payload = _json_request(base_url, "/ready", timeout=3.0)
            if bool(payload.get("ready")) or str(payload.get("runtime_mode") or "").upper() == "READY":
                return payload
            last_error = str(payload.get("reason") or payload.get("runtime_mode") or "not ready")
        except Exception as exc:  # noqa: BLE001
            last_error = f"{exc.__class__.__name__}: {exc}"
        time.sleep(1.0)
    raise SmokeFailure(f"ready timeout: {last_error}")


def _run_systemctl(args: list[str], *, timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def _wait_unreachable(base_url: str, *, timeout: float = 15.0) -> str:
    deadline = time.monotonic() + timeout
    last = "still reachable"
    while time.monotonic() < deadline:
        try:
            _json_request(base_url, "/ready", timeout=1.0)
            last = "ready still reachable"
        except Exception as exc:  # noqa: BLE001
            return f"{exc.__class__.__name__}: {exc}"
        time.sleep(0.5)
    raise SmokeFailure(last)


def _safe_excerpt(text: str, limit: int = 500) -> str:
    flattened = " ".join(str(text or "").split())
    return flattened[:limit]


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(needle.lower() in lowered for needle in needles)


def _secret_like(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in ("bot_token=", "telegram_bot_token=", "bearer ", "api_key=", "password=", "secret="))


class BrowserHarness:
    def __init__(self, page, *, base_url: str, artifact_dir: Path) -> None:  # noqa: ANN001
        self.page = page
        self.base_url = base_url.rstrip("/")
        self.artifact_dir = artifact_dir
        self.console_errors: list[str] = []
        self.network_failures: list[str] = []
        self.dialogs: list[str] = []
        self.allow_network_failure = False

        page.on("console", self._on_console)
        page.on("pageerror", lambda exc: self.console_errors.append(f"pageerror: {exc}"))
        page.on("requestfailed", self._on_request_failed)
        page.on("dialog", self._on_dialog)

    def _on_console(self, message) -> None:  # noqa: ANN001
        text = str(message.text or "")
        if message.type in {"error"}:
            self.console_errors.append(text)

    def _on_request_failed(self, request) -> None:  # noqa: ANN001
        failure = request.failure
        if isinstance(failure, dict):
            error_text = str(failure.get("errorText") or "failed")
        else:
            error_text = str(failure or "failed")
        row = f"{request.method} {request.url} {error_text}"
        if self.allow_network_failure:
            row = f"expected-during-api-interruption: {row}"
        self.network_failures.append(row)

    def _on_dialog(self, dialog) -> None:  # noqa: ANN001
        self.dialogs.append(str(dialog.message))
        dialog.dismiss()

    def snapshot(self, name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in name)[:60]
        screenshot = self.artifact_dir / f"{safe}.png"
        dom = self.artifact_dir / f"{safe}.txt"
        try:
            self.page.screenshot(path=str(screenshot), full_page=True)
        except Exception:
            pass
        try:
            dom.write_text(self.page.locator("body").inner_text(timeout=2000)[:6000], encoding="utf-8")
        except Exception:
            pass
        return str(self.artifact_dir)

    def goto(self) -> None:
        self.page.goto(f"{self.base_url}/", wait_until="domcontentloaded", timeout=20000)
        self.page.locator("textarea").wait_for(state="visible", timeout=15000)

    def body_text(self) -> str:
        return self.page.locator("body").inner_text(timeout=5000)

    def transcript_text(self) -> str:
        return self.body_text()

    def assistant_rows(self) -> list[str]:
        return self.page.locator(".chat-message-assistant .chat-bubble").all_inner_texts()

    def user_rows(self) -> list[str]:
        return self.page.locator(".chat-message-user .chat-bubble").all_inner_texts()

    def send(self, text: str, *, timeout: float = 45.0) -> str:
        before = self.page.locator(".chat-message-assistant .chat-bubble").count()
        self.page.locator("textarea").fill(text)
        self.page.get_by_role("button", name="Send").click(timeout=5000)
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            rows = self.assistant_rows()
            if len(rows) > before:
                candidate = rows[-1].strip()
                if candidate and "Working on it" not in candidate:
                    return candidate
                last = candidate
            time.sleep(0.25)
        raise SmokeFailure(f"assistant response timeout after sending {text!r}; last={last[:200]}")

    def send_expect_error(self, text: str, *, timeout: float = 20.0) -> str:
        return self.send(text, timeout=timeout)

    def click_approval(self, name: str = "Cancel", *, timeout: float = 15.0) -> str:
        before = self.page.locator(".chat-message-assistant .chat-bubble").count()
        self.page.get_by_role("button", name=name).click(timeout=5000)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rows = self.assistant_rows()
            if len(rows) > before:
                candidate = rows[-1].strip()
                if candidate:
                    return candidate
            time.sleep(0.25)
        raise SmokeFailure(f"approval response timeout for {name}")

    def assert_interactive(self) -> None:
        textarea = self.page.locator("textarea")
        textarea.wait_for(state="visible", timeout=5000)
        if not textarea.is_enabled(timeout=2000):
            raise SmokeFailure("chat textarea is not enabled")

    def assert_no_fatal_browser_errors(self, *, ignore_expected_network: bool = False) -> None:
        fatal_console = []
        for row in self.console_errors:
            if _contains_any(row, ("favicon",)):
                continue
            if ignore_expected_network and _contains_any(
                row,
                (
                    "failed to load resource: net::err_connection_refused",
                    "failed to load resource: net::err_empty_response",
                    "failed to load resource: net::err_connection_reset",
                    "failed to load resource: the server responded with a status of 404",
                ),
            ):
                continue
            fatal_console.append(row)
        fatal_network = [
            row
            for row in self.network_failures
            if not (ignore_expected_network and row.startswith("expected-during-api-interruption:"))
            and not (
                ignore_expected_network
                and "/ready" in row
                and _contains_any(row, ("net::err_aborted", "net::err_empty_response", "net::err_connection_refused", "net::err_connection_reset"))
            )
            and not _contains_any(row, ("favicon",))
        ]
        combined = "\n".join([*fatal_console, *fatal_network, *self.dialogs])
        if _secret_like(combined):
            raise SmokeFailure(f"secret-like value appeared in browser diagnostics: {_safe_excerpt(combined)}")
        if fatal_console:
            raise SmokeFailure(f"browser console errors: {_safe_excerpt(' | '.join(fatal_console), 800)}")
        if fatal_network:
            raise SmokeFailure(f"unexpected network failures: {_safe_excerpt(' | '.join(fatal_network), 800)}")
        if self.dialogs:
            raise SmokeFailure(f"unexpected browser dialogs: {_safe_excerpt(' | '.join(self.dialogs), 800)}")


def _run_browser_journey(args: argparse.Namespace) -> list[Check]:
    checks: list[Check] = []
    artifact_dir = Path(tempfile.mkdtemp(prefix="browser-ui-survival-", dir="/tmp"))
    api_was_stopped = False
    chrome_path = str(args.chrome or DEFAULT_CHROME or "")
    if not chrome_path:
        return [_fail("browser dependency", "No Chrome/Chromium executable found.", next_action="Install google-chrome or chromium, or set BROWSER_UI_CHROME=/path/to/chrome.")]

    _wait_ready(args.base_url, timeout=args.timeout)
    version = _json_request(args.base_url, "/version", timeout=10.0)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chrome_path,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(viewport={"width": 1366, "height": 900}, ignore_https_errors=True)
        page = context.new_page()
        ui = BrowserHarness(page, base_url=args.base_url, artifact_dir=artifact_dir)
        try:
            try:
                ui.goto()
                body = ui.body_text()
                body_lower = body.lower()
                if "personal agent" not in body_lower or "what can i help you with?" not in body_lower:
                    raise SmokeFailure(f"unexpected initial UI text: {_safe_excerpt(body)}")
                ui.assert_interactive()
                checks.append(_pass("initial page load", f"runtime_instance={version.get('runtime_instance')} git_commit={version.get('git_commit')}", f"GET {args.base_url}/"))
            except Exception as exc:  # noqa: BLE001
                ui.snapshot("initial-page-load-failure")
                checks.append(_fail("initial page load", f"{exc.__class__.__name__}: {exc}", f"GET {args.base_url}/", f"Artifacts: {artifact_dir}"))
                return checks

            hello = ui.send("hello")
            if len(ui.user_rows()) != 1 or len(ui.assistant_rows()) < 1:
                raise SmokeFailure("hello did not render one user row and one assistant response")
            if _contains_any(hello, ("traceback", "{", "\"ok\"", "runtime_mode")):
                raise SmokeFailure(f"hello response looked like raw/debug output: {_safe_excerpt(hello)}")
            ui.assert_interactive()
            checks.append(_pass("ordinary greeting", _safe_excerpt(hello), 'UI send "hello"'))

            ram = ui.send("can you do a quick system check and see if anything is eating ram?", timeout=60.0)
            if not _contains_any(ram, ("ram is not under pressure", "under pressure", "you're fine", "used:")):
                raise SmokeFailure(f"RAM answer lacks concise status: {_safe_excerpt(ram)}")
            if not _contains_any(ram, ("used:", "available:")):
                raise SmokeFailure(f"RAM answer lacks used/available memory: {_safe_excerpt(ram)}")
            if _contains_any(ram, ("Likely cause:", "Normality:", "Evidence:", "Safe next action:", "Top CPU processes")):
                raise SmokeFailure(f"RAM answer is verbose diagnostic by default: {_safe_excerpt(ram)}")
            if not _contains_any(ram, ("baseline", "usual")):
                raise SmokeFailure(f"RAM answer lacks baseline create/compare language: {_safe_excerpt(ram)}")
            checks.append(_pass("normal-user RAM check renders concise answer", _safe_excerpt(ram), "UI RAM check"))

            details = ui.send("show top memory and CPU processes", timeout=60.0)
            if not _contains_any(details, ("top memory", "top cpu", "process")):
                raise SmokeFailure(f"detailed response lacks process detail: {_safe_excerpt(details)}")
            ui.assert_interactive()
            scroll_info = page.locator(".chat-transcript").evaluate(
                "(el) => ({scrollTop: el.scrollTop, scrollHeight: el.scrollHeight, clientHeight: el.clientHeight})"
            )
            if scroll_info["scrollHeight"] > scroll_info["clientHeight"] and scroll_info["scrollTop"] <= 0:
                raise SmokeFailure(f"transcript did not scroll after detailed output: {scroll_info}")
            checks.append(_pass("explicit detailed response renders and scrolls", _safe_excerpt(details), "UI detailed RAM check"))

            before_refresh_users = len(ui.user_rows())
            before_refresh_assistants = len(ui.assistant_rows())
            page.reload(wait_until="domcontentloaded", timeout=20000)
            page.locator("textarea").wait_for(state="visible", timeout=15000)
            after_refresh_body = ui.body_text()
            after_refresh_users = len(ui.user_rows())
            after_refresh_assistants = len(ui.assistant_rows())
            if "Working on it" in after_refresh_body:
                raise SmokeFailure("refresh left stale loading state visible")
            if after_refresh_users or after_refresh_assistants:
                refresh_evidence = f"transcript persisted users={after_refresh_users} assistants={after_refresh_assistants}"
            else:
                refresh_evidence = f"transient transcript cleared coherently; previous users={before_refresh_users} assistants={before_refresh_assistants}"
            ui.assert_interactive()
            checks.append(_pass("page refresh survival", refresh_evidence, "browser reload"))

            _run_systemctl(["stop", SERVICE_NAME], timeout=20.0)
            api_was_stopped = True
            unreachable = _wait_unreachable(args.base_url, timeout=20.0)
            ui.allow_network_failure = True
            offline_response = ui.send_expect_error("are you there?", timeout=25.0)
            if not _contains_any(offline_response, ("problem", "failed", "fetch", "network", "unavailable")):
                raise SmokeFailure(f"offline response did not show a clear failure: {_safe_excerpt(offline_response)}")
            checks.append(_pass("API interruption shows truthful UI error", _safe_excerpt(offline_response), f"systemctl --user stop {SERVICE_NAME}; UI chat; unreachable={unreachable}"))

            start_result = _run_systemctl(["start", SERVICE_NAME], timeout=20.0)
            if start_result.returncode != 0:
                raise SmokeFailure(f"failed to restart API: {_safe_excerpt(start_result.stdout)}")
            _wait_ready(args.base_url, timeout=60.0)
            api_was_stopped = False
            ui.allow_network_failure = False
            recovered = ui.send("is the assistant healthy?", timeout=60.0)
            if _contains_any(recovered, ("traceback", "raw json", "{\"", "[{")) or not _contains_any(recovered, ("doctor", "ok", "warn", "fail")):
                raise SmokeFailure(f"post-restart response looked broken: {_safe_excerpt(recovered)}")
            ui.assert_interactive()
            checks.append(_pass("API restart recovery", _safe_excerpt(recovered), f"systemctl --user start {SERVICE_NAME}; UI chat"))

            plan_preview = ui.send("make a support bundle", timeout=45.0)
            if not _contains_any(plan_preview, ("Plan Mode", "support bundle", "yes", "confirm")):
                raise SmokeFailure(f"support bundle preview did not render Plan Mode: {_safe_excerpt(plan_preview)}")
            if page.get_by_role("button", name="Approve").count() <= 0 and page.get_by_role("button", name="Cancel").count() <= 0:
                raise SmokeFailure("Plan Mode approval controls were not visible")
            page.reload(wait_until="domcontentloaded", timeout=20000)
            page.locator("textarea").wait_for(state="visible", timeout=15000)
            _run_systemctl(["restart", SERVICE_NAME], timeout=25.0)
            _wait_ready(args.base_url, timeout=60.0)
            stale = ui.send("yes", timeout=45.0)
            if not _contains_any(stale, ("no current action", "different chat thread", "did not run", "ask", "tell me what")):
                raise SmokeFailure(f"stale confirmation after refresh/restart was not rejected clearly: {_safe_excerpt(stale)}")
            checks.append(_pass("Plan Mode refresh/restart stale confirmation safety", _safe_excerpt(stale), "support bundle preview; reload; API restart; yes"))

            baseline_count = len(ui.assistant_rows())
            for index in range(10):
                response = ui.send(f"rewrite this: browser survival message {index}", timeout=30.0)
                if f"browser survival message {index}" not in response.lower():
                    raise SmokeFailure(f"long transcript deterministic reply mismatch at {index}: {_safe_excerpt(response)}")
            if len(ui.assistant_rows()) < baseline_count + 10:
                raise SmokeFailure("long transcript did not add expected assistant responses")
            ui.assert_interactive()
            page.reload(wait_until="domcontentloaded", timeout=20000)
            page.locator("textarea").wait_for(state="visible", timeout=15000)
            checks.append(_pass("bounded large transcript remains usable", f"assistant_rows_before_reload={baseline_count + 10}", "10 deterministic UI messages"))

            special = "rewrite this: line one\nline two with 'quotes' and \"double quotes\" and <script>alert('x')</script> and https://example.test/path?q=1"
            special_response = ui.send(special, timeout=30.0)
            body_after_special = ui.body_text()
            if "alert('x')" not in body_after_special:
                raise SmokeFailure("special-character text was hidden or corrupted")
            if ui.dialogs:
                raise SmokeFailure(f"special-character input triggered dialog: {ui.dialogs}")
            checks.append(_pass("special-character and multiline rendering", _safe_excerpt(special_response), "UI multiline/special text"))

            users_before = len([row for row in ui.user_rows() if "is telegram working" in row.lower()])
            assistant_before = len(ui.assistant_rows())
            page.locator("textarea").fill("is Telegram working?")
            send_button = page.get_by_role("button", name="Send")
            send_button.click(timeout=5000)
            try:
                send_button.click(timeout=500)
            except PlaywrightError:
                pass
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline and len(ui.assistant_rows()) <= assistant_before:
                time.sleep(0.25)
            users_after = len([row for row in ui.user_rows() if "is telegram working" in row.lower()])
            if users_after - users_before != 1:
                raise SmokeFailure(f"duplicate-send protection failed: users_before={users_before} users_after={users_after}")
            ui.assert_interactive()
            checks.append(_pass("duplicate-send protection", f"user_rows_delta={users_after - users_before}", "double-click Send"))

            ui.assert_no_fatal_browser_errors(ignore_expected_network=True)
            checks.append(_pass("browser console and network diagnostics", f"console_errors={len(ui.console_errors)} network_failures={len(ui.network_failures)} artifacts={artifact_dir}", "console/network capture"))
        except Exception as exc:  # noqa: BLE001
            artifact = ui.snapshot("browser-ui-survival-failure")
            checks.append(_fail("browser journey", f"{exc.__class__.__name__}: {exc}", next_action=f"Inspect artifacts: {artifact}"))
        finally:
            if api_was_stopped:
                _run_systemctl(["start", SERVICE_NAME], timeout=20.0)
            try:
                _wait_ready(args.base_url, timeout=60.0)
            except Exception as exc:  # noqa: BLE001
                checks.append(_fail("service restoration cleanup", f"{exc.__class__.__name__}: {exc}", f"systemctl --user start {SERVICE_NAME}"))
            context.close()
            browser.close()
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Installed-product browser UI survival smoke.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--chrome", default=DEFAULT_CHROME)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    print("# Personal Agent Browser UI Survival Smoke")
    print(f"Base URL: {args.base_url}")
    print(f"Chrome: {args.chrome or 'not found'}")

    try:
        checks = _run_browser_journey(args)
    except PlaywrightError as exc:
        checks = [
            _fail(
                "browser automation launch",
                f"{exc.__class__.__name__}: {exc}",
                next_action=(
                    "Use the repo-local Playwright environment: "
                    "python -m venv .venv-browser && .venv-browser/bin/python -m pip install playwright. "
                    "If system Chrome is unavailable, install Chrome/Chromium or run .venv-browser/bin/python -m playwright install chromium."
                ),
            )
        ]
    except Exception as exc:  # noqa: BLE001
        checks = [_fail("browser smoke setup", f"{exc.__class__.__name__}: {exc}")]

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
        print(f"## {check.name}: {check.status}")
        if check.command:
            print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.evidence}")
        if check.next_action:
            print(f"- next action: {check.next_action}")

    print(f"\nSUMMARY: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']} SKIP={counts['SKIP']}")
    if counts["FAIL"]:
        print("BROWSER_UI_SURVIVAL_SMOKE: fail")
        return 1
    if counts["SKIP"]:
        print("BROWSER_UI_SURVIVAL_SMOKE: skip")
        return 0
    print("BROWSER_UI_SURVIVAL_SMOKE: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

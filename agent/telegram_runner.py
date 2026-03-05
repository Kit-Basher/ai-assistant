from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from agent.audit_log import AuditLog
from agent.logging_utils import log_event
from telegram_adapter.bot import (
    acquire_telegram_poll_lock,
    build_app,
    is_telegram_conflict_error,
    release_telegram_poll_lock,
    resolve_telegram_bot_token_with_source,
    telegram_conflict_backoff_seconds,
    telegram_poll_lock_path,
)


class TelegramRunner:
    """Run Telegram polling in-process with bounded retry/backoff."""

    def __init__(
        self,
        *,
        runtime: Any,
        log_path: str | None,
        audit_log: AuditLog | None,
        app_factory: Callable[..., Any] | None = None,
        token_resolver: Callable[[], tuple[str | None, str] | str | None] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._runtime = runtime
        self._log_path = log_path
        self._audit_log = audit_log
        self._app_factory = app_factory or build_app
        self._token_resolver = token_resolver or resolve_telegram_bot_token_with_source
        self._sleep_fn = sleep_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.embedded_running = False
        self.last_event = "idle"
        self.last_error: str | None = None
        self.last_ts = 0.0
        self.last_ts_iso: str | None = None
        self.token_source = "none"
        self.last_attempt = 0
        self._consecutive_failures = 0

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
        resolved = self._token_resolver()
        token_source = "resolver"
        if isinstance(resolved, tuple):
            token = str(resolved[0] or "").strip()
            token_source = str(resolved[1] or "resolver").strip() or "resolver"
        else:
            token = str(resolved or "").strip()
        self.token_source = token_source
        self._emit_event(
            "telegram.embedded.start",
            {"mode": "embedded", "token_source": token_source},
            running=False,
            error=None,
        )
        if not token:
            self._emit_event(
                "telegram.disabled",
                {"reason": "missing_token", "token_source": token_source, "mode": "embedded"},
                running=False,
                error="missing_token",
            )
            self._safe_audit(
                action="telegram.disabled",
                reason="missing_token",
                outcome="skipped",
                error_kind="missing_token",
            )
            return False
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._thread_main,
            kwargs={"token": token, "token_source": token_source},
            name="telegram-runner",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        thread: threading.Thread | None
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=5.0)
        self._emit_event(
            "telegram.stop",
            {"reason": "shutdown", "mode": "embedded", "token_source": self.token_source},
            running=False,
            error=None,
        )
        self._safe_audit(
            action="telegram.stop",
            reason="shutdown",
            outcome="success",
            error_kind=None,
        )

    def _thread_main(self, *, token: str, token_source: str) -> None:
        self._run_loop(token=token, token_source=token_source)

    def _run_loop(self, *, token: str, token_source: str, max_iters: int | None = None) -> None:
        consecutive_failures = 0
        iterations = 0
        while not self._stop_event.is_set():
            if max_iters is not None and iterations >= int(max_iters):
                break
            iterations += 1
            poll_lock = acquire_telegram_poll_lock(token)
            if poll_lock is None:
                consecutive_failures += 1
                self._consecutive_failures = int(consecutive_failures)
                backoff_seconds = telegram_conflict_backoff_seconds(consecutive_failures)
                lock_path = str(telegram_poll_lock_path(token))
                self._emit_event(
                    "telegram.crash",
                    {
                        "error_type": "Conflict",
                        "error": "poll_lock_held",
                        "message": (
                            "getUpdates conflict — another poller is active for this token. "
                            "Rotate token or stop the other instance."
                        ),
                        "lock_path": lock_path,
                        "consecutive_failures": int(consecutive_failures),
                        "backoff_seconds": int(backoff_seconds),
                        "mode": "embedded",
                        "token_source": token_source,
                        "attempt": int(iterations),
                    },
                    running=False,
                    error="Conflict: poll_lock_held",
                )
                self._emit_event(
                    "telegram.retry",
                    {
                        "backoff_seconds": int(backoff_seconds),
                        "consecutive_failures": int(consecutive_failures),
                        "mode": "embedded",
                        "token_source": token_source,
                        "attempt": int(iterations),
                    },
                    running=False,
                    error="Conflict: poll_lock_held",
                )
                self._safe_audit(
                    action="telegram.crash",
                    reason="Conflict",
                    outcome="failed",
                    error_kind="internal_error",
                )
                if self._wait(float(backoff_seconds)):
                    break
                continue
            try:
                app = self._app_factory(
                    config=self._runtime.config,
                    token=token,
                    llm_fixit_fn=self._runtime.llm_fixit,
                    llm_fixit_store=getattr(self._runtime, "_llm_fixit_store", None),
                    audit_log=self._audit_log,
                    runtime=self._runtime,
                )
                self._run_application_loop(
                    app,
                    token_source=token_source,
                    attempt=int(iterations),
                )
                consecutive_failures = 0
                self._consecutive_failures = 0
                if self._stop_event.is_set():
                    break
                raise RuntimeError("telegram_polling_stopped")
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                consecutive_failures += 1
                self._consecutive_failures = int(consecutive_failures)
                if is_telegram_conflict_error(exc):
                    backoff_seconds = telegram_conflict_backoff_seconds(consecutive_failures)
                    error_message = (
                        "getUpdates conflict — another poller is active for this token. "
                        "Rotate token or stop the other instance."
                    )
                else:
                    backoff_seconds = min(60, 5 * max(1, consecutive_failures))
                    error_message = str(exc)
                self._emit_event(
                    "telegram.crash",
                    {
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "message": error_message,
                        "consecutive_failures": int(consecutive_failures),
                        "backoff_seconds": int(backoff_seconds),
                        "mode": "embedded",
                        "token_source": token_source,
                        "attempt": int(iterations),
                    },
                    running=False,
                    error=f"{exc.__class__.__name__}: {error_message}",
                )
                self._emit_event(
                    "telegram.retry",
                    {
                        "backoff_seconds": int(backoff_seconds),
                        "consecutive_failures": int(consecutive_failures),
                        "mode": "embedded",
                        "token_source": token_source,
                        "attempt": int(iterations),
                    },
                    running=False,
                    error=f"{exc.__class__.__name__}: {error_message}",
                )
                self._safe_audit(
                    action="telegram.crash",
                    reason=exc.__class__.__name__,
                    outcome="failed",
                    error_kind="internal_error",
                )
                if self._wait(float(backoff_seconds)):
                    break
            finally:
                release_telegram_poll_lock(poll_lock)

    def _wait(self, seconds: float) -> bool:
        duration = max(0.0, float(seconds))
        if self._sleep_fn is None:
            return bool(self._stop_event.wait(duration))
        try:
            self._sleep_fn(duration)
        except Exception:
            return bool(self._stop_event.is_set())
        return bool(self._stop_event.is_set())

    def _run_application_loop(self, app: Any, *, token_source: str, attempt: int) -> None:
        asyncio.run(
            self._run_app_async(
                app,
                token_source=token_source,
                attempt=attempt,
            )
        )

    @staticmethod
    async def _await_maybe(maybe_callable: Any, **kwargs: Any) -> Any:
        if not callable(maybe_callable):
            return None
        result = maybe_callable(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _run_app_async(self, app: Any, *, token_source: str, attempt: int) -> None:
        updater = getattr(app, "updater", None)
        if updater is None:
            raise RuntimeError("telegram_updater_missing")

        app_started = False
        try:
            await self._await_maybe(getattr(app, "initialize", None))
            await self._await_maybe(getattr(app, "start", None))
            app_started = True

            polling_kwargs: dict[str, Any] = {"drop_pending_updates": True}
            allowed_updates = None
            try:
                from telegram import Update  # type: ignore

                allowed_updates = getattr(Update, "ALL_TYPES", None)
            except Exception:
                allowed_updates = None
            if allowed_updates is not None:
                polling_kwargs["allowed_updates"] = allowed_updates

            await self._await_maybe(getattr(updater, "start_polling", None), **polling_kwargs)

            started_payload = {
                "mode": "embedded",
                "attempt": int(attempt),
                "token_source": token_source,
            }
            self._emit_event("telegram.started", started_payload, running=True, error=None)
            self._consecutive_failures = 0
            self._safe_audit(
                action="telegram.start",
                reason="embedded",
                outcome="success",
                error_kind=None,
            )

            while not self._stop_event.is_set():
                await asyncio.sleep(0.25)
        finally:
            await self._await_maybe(getattr(updater, "stop", None))
            if app_started:
                await self._await_maybe(getattr(app, "stop", None))
            await self._await_maybe(getattr(app, "shutdown", None))
            self.embedded_running = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            embedded_running = bool(self.embedded_running)
            last_event = str(self.last_event or "")
            last_error = str(self.last_error or "") or None
            last_ts = float(self.last_ts or 0.0)
            last_ts_iso = str(self.last_ts_iso or "") or None
            token_source = str(self.token_source or "none")
            attempt = int(self.last_attempt or 0)
            consecutive_failures = int(self._consecutive_failures or 0)
        state = "stopped"
        if last_event == "telegram.disabled" and last_error == "missing_token":
            state = "disabled_missing_token"
        elif embedded_running:
            state = "running"
        elif last_event in {"telegram.crash", "telegram.retry"}:
            state = "crash_loop"
        elif last_event == "telegram.embedded.start" and not embedded_running:
            state = "starting"
        return {
            "state": state,
            "embedded_running": embedded_running,
            "last_event": last_event,
            "last_error": last_error,
            "last_ts": last_ts,
            "last_ts_iso": last_ts_iso,
            "token_source": token_source,
            "attempt": attempt,
            "consecutive_failures": consecutive_failures,
        }

    def _emit_event(
        self,
        action: str,
        payload: dict[str, Any],
        *,
        running: bool,
        error: str | None,
    ) -> None:
        now_ts = time.time()
        with self._lock:
            self.embedded_running = bool(running)
            self.last_event = str(action or "telegram.unknown")
            self.last_error = str(error).strip() if error else None
            self.last_ts = float(now_ts)
            self.last_ts_iso = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()
            try:
                self.last_attempt = int(payload.get("attempt") or self.last_attempt or 0)
            except Exception:
                pass
        self._safe_log(action, payload)

    def _safe_log(self, action: str, payload: dict[str, Any]) -> None:
        if str(action).startswith("telegram."):
            try:
                print(
                    f"{action} {json.dumps(payload if isinstance(payload, dict) else {}, ensure_ascii=True, sort_keys=True)}",
                    flush=True,
                )
            except Exception:
                pass
        if not self._log_path:
            return
        try:
            log_event(self._log_path, action, payload)
        except Exception:
            return

    def _safe_audit(
        self,
        *,
        action: str,
        reason: str,
        outcome: str,
        error_kind: str | None,
    ) -> None:
        if self._audit_log is None:
            return
        try:
            self._audit_log.append(
                actor="system",
                action=action,
                params={"mode": "embedded_telegram"},
                decision="allow",
                reason=str(reason or ""),
                dry_run=False,
                outcome=str(outcome or "unknown"),
                error_kind=error_kind,
                duration_ms=0,
            )
        except Exception:
            return

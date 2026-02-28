from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any, Callable

from agent.audit_log import AuditLog
from agent.logging_utils import log_event
from telegram_adapter.bot import build_app, resolve_telegram_bot_token_with_source


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
        if not token:
            self._safe_log(
                "telegram.disabled",
                {"reason": "missing_token", "token_source": token_source, "mode": "embedded"},
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
        self._safe_log("telegram.stop", {"reason": "shutdown"})
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
            try:
                app = self._app_factory(
                    config=self._runtime.config,
                    token=token,
                    llm_fixit_fn=self._runtime.llm_fixit,
                    llm_fixit_store=getattr(self._runtime, "_llm_fixit_store", None),
                    audit_log=self._audit_log,
                )
                self._run_application_loop(
                    app,
                    token_source=token_source,
                    attempt=int(iterations),
                )
                if self._stop_event.is_set():
                    break
                raise RuntimeError("telegram_polling_stopped")
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                consecutive_failures += 1
                backoff_seconds = min(60, 5 * max(1, consecutive_failures))
                self._safe_log(
                    "telegram.crash",
                    {
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "consecutive_failures": int(consecutive_failures),
                        "backoff_seconds": int(backoff_seconds),
                        "mode": "embedded",
                    },
                )
                self._safe_log(
                    "telegram.retry",
                    {
                        "backoff_seconds": int(backoff_seconds),
                        "consecutive_failures": int(consecutive_failures),
                        "mode": "embedded",
                    },
                )
                self._safe_audit(
                    action="telegram.crash",
                    reason=exc.__class__.__name__,
                    outcome="failed",
                    error_kind="internal_error",
                )
                if self._wait(float(backoff_seconds)):
                    break
            else:
                consecutive_failures = 0

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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        updater = getattr(app, "updater", None)
        started = False
        try:
            self._maybe_await(loop, getattr(app, "initialize", None))
            self._maybe_await(loop, getattr(app, "start", None))
            started = True
            polling_kwargs: dict[str, Any] = {"drop_pending_updates": True}
            allowed_updates = None
            try:
                from telegram import Update  # type: ignore

                allowed_updates = getattr(Update, "ALL_TYPES", None)
            except Exception:
                allowed_updates = None
            if allowed_updates is not None:
                polling_kwargs["allowed_updates"] = allowed_updates
            self._maybe_await(loop, getattr(updater, "start_polling", None), **polling_kwargs)
            self._safe_log(
                "telegram.started",
                {
                    "mode": "embedded",
                    "attempt": int(attempt),
                    "token_source": token_source,
                },
            )
            self._safe_audit(
                action="telegram.start",
                reason="embedded",
                outcome="success",
                error_kind=None,
            )
            while not self._stop_event.is_set():
                if self._wait(0.25):
                    break
        finally:
            self._maybe_await(loop, getattr(updater, "stop", None))
            if started:
                self._maybe_await(loop, getattr(app, "stop", None))
            self._maybe_await(loop, getattr(app, "shutdown", None))
            asyncio.set_event_loop(None)
            loop.close()

    @staticmethod
    def _maybe_await(loop: asyncio.AbstractEventLoop, maybe_callable: Any, **kwargs: Any) -> Any:
        if not callable(maybe_callable):
            return None
        result = maybe_callable(**kwargs)
        if inspect.isawaitable(result):
            return loop.run_until_complete(result)
        return result

    def _safe_log(self, action: str, payload: dict[str, Any]) -> None:
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

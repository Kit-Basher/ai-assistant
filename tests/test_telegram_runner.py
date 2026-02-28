from __future__ import annotations

import json
import os
import tempfile
import threading
import types
import unittest
import warnings

from agent.telegram_runner import TelegramRunner


class _AuditCapture:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def append(self, **kwargs: object) -> None:
        self.rows.append(dict(kwargs))


class TestTelegramRunner(unittest.TestCase):
    def test_start_and_stop_with_token_runs_embedded_polling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            started_event = threading.Event()
            runtime = types.SimpleNamespace(config=object(), llm_fixit=lambda payload: (True, {"ok": True}))
            audit = _AuditCapture()
            log_path = os.path.join(tmpdir, "agent.log")

            class _FakeUpdater:
                async def start_polling(self, **_kwargs: object) -> None:
                    started_event.set()

                async def stop(self) -> None:
                    return

            class _FakeApp:
                def __init__(self) -> None:
                    self.updater = _FakeUpdater()

                async def initialize(self) -> None:
                    return

                async def start(self) -> None:
                    return

                async def stop(self) -> None:
                    return

                async def shutdown(self) -> None:
                    return

            runner = TelegramRunner(
                runtime=runtime,
                log_path=log_path,
                audit_log=audit,  # type: ignore[arg-type]
                app_factory=lambda **_kwargs: _FakeApp(),
                token_resolver=lambda: ("token", "env"),
            )
            self.assertTrue(runner.start())
            self.assertTrue(started_event.wait(1.0))
            runner.stop()
            actions = [str(row.get("action")) for row in audit.rows]
            self.assertIn("telegram.start", actions)
            self.assertIn("telegram.stop", actions)
            with open(log_path, "r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
            started_rows = [row for row in events if str(row.get("type")) == "telegram.started"]
            self.assertTrue(started_rows)
            started_payload = started_rows[-1].get("payload") if isinstance(started_rows[-1], dict) else {}
            self.assertEqual("embedded", str((started_payload or {}).get("mode")))
            self.assertEqual("env", str((started_payload or {}).get("token_source")))

    def test_missing_token_disables_runner_and_logs_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "agent.log")
            runtime = types.SimpleNamespace(config=object(), llm_fixit=lambda payload: (True, {"ok": True}))
            audit = _AuditCapture()
            runner = TelegramRunner(
                runtime=runtime,
                log_path=log_path,
                audit_log=audit,  # type: ignore[arg-type]
                app_factory=lambda **_kwargs: None,
                token_resolver=lambda: (None, "missing"),
            )
            started = runner.start()
            self.assertFalse(started)
            with open(log_path, "r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
            self.assertTrue(any(str(item.get("type")) == "telegram.disabled" for item in events))
            self.assertTrue(any(str(row.get("action")) == "telegram.disabled" for row in audit.rows))

    def test_run_loop_crash_is_caught_and_backoff_applies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sleep_calls: list[float] = []
            app_calls: list[int] = []
            runtime = types.SimpleNamespace(config=object(), llm_fixit=lambda payload: (True, {"ok": True}))
            audit = _AuditCapture()
            log_path = os.path.join(tmpdir, "agent.log")

            def _failing_app_factory(**_kwargs: object) -> object:
                app_calls.append(1)
                raise RuntimeError("boom")

            runner = TelegramRunner(
                runtime=runtime,
                log_path=log_path,
                audit_log=audit,  # type: ignore[arg-type]
                app_factory=_failing_app_factory,
                token_resolver=lambda: ("token", "env"),
                sleep_fn=lambda seconds: sleep_calls.append(float(seconds)),
            )
            runner._run_loop(token="token", token_source="env", max_iters=2)
            self.assertEqual(2, len(app_calls))
            self.assertEqual([5.0, 10.0], sleep_calls)
            self.assertTrue(any(str(row.get("action")) == "telegram.crash" for row in audit.rows))
            with open(log_path, "r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
            event_types = [str(item.get("type")) for item in events]
            self.assertIn("telegram.crash", event_types)
            self.assertIn("telegram.retry", event_types)

    def test_async_polling_coroutine_is_awaited_without_runtime_warning(self) -> None:
        runtime = types.SimpleNamespace(config=object(), llm_fixit=lambda payload: (True, {"ok": True}))
        audit = _AuditCapture()
        awaited: dict[str, bool] = {"start_polling": False}

        class _FakeUpdater:
            async def start_polling(self, **_kwargs: object) -> None:
                awaited["start_polling"] = True

            async def stop(self) -> None:
                return

        class _FakeApp:
            def __init__(self) -> None:
                self.updater = _FakeUpdater()

            async def initialize(self) -> None:
                return

            async def start(self) -> None:
                return

            async def stop(self) -> None:
                return

            async def shutdown(self) -> None:
                return

        runner = TelegramRunner(
            runtime=runtime,
            log_path=None,
            audit_log=audit,  # type: ignore[arg-type]
            app_factory=lambda **_kwargs: _FakeApp(),
            token_resolver=lambda: ("token", "env"),
        )
        runner._stop_event.set()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            runner._run_application_loop(_FakeApp(), token_source="env", attempt=1)
        self.assertTrue(awaited["start_polling"])
        warning_messages = [str(item.message) for item in caught]
        self.assertFalse(any("never awaited" in msg for msg in warning_messages))


if __name__ == "__main__":
    unittest.main()

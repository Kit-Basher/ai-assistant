from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from agent.secret_store import SecretStore
from telegram_adapter.bot import _resolve_telegram_bot_token, build_app, main, run_polling_with_backoff


class TestTelegramTokenLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["TELEGRAM_ENABLED"] = "1"

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_secret_store_token_takes_precedence(self) -> None:
        store = SecretStore(path=os.environ["AGENT_SECRET_STORE_PATH"])
        store.set_secret("telegram:bot_token", "secret-token")
        os.environ["TELEGRAM_BOT_TOKEN"] = "env-token"

        token = _resolve_telegram_bot_token()
        self.assertEqual("secret-token", token)

    def test_env_token_used_when_secret_missing(self) -> None:
        os.environ["TELEGRAM_BOT_TOKEN"] = "env-token"
        token = _resolve_telegram_bot_token()
        self.assertEqual("env-token", token)

    def test_build_app_exits_with_clear_error_when_token_missing(self) -> None:
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                build_app()
        self.assertEqual(1, int(raised.exception.code))
        self.assertIn("Missing Telegram bot token", stderr.getvalue())

    def test_main_exits_cleanly_when_poll_lock_is_held(self) -> None:
        token = "1234567:abcdefghijklmnopqrstuvwxyz_123456"
        with patch("telegram_adapter.bot.resolve_telegram_bot_token_with_source", return_value=(token, "secret_store")):
            with patch("telegram_adapter.bot.acquire_telegram_poll_lock", return_value=None):
                with patch("telegram_adapter.bot.run_polling_with_backoff") as polling_mock:
                    with self.assertLogs("telegram_adapter.bot", level="WARNING") as logs:
                        main()
        polling_mock.assert_not_called()
        joined = "\n".join(logs.output)
        self.assertIn("already active", joined)

    def test_main_exits_cleanly_when_telegram_disabled(self) -> None:
        os.environ["TELEGRAM_ENABLED"] = "0"
        with patch("telegram_adapter.bot.run_polling_with_backoff") as polling_mock:
            with self.assertLogs("telegram_adapter.bot", level="INFO") as logs:
                main()
        polling_mock.assert_not_called()
        joined = "\n".join(logs.output)
        self.assertIn("telegram.disabled", joined)

    def test_run_polling_with_backoff_handles_conflict_and_retries(self) -> None:
        token = "1234567:abcdefghijklmnopqrstuvwxyz_123456"
        attempts = {"count": 0}
        sleep_calls: list[float] = []

        class Conflict(Exception):
            pass

        class _FakeApp:
            def __init__(self, should_conflict: bool) -> None:
                self._should_conflict = should_conflict

            def run_polling(self, **_kwargs: object) -> None:
                if self._should_conflict:
                    raise Conflict("terminated by other getUpdates request")
                return None

        def _app_factory(**_kwargs: object) -> _FakeApp:
            attempts["count"] += 1
            return _FakeApp(should_conflict=attempts["count"] == 1)

        with self.assertLogs("telegram_adapter.bot", level="ERROR") as logs:
            code = run_polling_with_backoff(
                token=token,
                token_source="secret_store",
                app_factory=_app_factory,
                sleep_fn=lambda seconds: sleep_calls.append(float(seconds)),
            )

        self.assertEqual(0, code)
        self.assertEqual(2, attempts["count"])
        self.assertEqual([2.0], sleep_calls)
        joined = "\n".join(logs.output)
        self.assertIn("getUpdates conflict", joined)


if __name__ == "__main__":
    unittest.main()

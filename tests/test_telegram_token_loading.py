from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr

from agent.secret_store import SecretStore
from telegram_adapter.bot import _resolve_telegram_bot_token, build_app


class TestTelegramTokenLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")

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


if __name__ == "__main__":
    unittest.main()

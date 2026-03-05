from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from agent.secret_store import SecretStore
from agent.secrets import main


class TestAgentSecretsCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_set_telegram_token_saves_without_echoing_secret(self) -> None:
        token = "1234567:abcdefghijklmnopqrstuvwxyz_123456"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(["set", "telegram:bot_token", "--value", token])
        self.assertEqual(0, exit_code)
        self.assertEqual("saved", stdout.getvalue().strip())
        self.assertNotIn(token, stdout.getvalue())
        self.assertEqual("", stderr.getvalue().strip())
        store = SecretStore(path=os.environ["AGENT_SECRET_STORE_PATH"])
        self.assertEqual(token, store.get_secret("telegram:bot_token"))

    def test_get_redacted_masks_secret_value(self) -> None:
        token = "1234567:abcdefghijklmnopqrstuvwxyz_123456"
        store = SecretStore(path=os.environ["AGENT_SECRET_STORE_PATH"])
        store.set_secret("telegram:bot_token", token)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["get", "telegram:bot_token", "--redacted"])
        self.assertEqual(0, exit_code)
        value = stdout.getvalue().strip()
        self.assertTrue(value.startswith("1234..."))
        self.assertNotIn(token, value)

    def test_set_telegram_token_rejects_invalid_format(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(["set", "telegram:bot_token", "--value", "invalid"])
        self.assertEqual(2, exit_code)
        self.assertEqual("", stdout.getvalue().strip())
        self.assertIn("format looks invalid", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

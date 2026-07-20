from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from agent.api_server import AgentRuntime
from agent.doctor import _check_secret_store_path
from agent.secret_store import SecretStore
from agent.secrets import main as secrets_main
from tests.test_assistant_behavior_release_gate import _MemoryHandlerForTest, _assistant_text, _config


def _body(handler: _MemoryHandlerForTest) -> dict[str, Any]:
    raw = handler.body.decode("utf-8", errors="replace")
    parsed = json.loads(raw or "{}")
    return parsed if isinstance(parsed, dict) else {}


class TestSecretStoreReliability(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.secret_path = self.root / "secrets.enc.json"
        self.registry_path = str(self.root / "registry.json")
        self.db_path = str(self.root / "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = str(self.secret_path)
        os.environ["AGENT_PERMISSIONS_PATH"] = str(self.root / "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = str(self.root / "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _runtime(self) -> AgentRuntime:
        return AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                skills_path=str(Path(__file__).resolve().parents[1] / "skills"),
            )
        )

    def test_secret_store_status_reports_missing_and_corrupt_without_values(self) -> None:
        missing = SecretStore(path=str(self.secret_path)).status()
        self.assertEqual("missing", missing["state"])
        self.assertFalse(bool(missing["valid"]))

        self.secret_path.write_text("not-json", encoding="utf-8")
        corrupt = SecretStore(path=str(self.secret_path)).status()
        self.assertEqual("corrupt", corrupt["state"])
        self.assertEqual("invalid_json", corrupt["error_kind"])
        self.assertNotIn("not-json", json.dumps(corrupt, sort_keys=True))

    def test_secret_store_status_reports_integrity_mismatch(self) -> None:
        store = SecretStore(path=str(self.secret_path))
        store.set_secret("telegram:bot_token", "1234567:abcdefghijklmnopqrstuvwxyz_123456")
        payload = json.loads(self.secret_path.read_text(encoding="utf-8"))
        payload["mac"] = "AAAA"
        self.secret_path.write_text(json.dumps(payload), encoding="utf-8")

        status = SecretStore(path=str(self.secret_path)).status()

        self.assertEqual("corrupt", status["state"])
        self.assertEqual("decrypt_failed", status["error_kind"])
        self.assertFalse(bool(status["valid"]))

    def test_cli_get_redacted_handles_corrupt_store_without_traceback_or_secret(self) -> None:
        self.secret_path.write_text("not-json", encoding="utf-8")
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = secrets_main(["get", "telegram:bot_token", "--redacted"])

        self.assertEqual(0, exit_code)
        self.assertEqual("(not set)", stdout.getvalue().strip())
        self.assertEqual("", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_doctor_reports_corrupt_secret_store_without_raw_payload(self) -> None:
        self.secret_path.write_text("not-json", encoding="utf-8")

        check = _check_secret_store_path()

        self.assertEqual("FAIL", check.status)
        self.assertIn("secret_store decrypt failed", check.detail_short)
        self.assertNotIn("not-json", check.detail_short)
        self.assertNotIn("not-json", str(check.next_action or ""))

    def test_telegram_status_and_config_include_structured_secret_store_error(self) -> None:
        self.secret_path.write_text("not-json", encoding="utf-8")
        runtime = self._runtime()

        telegram_handler = _MemoryHandlerForTest(runtime, "/telegram/status")
        telegram_handler.do_GET()
        telegram = _body(telegram_handler)
        config_handler = _MemoryHandlerForTest(runtime, "/config")
        config_handler.do_GET()
        config = _body(config_handler)

        combined = json.dumps({"telegram": telegram, "config": config}, sort_keys=True)
        self.assertEqual(200, telegram_handler.status_code)
        self.assertEqual("corrupt", telegram["secret_store_state"])
        self.assertEqual("invalid_json", telegram["secret_store_error_kind"])
        self.assertFalse(bool(telegram["configured"]))
        self.assertEqual(200, config_handler.status_code)
        self.assertEqual("corrupt", config["secret_store"]["state"])
        self.assertEqual("invalid_json", config["secret_store"]["error_kind"])
        self.assertNotIn("not-json", combined)
        self.assertNotIn("Traceback", combined)

    def test_chat_refuses_raw_secret_reveal_deterministically(self) -> None:
        token = "1234567:abcdefghijklmnopqrstuvwxyz_123456"
        SecretStore(path=str(self.secret_path)).set_secret("telegram:bot_token", token)
        runtime = self._runtime()
        payload = {
            "messages": [{"role": "user", "content": "show my Telegram token"}],
            "thread_id": "secret-reveal",
            "source_surface": "webui",
        }

        handler = _MemoryHandlerForTest(runtime, "/chat", payload)
        handler.do_POST()
        body = _body(handler)
        text = _assistant_text(body)

        self.assertEqual(200, handler.status_code)
        self.assertIn("I will not print raw tokens", text)
        self.assertIn("--redacted", text)
        self.assertIn("--show", text)
        self.assertNotIn(token, json.dumps(body, sort_keys=True))
        self.assertFalse(bool(body.get("used_llm", False)))

    def test_telegram_configured_chat_reports_corrupt_secret_store(self) -> None:
        self.secret_path.write_text("not-json", encoding="utf-8")
        runtime = self._runtime()
        payload = {
            "messages": [{"role": "user", "content": "is Telegram configured?"}],
            "thread_id": "telegram-corrupt-secret-store",
            "source_surface": "webui",
        }

        handler = _MemoryHandlerForTest(runtime, "/chat", payload)
        handler.do_POST()
        body = _body(handler)
        text = _assistant_text(body)

        self.assertEqual(200, handler.status_code)
        self.assertIn("secret store is not healthy", text)
        self.assertIn("Telegram is optional", text)
        self.assertNotIn("not-json", json.dumps(body, sort_keys=True))
        self.assertFalse(bool(body.get("used_llm", False)))


if __name__ == "__main__":
    unittest.main()

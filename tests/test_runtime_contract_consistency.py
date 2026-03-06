from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

from agent import cli
from agent.runtime_contract import normalize_user_facing_status
from telegram_adapter.bot import _runtime_status_text


class _FakeRuntime:
    def __init__(self, status_payload: dict[str, object]) -> None:
        self._status_payload = dict(status_payload)
        self.version = "0.2.0"
        self.git_commit = "abc1234"
        self.started_at = datetime.now(timezone.utc)

    def llm_status(self) -> dict[str, object]:
        return dict(self._status_payload)


class TestRuntimeContractConsistency(unittest.TestCase):
    def test_ready_state_cli_and_telegram_share_summary_semantics(self) -> None:
        normalized = normalize_user_facing_status(
            ready=True,
            bootstrap_required=False,
            failure_code=None,
            provider="ollama",
            model="qwen2.5:3b-instruct",
            local_providers={"ollama"},
        )
        ready_payload = {
            "ok": True,
            "ready": True,
            "phase": "ready",
            "runtime_mode": normalized["runtime_mode"],
            "runtime_status": normalized,
            "telegram": {"state": "running"},
            "message": str(normalized["summary"]),
            "llm": {"provider": "ollama", "model": "qwen2.5:3b-instruct"},
        }
        output = io.StringIO()
        with patch("agent.cli._http_json", return_value=(True, ready_payload)), redirect_stdout(output):
            code = cli.main(["status"])
        self.assertEqual(0, code)
        cli_text = output.getvalue()
        self.assertIn(str(normalized["summary"]), cli_text)
        self.assertIn("runtime_mode: READY", cli_text)

        status_payload = {
            "default_provider": "ollama",
            "resolved_default_model": "qwen2.5:3b-instruct",
            "default_model": "qwen2.5:3b-instruct",
            "active_provider_health": {"status": "ok"},
            "active_model_health": {"status": "ok"},
        }
        telegram_text = _runtime_status_text({"runtime": _FakeRuntime(status_payload)})
        self.assertIn("✅ Agent is running", telegram_text)
        self.assertIn("Agent is ready. Using ollama / qwen2.5:3b-instruct.", telegram_text)

    def test_bootstrap_state_semantics_across_surfaces(self) -> None:
        normalized = normalize_user_facing_status(
            ready=False,
            bootstrap_required=True,
            failure_code="no_chat_model",
            provider=None,
            model=None,
            local_providers={"ollama"},
        )
        self.assertEqual("BOOTSTRAP_REQUIRED", normalized["runtime_mode"])
        self.assertIn("Setup needed.", str(normalized["summary"]))


if __name__ == "__main__":
    unittest.main()


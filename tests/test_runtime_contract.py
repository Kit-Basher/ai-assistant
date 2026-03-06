from __future__ import annotations

import unittest

from agent.runtime_contract import (
    RUNTIME_MODE_BOOTSTRAP_REQUIRED,
    RUNTIME_MODE_DEGRADED,
    RUNTIME_MODE_FAILED,
    RUNTIME_MODE_READY,
    get_effective_llm_identity,
    get_effective_next_action,
    get_runtime_mode,
    normalize_user_facing_status,
)


class TestRuntimeContract(unittest.TestCase):
    def test_get_runtime_mode_priority(self) -> None:
        self.assertEqual(
            RUNTIME_MODE_FAILED,
            get_runtime_mode(ready=False, bootstrap_required=True, failure_code="config_load_failed"),
        )
        self.assertEqual(
            RUNTIME_MODE_BOOTSTRAP_REQUIRED,
            get_runtime_mode(ready=False, bootstrap_required=True, failure_code=None),
        )
        self.assertEqual(
            RUNTIME_MODE_READY,
            get_runtime_mode(ready=True, bootstrap_required=False, failure_code=None, phase="ready"),
        )
        self.assertEqual(
            RUNTIME_MODE_DEGRADED,
            get_runtime_mode(ready=False, bootstrap_required=False, failure_code=None, phase="warming"),
        )

    def test_get_effective_llm_identity(self) -> None:
        known = get_effective_llm_identity(
            provider="ollama",
            model="qwen2.5:3b-instruct",
            local_providers={"ollama"},
        )
        self.assertTrue(bool(known.get("known")))
        self.assertEqual("local", known.get("local_remote"))
        unknown = get_effective_llm_identity(
            provider=None,
            model=None,
            local_providers={"ollama"},
        )
        self.assertFalse(bool(unknown.get("known")))
        self.assertEqual("unknown", unknown.get("local_remote"))

    def test_get_effective_next_action(self) -> None:
        self.assertIsNone(get_effective_next_action(runtime_mode=RUNTIME_MODE_READY, failure_code=None))
        self.assertEqual(
            "Run: python -m agent.secrets set telegram:bot_token",
            get_effective_next_action(runtime_mode=RUNTIME_MODE_DEGRADED, failure_code="telegram_token_missing"),
        )
        self.assertEqual(
            "Run: python -m agent doctor",
            get_effective_next_action(runtime_mode=RUNTIME_MODE_DEGRADED, failure_code="llm_unavailable"),
        )

    def test_normalize_user_facing_status(self) -> None:
        ready = normalize_user_facing_status(
            ready=True,
            bootstrap_required=False,
            failure_code=None,
            provider="ollama",
            model="qwen2.5:3b-instruct",
            local_providers={"ollama"},
        )
        self.assertEqual(RUNTIME_MODE_READY, ready.get("runtime_mode"))
        self.assertIn("Agent is ready.", str(ready.get("summary")))

        bootstrap = normalize_user_facing_status(
            ready=False,
            bootstrap_required=True,
            failure_code="no_chat_model",
            provider=None,
            model=None,
            local_providers={"ollama"},
        )
        self.assertEqual(RUNTIME_MODE_BOOTSTRAP_REQUIRED, bootstrap.get("runtime_mode"))
        self.assertIn("Setup needed.", str(bootstrap.get("summary")))
        self.assertIn("Next:", str(bootstrap.get("summary")))

        failed = normalize_user_facing_status(
            ready=False,
            bootstrap_required=False,
            failure_code="config_load_failed",
            provider=None,
            model=None,
            local_providers={"ollama"},
        )
        self.assertEqual(RUNTIME_MODE_FAILED, failed.get("runtime_mode"))
        self.assertIn("Agent failed.", str(failed.get("summary")))


if __name__ == "__main__":
    unittest.main()


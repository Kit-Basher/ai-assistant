import os
import unittest
from unittest.mock import patch

from agent.config import load_config


class TestConfig(unittest.TestCase):
    def test_telegram_enabled_default_false(self) -> None:
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "token", "LLM_PROVIDER": "none"},
            clear=False,
        ):
            os.environ.pop("TELEGRAM_ENABLED", None)
            config = load_config()
        self.assertFalse(config.telegram_enabled)

    def test_telegram_enabled_env_true(self) -> None:
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "token", "LLM_PROVIDER": "none", "TELEGRAM_ENABLED": "1"},
            clear=False,
        ):
            config = load_config()
        self.assertTrue(config.telegram_enabled)

    def test_require_telegram_token_only_when_enabled(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "none", "TELEGRAM_ENABLED": "0"},
            clear=False,
        ):
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            config = load_config()
        self.assertFalse(config.telegram_enabled)
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "none", "TELEGRAM_ENABLED": "1"},
            clear=False,
        ):
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            with self.assertRaises(RuntimeError):
                load_config()

    def test_enable_writes_default_false(self) -> None:
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "token", "LLM_PROVIDER": "none"},
            clear=False,
        ):
            config = load_config()
        self.assertFalse(config.enable_writes)

    def test_enable_writes_env_true(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
                "ENABLE_WRITES": "true",
            },
            clear=False,
        ):
            config = load_config()
        self.assertTrue(config.enable_writes)

    def test_llm_selector_broker_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
                "LLM_SELECTOR": "broker",
                "LLM_BROKER_POLICY_PATH": "/tmp/policy.yaml",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "no longer supported"):
                load_config()

    def test_memory_v2_default_false(self) -> None:
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "token", "LLM_PROVIDER": "none"},
            clear=False,
        ):
            os.environ.pop("AGENT_MEMORY_V2_ENABLED", None)
            config = load_config()
        self.assertFalse(config.memory_v2_enabled)

    def test_memory_v2_env_true(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
                "AGENT_MEMORY_V2_ENABLED": "true",
            },
            clear=False,
        ):
            config = load_config()
        self.assertTrue(config.memory_v2_enabled)

    def test_llm_notifications_allow_test_env_true(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
                "LLM_NOTIFICATIONS_ALLOW_TEST": "true",
            },
            clear=False,
        ):
            config = load_config()
        self.assertTrue(config.llm_notifications_allow_test)

    def test_llm_notifications_allow_test_env_false(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
                "LLM_NOTIFICATIONS_ALLOW_TEST": "false",
            },
            clear=False,
        ):
            config = load_config()
        self.assertFalse(config.llm_notifications_allow_test)

    def test_llm_notifications_allow_test_unset_is_auto(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
            },
            clear=False,
        ):
            os.environ.pop("LLM_NOTIFICATIONS_ALLOW_TEST", None)
            config = load_config()
        self.assertIsNone(config.llm_notifications_allow_test)

    def test_llm_notifications_allow_send_env_true(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
                "LLM_NOTIFICATIONS_ALLOW_SEND": "true",
            },
            clear=False,
        ):
            config = load_config()
        self.assertTrue(config.llm_notifications_allow_send)

    def test_llm_notifications_allow_send_env_false(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
                "LLM_NOTIFICATIONS_ALLOW_SEND": "false",
            },
            clear=False,
        ):
            config = load_config()
        self.assertFalse(config.llm_notifications_allow_send)

    def test_llm_notifications_retention_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
            },
            clear=False,
        ):
            os.environ.pop("LLM_NOTIFICATIONS_MAX_ITEMS", None)
            os.environ.pop("LLM_NOTIFICATIONS_MAX_AGE_DAYS", None)
            os.environ.pop("LLM_NOTIFICATIONS_COMPACT", None)
            config = load_config()
        self.assertEqual(200, config.llm_notifications_max_items)
        self.assertEqual(30, config.llm_notifications_max_age_days)
        self.assertTrue(config.llm_notifications_compact)

    def test_llm_notifications_retention_env_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
                "LLM_NOTIFICATIONS_MAX_ITEMS": "17",
                "LLM_NOTIFICATIONS_MAX_AGE_DAYS": "14",
                "LLM_NOTIFICATIONS_COMPACT": "false",
            },
            clear=False,
        ):
            config = load_config()
        self.assertEqual(17, config.llm_notifications_max_items)
        self.assertEqual(14, config.llm_notifications_max_age_days)
        self.assertFalse(config.llm_notifications_compact)

    def test_llm_self_heal_allow_apply_env_true(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
                "LLM_SELF_HEAL_ALLOW_APPLY": "true",
            },
            clear=False,
        ):
            config = load_config()
        self.assertTrue(config.llm_self_heal_allow_apply)

    def test_llm_self_heal_allow_apply_unset_is_auto(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
            },
            clear=False,
        ):
            os.environ.pop("LLM_SELF_HEAL_ALLOW_APPLY", None)
            config = load_config()
        self.assertIsNone(config.llm_self_heal_allow_apply)

    def test_llm_self_heal_interval_env_override(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
                "LLM_SELF_HEAL_INTERVAL_S": "4321",
            },
            clear=False,
        ):
            config = load_config()
        self.assertEqual(4321, config.llm_self_heal_interval_seconds)

    def test_llm_registry_rollback_allow_unset_is_auto(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
            },
            clear=False,
        ):
            os.environ.pop("LLM_REGISTRY_ROLLBACK_ALLOW", None)
            config = load_config()
        self.assertIsNone(config.llm_registry_rollback_allow)

    def test_llm_autopilot_safe_mode_defaults_true(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
            },
            clear=False,
        ):
            os.environ.pop("LLM_AUTOPILOT_SAFE_MODE", None)
            config = load_config()
        self.assertTrue(config.llm_autopilot_safe_mode)

    def test_llm_autopilot_safe_mode_env_false(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
                "LLM_AUTOPILOT_SAFE_MODE": "false",
            },
            clear=False,
        ):
            config = load_config()
        self.assertFalse(config.llm_autopilot_safe_mode)

    def test_llm_autopilot_bootstrap_allow_apply_unset_is_auto(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
            },
            clear=False,
        ):
            os.environ.pop("LLM_AUTOPILOT_BOOTSTRAP_ALLOW_APPLY", None)
            config = load_config()
        self.assertIsNone(config.llm_autopilot_bootstrap_allow_apply)

    def test_llm_autopilot_churn_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "LLM_PROVIDER": "none",
            },
            clear=False,
        ):
            os.environ.pop("LLM_AUTOPILOT_CHURN_WINDOW_SECONDS", None)
            os.environ.pop("LLM_AUTOPILOT_CHURN_MIN_APPLIES", None)
            os.environ.pop("LLM_AUTOPILOT_CHURN_RECENT_LIMIT", None)
            config = load_config()
        self.assertEqual(1800, config.llm_autopilot_churn_window_seconds)
        self.assertEqual(4, config.llm_autopilot_churn_min_applies)
        self.assertEqual(80, config.llm_autopilot_churn_recent_limit)


if __name__ == "__main__":
    unittest.main()

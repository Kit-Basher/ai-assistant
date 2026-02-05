import os
import unittest
from unittest.mock import patch

from agent.config import load_config


class TestConfig(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

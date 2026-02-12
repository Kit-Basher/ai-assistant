import os
import unittest
from pathlib import Path
from unittest import mock

from agent.config import load_observe_config


class TestObserveConfig(unittest.TestCase):
    def test_uses_env_db_path_when_set(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_DB_PATH": "/tmp/custom-agent.db"}, clear=False):
            cfg = load_observe_config()
        self.assertEqual("/tmp/custom-agent.db", cfg.db_path)

    def test_falls_back_to_repo_default_when_env_missing(self) -> None:
        env = dict(os.environ)
        env.pop("AGENT_DB_PATH", None)
        env.pop("TELEGRAM_BOT_TOKEN", None)
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = load_observe_config()
        expected = str(Path(__file__).resolve().parents[1] / "memory" / "agent.db")
        self.assertEqual(expected, cfg.db_path)

    def test_does_not_require_telegram_bot_token(self) -> None:
        env = dict(os.environ)
        env.pop("AGENT_DB_PATH", None)
        env.pop("TELEGRAM_BOT_TOKEN", None)
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = load_observe_config()
        self.assertTrue(cfg.db_path.endswith("memory/agent.db"))


if __name__ == "__main__":
    unittest.main()

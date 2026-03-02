from __future__ import annotations

import os
import unittest


class TestTelegramCommandRegistration(unittest.TestCase):
    def test_done_and_task_add_are_registered(self) -> None:
        bot_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "telegram_adapter", "bot.py")
        )
        with open(bot_path, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn('CommandHandler("task_add", _handle_task_add)', source)
        self.assertIn('CommandHandler("done", _handle_done)', source)
        self.assertIn('CommandHandler("breif", _handle_brief_alias)', source)
        self.assertIn('CommandHandler("help", _handle_help)', source)
        self.assertIn('CommandHandler("model", _handle_model)', source)
        self.assertIn('CommandHandler("scout", _handle_scout)', source)
        self.assertIn('CommandHandler("scout_dismiss", _handle_scout_dismiss)', source)
        self.assertIn('CommandHandler("scout_installed", _handle_scout_installed)', source)
        self.assertIn('CommandHandler("permissions", _handle_permissions)', source)


if __name__ == "__main__":
    unittest.main()

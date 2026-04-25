from __future__ import annotations

import unittest

from agent.public_chat import build_trivial_social_turn_message, classify_trivial_social_turn


class TestPublicChat(unittest.TestCase):
    def test_social_turn_classifier_handles_real_world_greeting_variants(self) -> None:
        cases = {
            "hello there": "greeting",
            "say hi": "greeting",
            "herllo": "greeting",
            "are you really there": "presence_check",
            "hello are you there": "presence_check",
            "hello are you working": "presence_check",
            "are you working?": "presence_check",
            "Actually, keep the answer short.": "style_short",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(expected, classify_trivial_social_turn(text))

    def test_social_turn_message_stays_user_facing_for_presence_check(self) -> None:
        message = build_trivial_social_turn_message("are you really there")
        self.assertIsNotNone(message)
        self.assertIn("I’m here", str(message))
        self.assertIn("ready to help", str(message))

    def test_social_turn_message_acknowledges_brief_style_request(self) -> None:
        message = build_trivial_social_turn_message("keep it short")
        self.assertEqual("Okay. I’ll keep it short.", message)


if __name__ == "__main__":
    unittest.main()

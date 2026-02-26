from __future__ import annotations

import unittest

from agent.api_server import APIServerHandler


class TestLowConfidenceExtraction(unittest.TestCase):
    def test_extract_user_text_from_messages_string(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        self.assertEqual("hello", APIServerHandler._extract_user_text_for_low_confidence(payload))

    def test_extract_user_text_from_messages_text_parts(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "world"},
                    ],
                }
            ]
        }
        self.assertEqual("hello world", APIServerHandler._extract_user_text_for_low_confidence(payload))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from agent.intent.low_confidence import detect_low_confidence


class TestLowConfidenceDetector(unittest.TestCase):
    def test_table_driven_detection(self) -> None:
        cases = [
            ("", True, "empty"),
            ("   ", True, "empty"),
            ("???", True, "no_semantic_tokens"),
            ("🔥🔥🔥", True, "no_semantic_tokens"),
            ("ok", True, "too_short"),
            ("FFFFFFFFFFFFF", True, "repetition_spam"),
            ("help", True, "underspecified"),
            ("please summarize this text", False, "confident"),
        ]
        for text, expected_low, expected_reason in cases:
            with self.subTest(text=text):
                result = detect_low_confidence(text)
                self.assertEqual(expected_low, result.is_low_confidence)
                self.assertEqual(expected_reason, result.reason)
                if expected_low:
                    self.assertTrue(result.next_question.strip())
                else:
                    self.assertEqual("", result.next_question)
                self.assertEqual(result.debug.get("norm"), detect_low_confidence(text).debug.get("norm"))
                self.assertIn("length", result.debug)
                self.assertIn("word_count", result.debug)
                self.assertIn("unique_word_count", result.debug)
                self.assertIn("has_letter_or_digit", result.debug)
                self.assertIn("is_punct_only", result.debug)
                self.assertIn("has_big_repeat", result.debug)


if __name__ == "__main__":
    unittest.main()

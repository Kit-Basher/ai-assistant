from __future__ import annotations

import unittest

from agent.intent.thread_integrity import detect_thread_drift, normalize_text


class TestThreadIntegrityGuard(unittest.TestCase):
    def test_explicit_switch_triggers(self) -> None:
        text = "let's switch topics and discuss a different thing"
        norm = normalize_text(text)
        result = detect_thread_drift(
            user_text_raw=text,
            user_text_norm=norm,
            intent="chat",
            last_user_text_norm="help me debug this traceback",
            last_assistant_text_norm="paste the traceback",
        )
        self.assertTrue(result.is_thread_drift)
        self.assertEqual("explicit_switch", result.reason)
        self.assertTrue(result.message.strip())
        self.assertEqual(result.message, result.next_question)

    def test_multi_intent_triggers_when_long(self) -> None:
        text = "help me with this traceback and also write a dinner recipe for chicken and rice tonight"
        norm = normalize_text(text)
        result = detect_thread_drift(
            user_text_raw=text,
            user_text_norm=norm,
            intent="chat",
            last_user_text_norm="help me debug a python traceback",
            last_assistant_text_norm="sure, paste the traceback",
        )
        self.assertTrue(result.is_thread_drift)
        self.assertEqual("multi_intent", result.reason)
        self.assertEqual(result.message, result.next_question)

    def test_topic_shift_triggers_with_low_similarity(self) -> None:
        text = "write a chicken rice dinner recipe with prep steps and grocery list"
        norm = normalize_text(text)
        result = detect_thread_drift(
            user_text_raw=text,
            user_text_norm=norm,
            intent="chat",
            last_user_text_norm="help me debug python traceback import error in flask app",
            last_assistant_text_norm="paste the traceback and environment details",
        )
        self.assertTrue(result.is_thread_drift)
        self.assertEqual("topic_shift", result.reason)

    def test_no_drift_when_similarity_high(self) -> None:
        text = "debug python traceback in flask app with import error"
        norm = normalize_text(text)
        result = detect_thread_drift(
            user_text_raw=text,
            user_text_norm=norm,
            intent="chat",
            last_user_text_norm="help debug python traceback in flask app",
            last_assistant_text_norm="share full traceback and stack",
        )
        self.assertFalse(result.is_thread_drift)
        self.assertEqual("none", result.reason)


if __name__ == "__main__":
    unittest.main()

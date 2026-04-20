from __future__ import annotations

import json
from pathlib import Path
import unittest

from agent.llm.task_classifier import classify_task_request
from agent.public_chat import normalize_public_assistant_text


class TestPromptRegressionPack(unittest.TestCase):
    def test_p0_prompt_pack_task_classification(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "p0_prompt_regression_pack.json"
        rows = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertTrue(isinstance(rows, list) and rows)
        for row in rows:
            prompt = str((row or {}).get("prompt") or "")
            expected = str((row or {}).get("expected_task_type") or "chat")
            result = classify_task_request(prompt)
            self.assertEqual(expected, result.get("task_type"), msg=f"prompt={prompt!r}")
            self.assertGreaterEqual(float(result.get("confidence") or 0.0), 0.45)

    def test_public_text_fallback_is_not_terse_done(self) -> None:
        text = normalize_public_assistant_text("", fallback="I couldn't complete that yet. Please try rephrasing or ask me to run a quick runtime check.")
        self.assertNotEqual("Done.", text)
        self.assertIn("try rephrasing", text.lower())


if __name__ == "__main__":
    unittest.main()

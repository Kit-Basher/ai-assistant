from __future__ import annotations

import unittest

from agent.identity import get_public_identity


class TestIdentity(unittest.TestCase):
    def test_identity_known_local(self) -> None:
        payload = get_public_identity(
            provider="ollama",
            model="qwen2.5:3b-instruct",
            local_providers={"ollama"},
        )
        self.assertEqual("ollama", payload["provider"])
        self.assertEqual("qwen2.5:3b-instruct", payload["model"])
        self.assertEqual("local", payload["locality"])
        self.assertEqual("your Personal Agent", payload["assistant_label"])
        self.assertIn("Current provider/model:", str(payload["summary"]))
        self.assertIn("I’m your Personal Agent.", str(payload["summary"]))

    def test_identity_unknown_message(self) -> None:
        payload = get_public_identity(provider=None, model=None, local_providers={"ollama"})
        self.assertEqual("unknown", payload["locality"])
        text = str(payload["summary"])
        self.assertIn("The active model is currently unknown.", text)
        self.assertIn("Current provider/model: unknown / unknown.", text)
        self.assertNotIn("created by", text.lower())

    def test_identity_uses_optional_names_when_configured(self) -> None:
        payload = get_public_identity(
            provider="ollama",
            model="qwen2.5:3b-instruct",
            local_providers={"ollama"},
            assistant_name="Nova",
            user_name="Casey",
        )
        self.assertEqual("Nova, your Personal Agent", payload["assistant_label"])
        self.assertEqual("Casey", payload["user_label"])
        self.assertIn("I’m Nova, your Personal Agent.", str(payload["summary"]))


if __name__ == "__main__":
    unittest.main()

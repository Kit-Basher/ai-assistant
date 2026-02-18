from __future__ import annotations

import unittest

from agent.llm.provider_validation import validate_provider_call_format


class TestLLMProviderValidation(unittest.TestCase):
    def test_bad_base_url_maps_to_error_kind(self) -> None:
        result = validate_provider_call_format(
            "openrouter",
            {
                "base_url": "not-a-url",
                "chat_path": "/chat/completions",
                "local": False,
            },
            headers={},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("bad_base_url", result["error_kind"])

    def test_misconfigured_path_maps_to_error_kind(self) -> None:
        result = validate_provider_call_format(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "chat_path": "/v1/embeddings",
                "local": False,
            },
            headers={},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("misconfigured_path", result["error_kind"])

    def test_missing_auth_maps_to_error_kind(self) -> None:
        result = validate_provider_call_format(
            "openai",
            {
                "base_url": "https://api.openai.com",
                "chat_path": "/v1/chat/completions",
                "local": False,
                "api_key_source": {"type": "env", "name": "OPENAI_API_KEY"},
            },
            headers={},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("missing_auth", result["error_kind"])

    def test_valid_openai_compat_payload(self) -> None:
        result = validate_provider_call_format(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "chat_path": "/chat/completions",
                "local": False,
                "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()

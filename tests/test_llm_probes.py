from __future__ import annotations

import unittest

from agent.llm.probes import probe_model, probe_provider


class TestLLMProbes(unittest.TestCase):
    def test_openai_missing_auth_is_explicit(self) -> None:
        cfg = {
            "id": "openai",
            "provider_type": "openai_compat",
            "base_url": "https://api.openai.com",
            "chat_path": "/v1/chat/completions",
            "enabled": True,
            "local": False,
            "allow_remote_fallback": True,
            "api_key_source": {"type": "env", "name": "OPENAI_API_KEY"},
            "headers": {},
            "_resolved_api_key_present": False,
        }
        provider_result = probe_provider(cfg)
        model_result = probe_model(cfg, "gpt-4o-mini", model_capabilities=["chat"])
        self.assertEqual("missing_auth", provider_result["error_kind"])
        self.assertEqual("missing_auth", model_result["error_kind"])

    def test_openrouter_missing_auth_is_explicit(self) -> None:
        cfg = {
            "id": "openrouter",
            "provider_type": "openai_compat",
            "base_url": "https://openrouter.ai/api/v1",
            "chat_path": "/v1/chat/completions",
            "enabled": True,
            "local": False,
            "allow_remote_fallback": True,
            "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
            "headers": {},
            "_resolved_api_key_present": False,
        }
        provider_result = probe_provider(cfg)
        model_result = probe_model(cfg, "openai/gpt-4o-mini", model_capabilities=["chat"])
        self.assertEqual("missing_auth", provider_result["error_kind"])
        self.assertEqual("missing_auth", model_result["error_kind"])

    def test_embedding_only_model_probe_is_not_applicable(self) -> None:
        cfg = {
            "id": "ollama",
            "provider_type": "openai_compat",
            "base_url": "http://127.0.0.1:11434",
            "chat_path": "/v1/chat/completions",
            "enabled": True,
            "local": True,
            "allow_remote_fallback": True,
            "api_key_source": None,
            "headers": {},
        }
        result = probe_model(cfg, "nomic-embed-text:latest", model_capabilities=["embedding"])
        self.assertEqual("not_applicable", result["error_kind"])


if __name__ == "__main__":
    unittest.main()

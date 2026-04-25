from __future__ import annotations

import unittest

from agent.llm.chat_preflight import prepare_chat_request


class TestLLMChatPreflight(unittest.TestCase):
    def test_interactive_chat_surfaces_pin_to_default_target(self) -> None:
        prepared = prepare_chat_request(
            payload={"source_surface": "webui"},
            messages=[{"role": "user", "content": "What colour is a bluejay?"}],
            defaults={
                "default_model": "ollama:qwen2.5:7b-instruct",
                "default_provider": "ollama",
                "allow_remote_fallback": True,
            },
            request_started_epoch=0,
            default_policy={},
            premium_policy={},
            select_chat_candidates=lambda **_: {},
            premium_override_active=lambda _epoch: False,
            premium_override_once=False,
            persist_premium_over_cap_prompt=lambda **_: "confirm",
            classify_authoritative_domain=lambda _text: set(),
            has_local_observations_block=lambda _text: False,
            collect_authoritative_observations=lambda _domains: {},
            authoritative_tool_failure_text=lambda _domains, _exc: "tool failure",
        )

        self.assertEqual("ollama", prepared.provider_override)
        self.assertEqual("ollama:qwen2.5:7b-instruct", prepared.model_override)
        self.assertEqual("default_target_pin", prepared.selection_reason)

    def test_noninteractive_surface_keeps_broker_selection_open(self) -> None:
        prepared = prepare_chat_request(
            payload={"source_surface": "api"},
            messages=[{"role": "user", "content": "What colour is a bluejay?"}],
            defaults={
                "default_model": "ollama:qwen2.5:7b-instruct",
                "default_provider": "ollama",
                "allow_remote_fallback": True,
            },
            request_started_epoch=0,
            default_policy={},
            premium_policy={},
            select_chat_candidates=lambda **_: {},
            premium_override_active=lambda _epoch: False,
            premium_override_once=False,
            persist_premium_over_cap_prompt=lambda **_: "confirm",
            classify_authoritative_domain=lambda _text: set(),
            has_local_observations_block=lambda _text: False,
            collect_authoritative_observations=lambda _domains: {},
            authoritative_tool_failure_text=lambda _domains, _exc: "tool failure",
        )

        self.assertIsNone(prepared.provider_override)
        self.assertIsNone(prepared.model_override)
        self.assertEqual("default_policy", prepared.selection_reason)


if __name__ == "__main__":
    unittest.main()

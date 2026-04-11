from __future__ import annotations

import unittest

from agent.setup_chat_flow import classify_runtime_chat_route


class TestSetupChatFlow(unittest.TestCase):
    def test_natural_runtime_and_provider_phrases_route_deterministically(self) -> None:
        cases = (
            ("can you tell if everything is working with the agent?", "runtime_status", "runtime_status"),
            ("can you read the runtime now?", "runtime_status", "runtime_status"),
            ("what is the agent status?", "runtime_status", "runtime_status"),
            ("openrouter health", "provider_status", "provider_status"),
            ("is ollama working?", "provider_status", "provider_status"),
        )
        for text, expected_route, expected_kind in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual(expected_route, decision.get("route"))
                self.assertEqual(expected_kind, decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_operational_queries_bypass_generic_meta_choice(self) -> None:
        cases = (
            ("agent doctor", "operational_status", "operational_doctor"),
            ("how much memory am I using?", "operational_status", "operational_observe"),
            ("how is my storage?", "operational_status", "operational_observe"),
            ("disk usage", "operational_status", "operational_observe"),
            ("ram usage", "operational_status", "operational_observe"),
            ("what other pc stats can you find?", "operational_status", "operational_observe"),
            ("can you tell what CPU and GPU I have?", "operational_status", "operational_observe"),
            ("can you see the GPU?", "operational_status", "operational_observe"),
            ("can you dig deeper into my system?", "operational_status", "operational_observe"),
            ("run a system check", "operational_status", "operational_observe"),
        )
        for text, expected_route, expected_kind in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual(expected_route, decision.get("route"))
                self.assertEqual(expected_kind, decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_agent_health_stays_runtime_status_while_machine_stats_stay_operational(self) -> None:
        agent_health = classify_runtime_chat_route("agent health")
        machine_stats = classify_runtime_chat_route("what are my pc specs?")

        self.assertEqual("runtime_status", agent_health.get("route"))
        self.assertEqual("runtime_status", agent_health.get("kind"))
        self.assertEqual("operational_status", machine_stats.get("route"))
        self.assertEqual("operational_observe", machine_stats.get("kind"))

    def test_broad_runtime_report_phrases_route_to_runtime_status(self) -> None:
        cases = (
            "system report",
            "system status",
            "give me a system report",
            "agent health report",
            "what is happening",
        )
        for text in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual("runtime_status", decision.get("route"))
                self.assertEqual("runtime_status", decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_configure_ollama_keeps_setup_precedence(self) -> None:
        for text in ("configure ollama", "setup ollama", "use ollama", "switch to ollama", "can you fix ollama?"):
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual("setup_flow", decision.get("route"))
                self.assertEqual("configure_ollama", decision.get("kind"))

    def test_setup_explanation_phrases_route_to_setup_flow(self) -> None:
        cases = (
            "Check setup and explain what's wrong",
            "check setup",
            "what's wrong with setup?",
        )
        for text in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual("setup_flow", decision.get("route"))
                self.assertEqual("setup_explanation", decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_direct_model_switch_phrases_route_to_model_status(self) -> None:
        cases = (
            "switch to qwen2.5:7b-instruct",
            "use deepseek-r1:7b",
            "change to qwen2.5:7b-instruct",
            "switch chat to qwen2.5:7b-instruct",
            "switch to ollama:qwen2.5:7b-instruct",
            "switch to openrouter:openai/gpt-4o-mini",
        )
        for text in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual("model_status", decision.get("route"))
                self.assertEqual("set_default_model", decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_explicit_controller_phrases_with_provider_targets_do_not_route_to_setup_flow(self) -> None:
        cases = (
            "switch temporarily to ollama:qwen2.5:7b-instruct",
            "test ollama:qwen2.5:7b-instruct without adopting it",
            "make ollama:deepseek-r1:7b the default",
        )
        for text in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertNotEqual("setup_flow", decision.get("route"))

    def test_local_model_inventory_phrases_route_to_model_status(self) -> None:
        cases = (
            "what ollama models do we have downloaded?",
            "do we have any other models downloaded?",
            "do we have any other local models?",
            "what local models do we have?",
            "what models are installed?",
        )
        for text in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual("model_status", decision.get("route"))
                self.assertEqual("local_model_inventory", decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_model_ready_now_phrases_route_to_model_status(self) -> None:
        for text in ("what models are ready now?", "which models are usable right now?"):
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual("model_status", decision.get("route"))
                self.assertEqual("model_ready_now", decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_model_lifecycle_phrases_route_to_model_status(self) -> None:
        cases = (
            "is ollama:qwen2.5:7b-instruct installed?",
            "what models are downloading?",
            "did ollama:qwen2.5:7b-instruct install successfully?",
            "what model installs failed?",
            "what is the status of ollama:deepseek-r1:7b?",
            "can you install ollama:qwen2.5-coder:7b?",
        )
        for text in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual("model_status", decision.get("route"))
                self.assertEqual("model_lifecycle_status", decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_model_controller_policy_questions_route_to_model_policy_status(self) -> None:
        cases = (
            "what mode am i in?",
            "what does this mode allow?",
            "why can't you switch that here?",
            "what would you need my approval for?",
        )
        for text in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual("model_policy_status", decision.get("route"))
                self.assertEqual("model_controller_policy", decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_pack_capability_prompts_route_to_pack_recommendation(self) -> None:
        cases = (
            ("Talk to me out loud", "voice_output"),
            ("Use the avatar", "avatar_visual"),
            ("Open the robot camera feed", "camera_feed"),
            ("What pack do I need for voice output?", "voice_output"),
        )
        for text, capability in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual("action_tool", decision.get("route"))
                self.assertEqual("pack_capability_recommendation", decision.get("kind"))
                self.assertEqual(capability, decision.get("capability"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_cheap_cloud_recommendation_phrases_route_to_action_tool(self) -> None:
        cases = (
            "what cheap cloud model should I use?",
            "what low-cost cloud model should I use for coding?",
            "what budget remote model should I use?",
            "what premium model should I use for research?",
            "what premium coding model should I use?",
            "best model",
        )
        for text in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual("action_tool", decision.get("route"))
                self.assertEqual("model_scout_strategy", decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_explicit_model_acquisition_requests_route_to_action_tool(self) -> None:
        cases = (
            "install ollama:qwen2.5:7b-instruct",
            "download ollama:qwen2.5:7b-instruct",
            "pull ollama:qwen2.5:7b-instruct",
            "acquire it first",
        )
        for text in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual("action_tool", decision.get("route"))
                self.assertEqual("model_acquisition_request", decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))

    def test_typo_tolerant_critical_model_phrases_route_deterministically(self) -> None:
        cases = (
            ("waht model are you using?", "model_status", "describe_current_model"),
            ("wat models are availble?", "model_status", "model_availability"),
            ("ollma status", "provider_status", "provider_status"),
        )
        for text, expected_route, expected_kind in cases:
            with self.subTest(text=text):
                decision = classify_runtime_chat_route(text)
                self.assertEqual(expected_route, decision.get("route"))
                self.assertEqual(expected_kind, decision.get("kind"))
                self.assertFalse(bool(decision.get("generic_allowed")))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestAssistantInteractionBarrage(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_module(
            REPO_ROOT / "scripts" / "assistant_interaction_barrage.py",
            "assistant_interaction_barrage_script",
        )

    def test_catalog_includes_mixed_session_scenarios(self) -> None:
        ids = [scenario.id for scenario in self.module._base_scenarios()]
        self.assertEqual(35, len(ids))
        self.assertIn("air-024-webui-rewind", ids)
        self.assertIn("air-025-telegram-correction", ids)
        self.assertIn("air-026-webui-typo-model", ids)
        self.assertIn("air-027-telegram-typo-runtime", ids)
        self.assertIn("air-028-webui-typo-memory", ids)
        self.assertIn("air-042-webui-summary-recap", ids)
        self.assertIn("air-043-telegram-next-step", ids)
        self.assertIn("air-044-webui-natural-role-summary", ids)
        self.assertIn("air-045-webui-thinking-help", ids)
        self.assertIn("air-046-webui-custom-helper-proposal", ids)
        self.assertIn("air-047-webui-assistant-agent-boundary", ids)
        self.assertIn("air-048-telegram-assistant-agent-boundary", ids)

        file_scenarios, token = self.module._file_scenarios()
        self.assertTrue(token.startswith("assistant-barrage-token-"))
        file_ids = [scenario.id for scenario in file_scenarios]
        self.assertEqual(
            [
                "air-029-webui-file-read",
                "air-030-telegram-file-read",
                "air-031-webui-mixed-session",
                "air-032-telegram-mixed-session",
            ],
            file_ids,
        )

    def test_mixed_session_checker_accepts_coherent_session_and_rejects_dead_end(self) -> None:
        module = self.module
        token = "assistant-barrage-token-test"
        good_turns = (
            module.TurnResult("help me plan my day", "Today priorities: finish the release checks and keep the plan tight.", "plan_day", True, "webui", 200),
            module.TurnResult("what is the runtime status?", "Ready. Using ollama / ollama:qwen2.5:7b-instruct.", "runtime_status", True, "webui", 200),
            module.TurnResult("what do you remember about my preferences?", "I remember a saved preference for concise replies.", "agent_memory", True, "webui", 200),
            module.TurnResult("read the file /tmp/note.txt", f"Text from /tmp/note.txt (79 bytes).\n{token}", "action_tool", True, "webui", 200),
            module.TurnResult("talk to me out loud", "Voice output isn't installed. The narrowest thing to add would be a small helper.", "assistant_capabilities", True, "webui", 200),
            module.TurnResult("go back to the day plan", "We were working on today plan. I can pick up today plan from there.", "agent_memory", True, "webui", 200),
            module.TurnResult("what are we doing?", "We were working on today plan. Last thing you asked me to do: what are we doing?.", "agent_memory", True, "webui", 200),
            module.TurnResult("what should we do next?", "Next I would continue the day plan and verify the remaining release work.", "generic_chat", True, "webui", 200),
        )
        self.assertEqual([], module._check_mixed_session_soak(good_turns, expected_file_token=token))

        bad_turns = good_turns[:-1] + (
            module.TurnResult("what should we do next?", "I can help with that.", "generic_chat", True, "webui", 200),
        )
        warnings = module._check_mixed_session_soak(bad_turns, expected_file_token=token)
        self.assertTrue(any("useful next step" in warning or "dead-end" in warning for warning in warnings))

    def test_model_truth_checker_accepts_grounded_model_status(self) -> None:
        module = self.module
        turns = (
            module.TurnResult(
                "what model are you uding",
                "Chat is currently using ollama:qwen2.5:7b-instruct on Ollama.",
                "model_status",
                True,
                "webui",
                200,
            ),
        )
        self.assertEqual([], module._check_model_truth(turns))

    def test_capability_checker_accepts_natural_role_summary(self) -> None:
        module = self.module
        turns = (
            module.TurnResult(
                "say what you do in one sentence, but keep it natural",
                "I can help inspect this system, check runtime and model status, read and update local memory, and point out when a task needs an extra pack.",
                "assistant_capabilities",
                True,
                "webui",
                200,
            ),
        )
        self.assertEqual([], module._check_capability(turns))

    def test_capability_checker_rejects_repeated_partial_help_intro(self) -> None:
        module = self.module
        turns = (
            module.TurnResult(
                "help me code",
                "I can help in text, but this would work better with a small coding helper. Coding tools isn't installed. I can help in text, but this would work better with a small coding helper.",
                "action_tool",
                True,
                "webui",
                200,
            ),
        )
        warnings = module._check_capability(turns)
        self.assertTrue(any("repeated its partial-help intro" in warning for warning in warnings))

    def test_thinking_help_checker_accepts_short_natural_reply(self) -> None:
        module = self.module
        turns = (
            module.TurnResult(
                "i need help thinking through something messy, but keep it simple",
                "Yes. Tell me the goal, the messy part, and any constraint, and I’ll help break it down simply.",
                "assistant_capabilities",
                True,
                "webui",
                200,
            ),
        )
        self.assertEqual([], module._check_thinking_help(turns))

    def test_custom_helper_checker_accepts_collaborative_proposal(self) -> None:
        module = self.module
        turns = (
            module.TurnResult(
                "make an assistant that coordinates my studio light cues with my music cues during live shows",
                "I couldn't find a ready-made helper for this. The simplest way to add it would be a small helper that coordinates studio light cues with music cues during live shows. If you want, I can sketch that with you.",
                "action_tool",
                True,
                "webui",
                200,
            ),
        )
        self.assertEqual([], module._check_custom_helper(turns))

    def test_assistant_agent_boundary_checker_rejects_runtime_fallback(self) -> None:
        module = self.module
        good_turns = (
            module.TurnResult(
                "what are you and what is the agent layer supposed to do?",
                "You interact with me, the assistant. I decide when to ask the agent layer for grounded work and bounded facts.",
                "assistant_capabilities",
                True,
                "webui",
                200,
            ),
        )
        self.assertEqual([], module._check_assistant_agent_boundary(good_turns))

        bad_turns = (
            module.TurnResult(
                "what are you and what is the agent layer supposed to do?",
                "I can't read a clean runtime status yet.",
                "runtime_status",
                True,
                "webui",
                200,
            ),
        )
        warnings = module._check_assistant_agent_boundary(bad_turns)
        self.assertTrue(any("runtime fallback" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()

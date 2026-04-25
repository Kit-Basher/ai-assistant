from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
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


class TestAssistantViabilitySmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_module(REPO_ROOT / "scripts" / "assistant_viability_smoke.py", "assistant_viability_smoke_script")

    def test_catalog_includes_canonical_scenarios_and_minimum_gate(self) -> None:
        ids = [spec.id for spec in self.module.SCENARIOS]
        self.assertEqual(11, len(ids))
        self.assertEqual(
            (
                "greeting_followup_webui",
                "runtime_status_followup_webui",
                "continuity_resume_webui",
            ),
            self.module.MINIMUM_VIABILITY_GATE,
        )
        for scenario_id in (
            "greeting_followup_webui",
            "runtime_status_followup_webui",
            "local_system_inspection_webui",
            "preview_confirm_flow_webui",
            "interruption_topic_shift_webui",
            "continuity_resume_webui",
            "memory_persistence_soak_webui",
            "long_human_like_session_webui",
            "long_human_like_session_telegram",
            "telegram_surface_behavior",
            "webui_surface_behavior",
        ):
            self.assertIn(scenario_id, ids)

    def test_evaluate_scenario_classifies_transport_truth_and_memory_failures(self) -> None:
        module = self.module
        transport_spec = next(spec for spec in module.SCENARIOS if spec.id == "runtime_status_followup_webui")
        transport_turns = (
            module.TurnResult(
                user_text="what is the runtime status?",
                assistant_text="",
                route="error",
                ok=False,
                surface="webui",
                trace_id="trace-1",
                error="HTTP 500",
            ),
        )
        transport_result = module.evaluate_scenario(transport_spec, transport_turns)
        self.assertFalse(transport_result.passed)
        self.assertEqual("transport/runtime", transport_result.failure_category)

        grounding_spec = next(spec for spec in module.SCENARIOS if spec.id == "local_system_inspection_webui")
        grounding_turns = (
            module.TurnResult(
                user_text="what do i have for ram and vram right now?",
                assistant_text="I can help with that.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-2",
                payload={"ok": True},
            ),
        )
        grounding_result = module.evaluate_scenario(grounding_spec, grounding_turns)
        self.assertFalse(grounding_result.passed)
        self.assertEqual("grounding/truth", grounding_result.failure_category)

        memory_spec = next(spec for spec in module.SCENARIOS if spec.id == "continuity_resume_webui")
        memory_turns = (
            module.TurnResult(
                user_text="we are testing the assistant viability gate",
                assistant_text="Sure, I can help.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-3",
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="what are we doing?",
                assistant_text="I can help with that.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-4",
                payload={"ok": True},
            ),
        )
        memory_result = module.evaluate_scenario(memory_spec, memory_turns)
        self.assertFalse(memory_result.passed)
        self.assertEqual("memory/continuity", memory_result.failure_category)

        soak_spec = next(spec for spec in module.SCENARIOS if spec.id == "memory_persistence_soak_webui")
        soak_turns = (
            module.TurnResult(
                user_text="we are testing the assistant viability gate",
                assistant_text="When you say memory here, I mean my local saved memory rather than system RAM.",
                route="agent_memory",
                ok=True,
                surface="webui",
                trace_id="trace-5",
                status=200,
                payload={
                    "ok": True,
                    "setup": {
                        "kind": "working_context",
                        "current_topic": "assistant_viability_gate",
                        "last_request": "we are testing the assistant viability gate",
                    },
                },
            ),
            module.TurnResult(
                user_text="what are we doing?",
                assistant_text="It looks like we were focused on assistant_viability_gate.",
                route="agent_memory",
                ok=True,
                surface="webui",
                trace_id="trace-6",
                status=200,
                payload={
                    "ok": True,
                    "setup": {
                        "kind": "working_context",
                        "current_topic": "assistant_viability_gate",
                        "last_request": "we are testing the assistant viability gate",
                    },
                },
            ),
            module.TurnResult(
                user_text="what is the runtime status?",
                assistant_text="Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
                route="runtime_status",
                ok=True,
                surface="webui",
                trace_id="trace-7",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="what were we working on before?",
                assistant_text="It looks like we were focused on assistant_viability_gate.",
                route="agent_memory",
                ok=True,
                surface="webui",
                trace_id="trace-8",
                status=200,
                payload={
                    "ok": True,
                    "setup": {
                        "kind": "working_context",
                        "current_topic": "assistant_viability_gate",
                        "last_request": "we are testing the assistant viability gate",
                    },
                },
            ),
            module.TurnResult(
                user_text="what are we doing right now?",
                assistant_text="It looks like we were focused on assistant_viability_gate.",
                route="agent_memory",
                ok=True,
                surface="webui",
                trace_id="trace-9",
                status=200,
                payload={
                    "ok": True,
                    "setup": {
                        "kind": "working_context",
                        "current_topic": "assistant_viability_gate",
                        "last_request": "we are testing the assistant viability gate",
                    },
                },
            ),
        )
        soak_result = module.evaluate_scenario(soak_spec, soak_turns)
        self.assertTrue(soak_result.passed)

        soak_failure_turns = soak_turns[:-1] + (
            module.TurnResult(
                user_text="what are we doing right now?",
                assistant_text="I can help with that.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-10",
                status=200,
                payload={"ok": True},
            ),
        )
        soak_failure_result = module.evaluate_scenario(soak_spec, soak_failure_turns)
        self.assertFalse(soak_failure_result.passed)
        self.assertEqual("memory/continuity", soak_failure_result.failure_category)

        long_spec = next(spec for spec in module.SCENARIOS if spec.id == "long_human_like_session_webui")
        long_turns = (
            module.TurnResult(
                user_text="I'm testing whether you can stay coherent through a long chat.",
                assistant_text="Sure, I can help with that.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-11",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="What are we working on right now?",
                assistant_text="We are testing the assistant viability gate.",
                route="agent_memory",
                ok=True,
                surface="webui",
                trace_id="trace-12",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="Actually, keep the answer short.",
                assistant_text="Absolutely.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-13",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="No, go back and explain the larger task.",
                assistant_text="We are testing the assistant viability gate and keeping the assistant coherent across mixed turns.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-14",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="What is the runtime status?",
                assistant_text="Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
                route="runtime_status",
                ok=True,
                surface="webui",
                trace_id="trace-15",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="What do you remember about my preferences?",
                assistant_text="You prefer concise replies.",
                route="agent_memory",
                ok=True,
                surface="webui",
                trace_id="trace-16",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="List the files in this repo.",
                assistant_text="/home/c/personal-agent contains 3 entries: agent/, scripts/, tests/.",
                route="action_tool",
                ok=True,
                surface="webui",
                trace_id="trace-17",
                status=200,
                payload={"ok": True, "used_tools": ["filesystem"]},
            ),
            module.TurnResult(
                user_text="Read the file /home/c/personal-agent/README.md.",
                assistant_text="Text from /home/c/personal-agent/README.md (26 bytes).\nCANARY_TOKEN=abc123",
                route="action_tool",
                ok=True,
                surface="webui",
                trace_id="trace-18",
                status=200,
                payload={"ok": True, "used_tools": ["filesystem"]},
            ),
            module.TurnResult(
                user_text="What skill packs can you use for extra abilities?",
                assistant_text="Here is what I can help with right now: coding helpers, filesystem helpers, and more.",
                route="assistant_capabilities",
                ok=True,
                surface="webui",
                trace_id="trace-19",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="Okay, now summarize the work in one sentence.",
                assistant_text="We are testing the assistant viability gate across memory, runtime, files, and skills.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-20",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="If you had to continue from here, what would you do next?",
                assistant_text="Next I would keep the same topic, verify the key surfaces again, and continue the soak.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-21",
                status=200,
                payload={"ok": True},
            ),
        )
        long_result = module.evaluate_scenario(long_spec, long_turns)
        self.assertTrue(long_result.passed)

        long_failure_turns = long_turns[:-1] + (
            module.TurnResult(
                user_text="If you had to continue from here, what would you do next?",
                assistant_text="I can help with that.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-22",
                status=200,
                payload={"ok": True},
            ),
        )
        long_failure_result = module.evaluate_scenario(long_spec, long_failure_turns)
        self.assertFalse(long_failure_result.passed)
        self.assertEqual("assistant-behavior", long_failure_result.failure_category)

    def test_evaluate_scenario_flags_confirmation_and_leaks_as_expected(self) -> None:
        module = self.module
        confirm_spec = next(spec for spec in module.SCENARIOS if spec.id == "preview_confirm_flow_webui")
        confirm_turns = (
            module.TurnResult(
                user_text="switch temporarily to ollama:qwen2.5:7b-instruct",
                assistant_text="Okay.",
                route="model_status",
                ok=True,
                surface="webui",
                trace_id="trace-5",
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="yes",
                assistant_text="Now using ollama:qwen2.5:7b-instruct for chat.",
                route="model_status",
                ok=True,
                surface="webui",
                trace_id="trace-6",
                payload={"ok": True},
            ),
        )
        confirm_result = module.evaluate_scenario(confirm_spec, confirm_turns)
        self.assertFalse(confirm_result.passed)
        self.assertEqual("confirmation/action", confirm_result.failure_category)

        leak_spec = next(spec for spec in module.SCENARIOS if spec.id == "greeting_followup_webui")
        leak_turns = (
            module.TurnResult(
                user_text="hi",
                assistant_text="Hello. trace_id: abc123",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-7",
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="what can you help me with right now?",
                assistant_text="I can help with that.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-8",
                payload={"ok": True},
            ),
        )
        leak_result = module.evaluate_scenario(leak_spec, leak_turns)
        self.assertFalse(leak_result.passed)
        self.assertEqual("assistant-behavior", leak_result.failure_category)

    def test_topic_shift_scenario_distinguishes_canned_reply_from_http_400(self) -> None:
        module = self.module
        spec = next(spec for spec in module.SCENARIOS if spec.id == "interruption_topic_shift_webui")

        canned_turns = (
            module.TurnResult(
                user_text="help me plan my day",
                assistant_text="I do not have any active tasks or urgent open loops saved right now.",
                route="plan_day",
                ok=True,
                surface="webui",
                trace_id="trace-9",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="actually, give me a one-line joke",
                assistant_text="The runtime is ready. Chat is temporarily busy, so try again in a moment or ask for status or setup help.",
                route="generic_chat",
                ok=True,
                surface="webui",
                trace_id="trace-10",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="go back to the day plan",
                assistant_text="It looks like we were focused on today_plan.",
                route="agent_memory",
                ok=True,
                surface="webui",
                trace_id="trace-11",
                status=200,
                payload={"ok": True},
            ),
        )
        canned_result = module.evaluate_scenario(spec, canned_turns)
        self.assertFalse(canned_result.passed)
        self.assertEqual("assistant-behavior", canned_result.failure_category)
        self.assertIn("busy fallback or placeholder text surfaced", canned_result.failure_reason)
        self.assertIn("status=200", canned_result.evidence)

        transport_turns = (
            module.TurnResult(
                user_text="help me plan my day",
                assistant_text="I do not have any active tasks or urgent open loops saved right now.",
                route="plan_day",
                ok=True,
                surface="webui",
                trace_id="trace-12",
                status=200,
                payload={"ok": True},
            ),
            module.TurnResult(
                user_text="actually, give me a one-line joke",
                assistant_text="",
                route="error",
                ok=False,
                surface="webui",
                trace_id="trace-13",
                status=400,
                error="HTTP 400",
            ),
            module.TurnResult(
                user_text="go back to the day plan",
                assistant_text="It looks like we were focused on today_plan.",
                route="agent_memory",
                ok=True,
                surface="webui",
                trace_id="trace-14",
                status=200,
                payload={"ok": True},
            ),
        )
        transport_result = module.evaluate_scenario(spec, transport_turns)
        self.assertFalse(transport_result.passed)
        self.assertEqual("transport/runtime", transport_result.failure_category)
        self.assertEqual("HTTP 400", transport_result.failure_reason)
        self.assertIn("status=400", transport_result.evidence)

    def test_list_mode_prints_minimum_gate_and_scenario_ids(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = self.module.main(["--list"])
        self.assertEqual(0, exit_code)
        rendered = stdout.getvalue()
        self.assertIn("Assistant viability catalog", rendered)
        self.assertIn("Minimum gate: greeting_followup_webui, runtime_status_followup_webui, continuity_resume_webui", rendered)
        self.assertIn("telegram_surface_behavior", rendered)
        self.assertIn("webui_surface_behavior", rendered)


if __name__ == "__main__":
    unittest.main()

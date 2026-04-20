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
        self.assertEqual(8, len(ids))
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

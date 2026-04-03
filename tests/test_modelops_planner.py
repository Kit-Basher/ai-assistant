from __future__ import annotations

import json
import tempfile
import unittest

from agent.modelops.planner import ModelOpsPlanner


class TestModelOpsPlanner(unittest.TestCase):
    def test_deterministic_plan_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = ModelOpsPlanner(installer_script_path=f"{tmpdir}/install_ollama.sh")
            payload = {
                "model": "hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF",
                "estimated_download_gb": 4.0,
            }
            plan_a = planner.plan("modelops.pull_ollama_model", payload)
            plan_b = planner.plan("modelops.pull_ollama_model", payload)

        self.assertEqual(plan_a, plan_b)
        self.assertEqual(
            json.dumps(plan_a, ensure_ascii=True, sort_keys=True),
            json.dumps(plan_b, ensure_ascii=True, sort_keys=True),
        )

    def test_unsupported_action_raises(self) -> None:
        planner = ModelOpsPlanner(installer_script_path="/tmp/install_ollama.sh")
        with self.assertRaises(ValueError):
            planner.plan("modelops.delete_everything", {})


if __name__ == "__main__":
    unittest.main()

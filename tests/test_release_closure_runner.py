from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v0_2_1_release_closure.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v0_2_1_release_closure", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("release_closure_module_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReleaseClosureRunnerTests(unittest.TestCase):
    def test_gate_order_and_no_uninstall_marker_enablement(self) -> None:
        module = _load_module()
        gates = module.build_gates(expected_commit="abc123", include_primary_update=True)
        self.assertEqual("checkpoint", gates[0].phase)
        self.assertEqual("promotion", [gate.phase for gate in gates][3])
        commands = [" ".join(gate.command) for gate in gates]
        joined = "\n".join(commands)
        self.assertIn("primary_uninstall_enablement_smoke.py", joined)
        self.assertIn("primary_update_enablement_smoke.py", joined)
        self.assertNotIn("primary_uninstall_policy.py enable", joined)
        self.assertNotIn("--acknowledge-primary-uninstall-capability", joined)

    def test_can_skip_primary_update_for_dry_audit(self) -> None:
        module = _load_module()
        gates = module.build_gates(expected_commit="abc123", include_primary_update=False)
        joined = "\n".join(" ".join(gate.command) for gate in gates)
        self.assertNotIn("primary_update_enablement_smoke.py", joined)
        self.assertIn("primary_uninstall_enablement_smoke.py", joined)


if __name__ == "__main__":
    unittest.main()

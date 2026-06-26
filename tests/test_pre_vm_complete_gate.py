from __future__ import annotations

import unittest

from scripts import prove_pre_vm_complete


class TestPreVmCompleteGate(unittest.TestCase):
    def test_subsystems_mark_backup_restore_hardened_after_proof(self) -> None:
        subsystems = {row.name: row for row in prove_pre_vm_complete._subsystems()}  # noqa: SLF001

        self.assertIn("Backup/restore", subsystems)
        backup = subsystems["Backup/restore"]
        self.assertEqual("hardened", backup.status)
        self.assertFalse(backup.blocker)
        self.assertIn("dry-run restore", " ".join(backup.evidence))
        self.assertIn("version mismatch", " ".join(backup.evidence))

    def test_former_unknown_areas_are_documented_partials(self) -> None:
        subsystems = {row.name: row for row in prove_pre_vm_complete._subsystems()}  # noqa: SLF001

        webui = subsystems["Web UI robustness"]
        self.assertEqual("partial", webui.status)
        self.assertFalse(webui.blocker)
        self.assertFalse(webui.unknown)
        self.assertIn("webui_robustness_smoke.py", " ".join(webui.evidence))

        release_ci = subsystems["Release/CI automation"]
        self.assertEqual("partial", release_ci.status)
        self.assertFalse(release_ci.blocker)
        self.assertFalse(release_ci.unknown)
        self.assertIn("RELEASE_GATE_MATRIX.md", " ".join(release_ci.evidence))

    def test_required_operator_docs_are_present(self) -> None:
        ok, missing = prove_pre_vm_complete._audit_doc_ready()  # noqa: SLF001

        self.assertTrue(ok, missing)


if __name__ == "__main__":
    unittest.main()

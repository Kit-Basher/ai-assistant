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

    def test_unknown_areas_are_not_marked_complete(self) -> None:
        subsystems = prove_pre_vm_complete._subsystems()  # noqa: SLF001
        unknowns = [row for row in subsystems if row.unknown]

        self.assertGreaterEqual(len(unknowns), 1)
        self.assertTrue(any(row.name == "Web UI robustness" for row in unknowns))
        self.assertTrue(any(row.status != "hardened" for row in unknowns))

    def test_required_operator_docs_are_present(self) -> None:
        ok, missing = prove_pre_vm_complete._audit_doc_ready()  # noqa: SLF001

        self.assertTrue(ok, missing)


if __name__ == "__main__":
    unittest.main()

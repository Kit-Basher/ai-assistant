from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import full_adversarial_authorization_proof as proof


class FullAdversarialAuthorizationProofTests(unittest.TestCase):
    def test_inventory_matches_implemented_cases(self) -> None:
        cases = proof.load_cases()
        self.assertEqual(set(proof.CASE_FUNCTIONS), {case.case_id for case in cases})
        proven = {prop for case in cases for prop in case.property_ids}
        self.assertEqual(set(proof.PROPERTIES), proven)

    def test_full_proof_has_no_failures_and_one_documented_limitation(self) -> None:
        cases = proof.load_cases()
        results = proof.run_cases(cases)
        failed = [result for result in results if result.status == "FAIL"]
        warned = [result for result in results if result.status == "WARN"]
        self.assertEqual([], failed)
        self.assertEqual(["AUTH-P10-002"], [result.case.case_id for result in warned])
        self.assertEqual({*proof.PROPERTIES}, {prop for result in results for prop in result.case.property_ids})

    def test_evidence_artifact_is_machine_readable(self) -> None:
        cases = proof.load_cases()
        results = proof.run_cases(cases[:3])
        with tempfile.TemporaryDirectory() as raw:
            path = proof.write_evidence(results, Path(raw) / "evidence.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual("full_adversarial_authorization_proof_v1", payload["proof"])
        self.assertEqual(3, payload["case_count"])
        self.assertEqual(3, payload["pass"])
        self.assertIn("results", payload)


if __name__ == "__main__":
    unittest.main()

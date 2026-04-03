from __future__ import annotations

import unittest

from agent.friction.canary import run_friction_canaries


class TestFrictionCanary(unittest.TestCase):
    def test_friction_canary_has_no_failures(self) -> None:
        result = run_friction_canaries()
        self.assertEqual(0, result["failed"])
        self.assertEqual(result["total"], result["passed"] + result["failed"])

    def test_failed_names_order_is_deterministic(self) -> None:
        result = run_friction_canaries()
        names = result.get("failed_names") or tuple()
        self.assertEqual(tuple(sorted(names)), tuple(names))


if __name__ == "__main__":
    unittest.main()


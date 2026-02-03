import unittest

from agent.policy import evaluate_policy


class TestPolicy(unittest.TestCase):
    def test_delete_requires_confirmation(self) -> None:
        decision = evaluate_policy(["db:write"], ["db:write"], {"action_type": "delete"})
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.requires_confirmation)

    def test_permission_denied(self) -> None:
        decision = evaluate_policy(["db:read"], ["db:write"], {"action_type": "insert"})
        self.assertFalse(decision.allowed)


if __name__ == "__main__":
    unittest.main()

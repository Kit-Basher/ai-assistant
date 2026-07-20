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

    def test_unknown_write_like_operation_fails_closed_to_confirmation(self) -> None:
        decision = evaluate_policy(["db:write"], ["db:write"], {"action_type": "insert"})
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.requires_confirmation)

    def test_explicit_observe_operation_remains_read_only(self) -> None:
        decision = evaluate_policy(["db:read"], ["db:read"], {"action_type": "observe"})
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_confirmation)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.executor_registry import ExecutorRegistry, ExecutorSpec


class TestExecutorRegistryHardening(unittest.TestCase):
    def test_duplicate_and_frozen_registration_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            registry = ExecutorRegistry(Path(raw) / "journal.jsonl")
            registry.register(
                ExecutorSpec(
                    executor_id="operator.fixture.v1",
                    action_type="operator.fixture",
                    status="enabled",
                    capability_id="files.create",
                )
            )
            with self.assertRaisesRegex(ValueError, "duplicate_executor_action_type"):
                registry.register(
                    ExecutorSpec(
                        executor_id="operator.fixture.other.v1",
                        action_type="operator.fixture",
                        status="enabled",
                        capability_id="files.create",
                    )
                )
            with self.assertRaisesRegex(ValueError, "duplicate_executor_id"):
                registry.register(
                    ExecutorSpec(
                        executor_id="operator.fixture.v1",
                        action_type="operator.fixture.other",
                        status="enabled",
                        capability_id="files.create",
                    )
                )
            registry.freeze()
            with self.assertRaisesRegex(ValueError, "executor_registry_frozen"):
                registry.register(
                    ExecutorSpec(
                        executor_id="operator.fixture.after_freeze.v1",
                        action_type="operator.fixture.after_freeze",
                        status="enabled",
                        capability_id="files.create",
                    )
                )


if __name__ == "__main__":
    unittest.main()

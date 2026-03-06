from __future__ import annotations

import io
import logging
import sys
import unittest

from agent.doctor import _check_logging_to_stdout
from agent.logging_bootstrap import configure_logging_if_needed


class TestLoggingBootstrap(unittest.TestCase):
    def setUp(self) -> None:
        self.root = logging.getLogger()
        self.saved_handlers = list(self.root.handlers)
        self.saved_level = self.root.level
        for handler in list(self.root.handlers):
            self.root.removeHandler(handler)
        self.root.setLevel(logging.NOTSET)

    def tearDown(self) -> None:
        for handler in list(self.root.handlers):
            self.root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        for handler in self.saved_handlers:
            self.root.addHandler(handler)
        self.root.setLevel(self.saved_level)

    def test_bootstrap_adds_stdout_handler_when_none_configured(self) -> None:
        stream = io.StringIO()
        configured = configure_logging_if_needed(stream=stream)
        self.assertTrue(configured)
        self.assertEqual(1, len(self.root.handlers))
        handler = self.root.handlers[0]
        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertIs(stream, getattr(handler, "stream", None))
        self.assertEqual(logging.INFO, self.root.level)

    def test_bootstrap_does_not_duplicate_handlers(self) -> None:
        stream = io.StringIO()
        configured_first = configure_logging_if_needed(stream=stream)
        configured_second = configure_logging_if_needed(stream=stream)
        self.assertTrue(configured_first)
        self.assertFalse(configured_second)
        self.assertEqual(1, len(self.root.handlers))

    def test_bootstrap_does_not_override_existing_logging_config(self) -> None:
        handler = logging.StreamHandler(sys.stderr)
        self.root.addHandler(handler)
        self.root.setLevel(logging.ERROR)
        configured = configure_logging_if_needed(stream=io.StringIO())
        self.assertFalse(configured)
        self.assertEqual(1, len(self.root.handlers))
        self.assertIs(handler, self.root.handlers[0])
        self.assertEqual(logging.ERROR, self.root.level)

    def test_doctor_logging_check_passes_when_bootstrap_active(self) -> None:
        configure_logging_if_needed(stream=io.StringIO())
        check = _check_logging_to_stdout()
        self.assertEqual("OK", check.status)
        self.assertIn("stdout/journald handlers=", check.detail_short)


if __name__ == "__main__":
    unittest.main()

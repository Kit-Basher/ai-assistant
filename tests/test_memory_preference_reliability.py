from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.onboarding_flow import (
    load_onboarding_state,
    mark_onboarding_completed_reliable,
    onboarding_completed_key,
    onboarding_intent_hint_key,
)
from memory.db import MemoryDB


def _schema_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))


class TestMemoryPreferenceReliability(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db = MemoryDB(os.path.join(self.tmpdir.name, "memory.db"))
        self.db.init_schema(_schema_path())

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_preference_save_journals_and_verifies(self) -> None:
        result = self.db.set_preference_reliable("response_style", "concise")

        self.assertTrue(result["ok"])
        self.assertEqual("concise", self.db.get_preference("response_style"))
        journal = result["managed_action_journal"]
        self.assertEqual("memory.preference.write", journal["action_type"])
        self.assertTrue(journal["verification_result"]["ok"])
        self.assertEqual("No rollback needed.", journal["rollback_result"]["summary"])

    def test_failed_preference_verification_restores_previous_value(self) -> None:
        self.db.set_preference("response_style", "concise")

        with patch.object(self.db, "_verify_preference_write", return_value=False):
            result = self.db.set_preference_reliable("response_style", "verbose")

        self.assertFalse(result["ok"])
        self.assertTrue(result["rollback_ok"])
        self.assertEqual("concise", self.db.get_preference("response_style"))
        self.assertIn("preference update did not finish", result["message"])
        self.assertIn("previous setting/state was restored", result["message"])
        payload = json.dumps(result, sort_keys=True)
        self.assertNotIn("concise", payload)
        self.assertNotIn("verbose", payload)

    def test_failed_new_preference_verification_removes_only_owned_key(self) -> None:
        self.db.set_preference("unrelated", "keep")

        with patch.object(self.db, "_verify_preference_write", return_value=False):
            result = self.db.set_preference_reliable("new_pref", "new-value")

        self.assertFalse(result["ok"])
        self.assertIsNone(self.db.get_preference("new_pref"))
        self.assertEqual("keep", self.db.get_preference("unrelated"))
        self.assertIn("removed failed new preference", result["message"])

    def test_preference_preflight_rejects_unknown_key_shape(self) -> None:
        with self.assertRaises(ValueError):
            self.db.set_preference_reliable("../profile", "unsafe")

    def test_preference_preflight_rejects_unknown_scope(self) -> None:
        with self.assertRaises(ValueError):
            self.db.set_preference_reliable("response_style", "concise", scope="other_state")

    def test_per_thread_preference_rollback_does_not_touch_global_prefs(self) -> None:
        self.db.set_user_pref("show_summary", "off")
        self.db.set_thread_pref("thread-a", "show_summary", "on")

        with patch.object(self.db, "_verify_thread_pref_write", return_value=False):
            result = self.db.set_thread_pref_reliable("thread-a", "show_summary", "off")

        self.assertFalse(result["ok"])
        self.assertEqual("on", self.db.get_thread_pref("thread-a", "show_summary"))
        self.assertEqual("off", self.db.get_user_pref("show_summary"))
        payload = json.dumps(result, sort_keys=True)
        self.assertNotIn("thread-a", payload)
        self.assertIn("thread_hash", payload)

    def test_onboarding_state_write_journals_and_verifies(self) -> None:
        result = mark_onboarding_completed_reliable(self.db, "user-123", intent_hint="system")

        self.assertTrue(result["ok"])
        state = load_onboarding_state(self.db, "user-123")
        self.assertTrue(state["completed"])
        self.assertEqual("system", state["intent_hint"])
        journal = result["managed_action_journal"]
        self.assertEqual("onboarding.completed.write", journal["action_type"])
        self.assertTrue(journal["verification_result"]["ok"])

    def test_failed_onboarding_verification_restores_previous_state(self) -> None:
        self.db.set_user_pref(onboarding_completed_key("user-123"), "false")
        self.db.set_user_pref(onboarding_intent_hint_key("user-123"), "coding")

        with patch("agent.onboarding_flow.load_onboarding_state", return_value={"available": True, "completed": False}):
            result = mark_onboarding_completed_reliable(self.db, "user-123", intent_hint="system")

        self.assertFalse(result["ok"])
        self.assertTrue(result["rollback_ok"])
        self.assertIn("onboarding update did not finish", result["message"])
        self.assertEqual("false", self.db.get_user_pref(onboarding_completed_key("user-123")))
        self.assertEqual("coding", self.db.get_user_pref(onboarding_intent_hint_key("user-123")))

    def test_memory_bootstrap_completion_marker_journals_and_verifies(self) -> None:
        result = self.db.set_memory_bootstrap_marker_reliable("memory_v2_completed", "true")

        self.assertTrue(result["ok"])
        self.assertEqual("true", self.db.get_preference("memory_bootstrap:memory_v2_completed"))
        self.assertEqual("memory.bootstrap_marker.write", result["managed_action_journal"]["action_type"])

    def test_failed_bootstrap_marker_verification_restores_owned_marker(self) -> None:
        self.db.set_memory_bootstrap_marker_reliable("memory_v2_completed", "false")

        with patch.object(self.db, "_verify_preference_write", return_value=False):
            result = self.db.set_memory_bootstrap_marker_reliable("memory_v2_completed", "true")

        self.assertFalse(result["ok"])
        self.assertEqual("false", self.db.get_preference("memory_bootstrap:memory_v2_completed"))

    def test_failed_new_bootstrap_marker_verification_removes_owned_marker_only(self) -> None:
        self.db.set_preference("memory_bootstrap:other_marker", "keep")

        with patch.object(self.db, "_verify_preference_write", return_value=False):
            result = self.db.set_memory_bootstrap_marker_reliable("memory_v2_completed", "true")

        self.assertFalse(result["ok"])
        self.assertIsNone(self.db.get_preference("memory_bootstrap:memory_v2_completed"))
        self.assertEqual("keep", self.db.get_preference("memory_bootstrap:other_marker"))

    def test_no_raw_private_memory_content_appears_in_journal(self) -> None:
        secret = "private memory: my token is sk-test and path /home/c/private/history.json"
        result = self.db.set_user_pref_reliable("profile_summary", secret)

        payload = json.dumps(result, sort_keys=True)
        self.assertNotIn(secret, payload)
        self.assertNotIn("sk-test", payload)
        self.assertNotIn("/home/c/private/history.json", payload)


if __name__ == "__main__":
    unittest.main()

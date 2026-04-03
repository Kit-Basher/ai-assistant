from __future__ import annotations

import tempfile
import unittest

from agent.llm.model_discovery import build_model_discovery_proposals
from agent.llm.model_discovery_policy import ModelDiscoveryPolicyStore


class TestModelDiscovery(unittest.TestCase):
    def test_policy_store_persists_known_good_and_known_stale_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelDiscoveryPolicyStore(path=f"{tmpdir}/model_discovery_policy.json")
            good = store.upsert_entry(
                "openrouter:vendor/coder-pro",
                status="known_good",
                role_hints=["coding"],
                notes="Reviewed as a strong coding option.",
                reviewed_at="2026-03-30T00:00:00Z",
            )
            stale = store.upsert_entry(
                "openrouter:vendor/old-chat",
                status="known_stale",
                notes="Superseded by newer reviewed options.",
                reviewed_at="2026-03-29T00:00:00Z",
            )

            reloaded = ModelDiscoveryPolicyStore(path=f"{tmpdir}/model_discovery_policy.json")
            entries = {row["model_id"]: row for row in reloaded.list_entries()}

        self.assertEqual("known_good", good["status"])
        self.assertEqual(["coding"], good["role_hints"])
        self.assertEqual("known_stale", stale["status"])
        self.assertIn("openrouter:vendor/coder-pro", entries)
        self.assertIn("openrouter:vendor/old-chat", entries)
        self.assertEqual("known_good", entries["openrouter:vendor/coder-pro"]["status"])
        self.assertEqual("known_stale", entries["openrouter:vendor/old-chat"]["status"])

    def test_policy_store_updates_removes_and_rejects_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelDiscoveryPolicyStore(path=f"{tmpdir}/model_discovery_policy.json")
            first = store.upsert_entry(
                "openrouter:vendor/coder-pro",
                status="known_good",
                role_hints=["coding"],
                notes="Initial review.",
            )
            updated = store.upsert_entry(
                "openrouter:vendor/coder-pro",
                status="avoid",
                role_hints=["research"],
                notes="Reclassified after review.",
                justification="Superseded in practice.",
            )
            removed = store.remove_entry("openrouter:vendor/coder-pro")
            missing = store.remove_entry("openrouter:vendor/coder-pro")

            with self.assertRaisesRegex(ValueError, "invalid_model_discovery_policy_entry"):
                store.upsert_entry("openrouter:vendor/bad", status="unsupported_status")
            with self.assertRaisesRegex(ValueError, "invalid_model_discovery_role_hints"):
                store.upsert_entry(
                    "openrouter:vendor/bad",
                    status="known_good",
                    role_hints=["unsupported_hint"],
                )

        self.assertEqual("known_good", first["status"])
        self.assertEqual("avoid", updated["status"])
        self.assertEqual(["research"], updated["role_hints"])
        self.assertEqual("Reclassified after review.", updated["notes"])
        self.assertTrue(removed)
        self.assertFalse(missing)

    def test_discovery_builds_structured_good_stale_and_insufficient_proposals(self) -> None:
        inventory_rows = [
            {
                "model_id": "openrouter:vendor/coder-pro",
                "provider_id": "openrouter",
                "model_name": "vendor/coder-pro",
                "capabilities": ["chat", "tools"],
                "task_types": ["coding", "general_chat"],
                "context_window": 65536,
                "price_in": 1.0,
                "price_out": 2.0,
                "available": True,
                "local": False,
            },
            {
                "model_id": "openrouter:vendor/research-pro",
                "provider_id": "openrouter",
                "model_name": "vendor/research-pro",
                "capabilities": ["chat"],
                "task_types": ["reasoning", "general_chat"],
                "context_window": 262144,
                "price_in": 2.0,
                "price_out": 4.0,
                "available": True,
                "local": False,
            },
            {
                "model_id": "openrouter:vendor/cheap-chat",
                "provider_id": "openrouter",
                "model_name": "vendor/cheap-chat",
                "capabilities": ["chat"],
                "task_types": ["general_chat"],
                "context_window": 65536,
                "price_in": 0.1,
                "price_out": 0.2,
                "available": True,
                "local": False,
            },
            {
                "model_id": "openrouter:vendor/old-chat",
                "provider_id": "openrouter",
                "model_name": "vendor/old-chat",
                "capabilities": ["chat"],
                "task_types": ["general_chat"],
                "context_window": 32768,
                "price_in": 1.0,
                "price_out": 1.0,
                "available": True,
                "local": False,
            },
            {
                "model_id": "openrouter:vendor/unknown",
                "provider_id": "openrouter",
                "model_name": "vendor/unknown",
                "capabilities": ["chat"],
                "task_types": [],
                "available": True,
                "local": False,
            },
        ]
        policy_entries = [
            {
                "model_id": "openrouter:vendor/coder-pro",
                "status": "known_good",
                "role_hints": ["coding"],
                "notes": "Reviewed and trusted for coding.",
                "source": "operator_review",
            },
            {
                "model_id": "openrouter:vendor/old-chat",
                "status": "known_stale",
                "role_hints": [],
                "notes": "Reviewed as stale.",
                "source": "operator_review",
            },
        ]

        proposals = build_model_discovery_proposals(
            inventory_rows=inventory_rows,
            policy_entries=policy_entries,
            cheap_remote_cap_per_1m=1.0,
        )
        by_model = {row["model_id"]: row for row in proposals}

        coding = by_model["openrouter:vendor/coder-pro"]
        research = by_model["openrouter:vendor/research-pro"]
        cheap = by_model["openrouter:vendor/cheap-chat"]
        stale = by_model["openrouter:vendor/old-chat"]
        unknown = by_model["openrouter:vendor/unknown"]

        self.assertEqual("candidate_good", coding["proposal_kind"])
        self.assertIn("coding", coding["proposed_roles"])
        self.assertEqual("known_good", coding["policy_status"])
        self.assertIn("explicit_coding_task_type", coding["reason_codes"])
        self.assertTrue(bool((coding.get("review_suggestion") or {}).get("available")))
        self.assertEqual("known_good", (coding.get("review_suggestion") or {}).get("suggested_status"))
        self.assertEqual("/llm/models/policy", (coding.get("review_suggestion") or {}).get("write_surface"))
        self.assertEqual(["coding"], (coding.get("review_suggestion") or {}).get("suggested_role_hints"))
        self.assertEqual(
            "known_good",
            ((((coding.get("review_suggestion") or {}).get("payload_template")) or {}).get("status")),
        )

        self.assertEqual("candidate_good", research["proposal_kind"])
        self.assertIn("research", research["proposed_roles"])
        self.assertIn("explicit_reasoning_task_type", research["reason_codes"])
        self.assertEqual(262144, research["evidence"]["context_window"])
        self.assertTrue(bool((research.get("review_suggestion") or {}).get("available")))
        self.assertEqual("known_good", (research.get("review_suggestion") or {}).get("suggested_status"))
        self.assertEqual(["research"], (research.get("review_suggestion") or {}).get("suggested_role_hints"))

        self.assertEqual("candidate_good", cheap["proposal_kind"])
        self.assertIn("cheap_cloud", cheap["proposed_roles"])
        self.assertIn("explicit_low_cost_remote", cheap["reason_codes"])
        self.assertTrue(bool((cheap.get("review_suggestion") or {}).get("available")))
        self.assertEqual("known_good", (cheap.get("review_suggestion") or {}).get("suggested_status"))
        self.assertEqual(["cheap_cloud"], (cheap.get("review_suggestion") or {}).get("suggested_role_hints"))

        self.assertEqual("candidate_stale", stale["proposal_kind"])
        self.assertEqual("known_stale", stale["policy_status"])
        self.assertTrue(bool((stale.get("review_suggestion") or {}).get("available")))
        self.assertEqual("known_stale", (stale.get("review_suggestion") or {}).get("suggested_status"))
        self.assertEqual([], (stale.get("review_suggestion") or {}).get("suggested_role_hints"))
        self.assertEqual("/llm/models/policy", (stale.get("review_suggestion") or {}).get("write_surface"))

        self.assertEqual("insufficient_metadata", unknown["proposal_kind"])
        self.assertEqual([], unknown["proposed_roles"])
        self.assertEqual(["insufficient_structured_metadata"], unknown["reason_codes"])
        self.assertFalse(bool((unknown.get("review_suggestion") or {}).get("available")))
        self.assertEqual("insufficient_metadata", (unknown.get("review_suggestion") or {}).get("reason_code"))

        for row in (coding, research, cheap, stale, unknown):
            self.assertTrue(bool(row["review_required"]))
            self.assertTrue(bool(row["non_canonical"]))
            self.assertEqual("not_adopted", row["canonical_status"])


if __name__ == "__main__":
    unittest.main()

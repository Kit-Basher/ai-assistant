from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import time
import tempfile
import unittest

from agent.api_server import AgentRuntime
from agent.policy import (
    build_mutator_plan,
    classify_operation,
    mutator_confirmation_required_payload,
    validate_mutator_apply,
)
from tests.test_api_packs_endpoints import _config
from tests.test_managed_local_services import _FakeSearchOpener


class TestPlanModePolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.skills_path = str(Path(__file__).resolve().parents[1] / "skills")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _runtime(self) -> AgentRuntime:
        config = replace(
            _config(self.registry_path, self.db_path, self.skills_path),
            search_enabled=False,
            search_provider="searxng",
            searxng_base_url=None,
        )
        return AgentRuntime(config)

    def _make_pack_dir(self, name: str = "plan-pack", *, adapter: bool = False) -> str:
        pack_dir = os.path.join(self.tmpdir.name, name)
        os.makedirs(pack_dir, exist_ok=True)
        if adapter:
            body = (
                "---\n"
                f"id: {name}\n"
                f"name: {name}\n"
                "version: 0.1.0\n"
                "description: plan mode pack\n"
                "managed_adapters:\n"
                "  - kind: local_file_import\n"
                "    purpose: selected fixture metadata grant\n"
                "    path_policy: user_selected_file_only\n"
                "    allowed_extensions: [.txt]\n"
                "    max_file_size_mb: 1\n"
                "    stores_local_index: false\n"
                "    network_allowed: false\n"
                "---\n"
                "# Plan Pack\n\n"
                "Safe text only.\n"
            )
        else:
            body = (
                "---\n"
                f"id: {name}\n"
                f"name: {name}\n"
                "version: 0.1.0\n"
                "description: plan mode pack\n"
                "---\n"
                "# Plan Pack\n\n"
                "Safe text only.\n"
            )
        with open(os.path.join(pack_dir, "SKILL.md"), "w", encoding="utf-8") as handle:
            handle.write(body)
        return pack_dir

    def _plan_apply_payload(self, plan_payload: dict[str, object]) -> dict[str, object]:
        plan = plan_payload["plan"]
        self.assertIsInstance(plan, dict)
        return {
            "plan_id": plan["plan_id"],
            "confirmation_token": plan["confirmation_token"],
            "mutation_plan": plan["mutation_plan"],
        }

    def test_unknown_operations_default_to_mutating(self) -> None:
        decision = classify_operation("do_magic")

        self.assertEqual("mutating", decision.classification)
        self.assertTrue(decision.requires_plan)
        self.assertTrue(decision.requires_confirmation)
        self.assertFalse(decision.allowed_without_confirmation)
        self.assertEqual("unknown_operations_default_to_mutating", decision.reason)

    def test_read_status_preview_search_are_allowed_without_confirmation(self) -> None:
        for action_type in ("safe_web_search.status", "managed_local_service.setup_plan", "external_pack.preview", "list"):
            decision = classify_operation(action_type)
            self.assertEqual("read_only", decision.classification, action_type)
            self.assertTrue(decision.allowed_without_confirmation, action_type)
            self.assertFalse(decision.requires_confirmation, action_type)

    def test_mutator_without_confirmation_returns_block_payload(self) -> None:
        payload = mutator_confirmation_required_payload("external_pack.install")

        self.assertFalse(payload["ok"])
        self.assertEqual("confirmation_required", payload["error"])
        self.assertEqual("mutating", payload["classification"])
        self.assertTrue(payload["requires_plan"])
        self.assertTrue(payload["requires_confirmation"])

    def test_mutator_plan_requires_expected_shape_and_token(self) -> None:
        plan = build_mutator_plan(
            action_type="external_pack.enable",
            resources={"created": [], "changed": ["pack:abc"], "deleted": []},
            rollback_scope="restore previous enablement metadata",
            rollback_supported=True,
            confirmation_token="confirm-token",
            expires_at=9999999999,
            plan_id="plan-1",
        )

        ok, error = validate_mutator_apply(
            plan,
            expected_action_type="external_pack.enable",
            confirmation_token="confirm-token",
            now=1,
        )
        self.assertTrue(ok)
        self.assertIsNone(error)
        tampered = dict(plan)
        tampered["action_type"] = "external_pack.install"
        ok, error = validate_mutator_apply(
            tampered,
            expected_action_type="external_pack.enable",
            confirmation_token="confirm-token",
            now=1,
        )
        self.assertFalse(ok)
        self.assertEqual("plan_action_type_mismatch", error)

    def test_managed_searxng_setup_plan_includes_policy_mutator_plan(self) -> None:
        runtime = self._runtime()
        body = runtime.search_setup_plan({"base_url": "http://127.0.0.1:8888"})

        plan = body["plan"]
        mutation_plan = plan["mutation_plan"]
        self.assertEqual("managed_local_service.setup_apply", mutation_plan["action_type"])
        self.assertEqual("mutating", mutation_plan["classification"])
        self.assertEqual(plan["confirmation_token"], mutation_plan["confirmation_token"])
        self.assertTrue(mutation_plan["rollback_supported"])
        self.assertIn("runtime_search_config", mutation_plan["resources"]["changed"])

    def test_managed_searxng_apply_rejects_tampered_policy_plan_before_executor(self) -> None:
        runtime = self._runtime()
        runtime._managed_local_service_executor = unittest.mock.Mock()  # noqa: SLF001
        plan_payload = runtime.search_setup_plan({"base_url": "http://127.0.0.1:8888"})
        plan = plan_payload["plan"]
        stored = runtime._search_setup_confirmations[plan["plan_id"]]  # noqa: SLF001
        stored["plan"]["mutation_plan"]["action_type"] = "external_pack.install"

        result = runtime.apply_search_setup({"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]})

        self.assertFalse(result["ok"])
        self.assertEqual("plan_action_type_mismatch", result["error"])
        runtime._managed_local_service_executor.execute_from_pending.assert_not_called()  # type: ignore[union-attr]

    def test_managed_searxng_apply_journals_executed_steps_and_changed_resources(self) -> None:
        runtime = self._runtime()
        plan_payload = runtime.search_setup_plan({"base_url": "http://127.0.0.1:8888"})
        plan = plan_payload["plan"]

        with unittest.mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeSearchOpener({"results": []})):
            result = runtime.apply_search_setup({"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]})

        self.assertTrue(result["ok"])
        journal = result["managed_action_journal"]
        self.assertTrue(journal.get("executed_steps"))
        changed = journal.get("changed_resources")
        self.assertTrue(changed)
        self.assertIn("runtime_search_config", str(changed))
        self.assertIn("search", str(changed))

    def test_pack_lifecycle_mutators_are_centrally_classified(self) -> None:
        runtime = self._runtime()

        for action_type in (
            "external_pack.install",
            "external_pack.approve",
            "external_pack.enable",
            "external_pack.grant",
            "external_pack.remove",
        ):
            payload = runtime.plan_mode_policy(action_type)
            self.assertEqual("mutating", payload["classification"], action_type)
            self.assertTrue(payload["requires_plan"], action_type)
            self.assertTrue(payload["requires_confirmation"], action_type)

    def test_external_pack_install_plan_apply_persists_pack(self) -> None:
        runtime = self._runtime()
        pack_dir = self._make_pack_dir("plan-install-pack")

        plan_payload = runtime.plan_pack_lifecycle("external_pack.install", {"source": pack_dir})
        self.assertTrue(plan_payload["ok"])
        plan = plan_payload["plan"]
        mutation_plan = plan["mutation_plan"]
        self.assertEqual("plan_mode", mutation_plan["policy_layer"])
        self.assertEqual("external_pack.install", mutation_plan["action_type"])
        self.assertEqual("mutating", mutation_plan["classification"])
        self.assertTrue(mutation_plan["requires_confirmation"])
        self.assertIn("external_pack_store", mutation_plan["resources"]["changed"])
        self.assertEqual(plan["confirmation_token"], mutation_plan["confirmation_token"])

        result = runtime.apply_pack_lifecycle("external_pack.install", self._plan_apply_payload(plan_payload))

        self.assertTrue(result["ok"])
        self.assertTrue(result["mutated"])
        self.assertEqual("plan-install-pack", result["pack"]["canonical_pack"]["audit"]["declared_id"])

    def test_external_pack_direct_mutator_endpoint_requires_plan(self) -> None:
        runtime = self._runtime()
        handler = _HandlerForPlanTest(runtime, "/packs/install", {"source": self._make_pack_dir("direct-blocked")})
        handler.do_POST()
        payload = handler.json_payload()

        self.assertEqual(400, handler.status_code)
        self.assertFalse(payload["ok"])
        self.assertEqual("confirmation_required", payload["error"])
        self.assertTrue(payload["requires_plan"])
        self.assertTrue(payload["requires_confirmation"])

    def test_external_pack_plan_apply_rejects_tampering(self) -> None:
        runtime = self._runtime()
        plan_payload = runtime.plan_pack_lifecycle("external_pack.install", {"source": self._make_pack_dir("tamper-pack")})
        apply_payload = self._plan_apply_payload(plan_payload)

        missing_plan = dict(apply_payload)
        missing_plan.pop("mutation_plan")
        self.assertEqual("plan_required", runtime.apply_pack_lifecycle("external_pack.install", missing_plan)["error"])

        bad_token = dict(apply_payload)
        bad_token["confirmation_token"] = "confirm-wrong"
        self.assertEqual("invalid_confirmation", runtime.apply_pack_lifecycle("external_pack.install", bad_token)["error"])

        bad_action = dict(apply_payload)
        bad_action["mutation_plan"] = dict(apply_payload["mutation_plan"])
        bad_action["mutation_plan"]["action_type"] = "external_pack.enable"
        self.assertEqual("plan_action_type_mismatch", runtime.apply_pack_lifecycle("external_pack.install", bad_action)["error"])

        bad_resources = dict(apply_payload)
        bad_resources["mutation_plan"] = dict(apply_payload["mutation_plan"])
        bad_resources["mutation_plan"]["resources"] = {"created": ["other"], "changed": [], "deleted": []}
        self.assertEqual("plan_apply_mismatch", runtime.apply_pack_lifecycle("external_pack.install", bad_resources)["error"])

        bad_plan_id = dict(apply_payload)
        bad_plan_id["mutation_plan"] = dict(apply_payload["mutation_plan"])
        bad_plan_id["mutation_plan"]["plan_id"] = "other-plan"
        self.assertEqual("plan_id_mismatch", runtime.apply_pack_lifecycle("external_pack.install", bad_plan_id)["error"])

    def test_external_pack_plan_apply_rejects_expired_plan(self) -> None:
        runtime = self._runtime()
        plan_payload = runtime.plan_pack_lifecycle("external_pack.install", {"source": self._make_pack_dir("expired-pack")})
        plan = plan_payload["plan"]
        runtime._pack_lifecycle_confirmations[plan["plan_id"]]["expires_at"] = time.time() - 1  # noqa: SLF001

        result = runtime.apply_pack_lifecycle("external_pack.install", self._plan_apply_payload(plan_payload))

        self.assertFalse(result["ok"])
        self.assertEqual("confirmation_expired", result["error"])

    def test_external_pack_approve_enable_grant_remove_use_plan_apply(self) -> None:
        runtime = self._runtime()
        pack_dir = self._make_pack_dir("plan-adapter-pack", adapter=True)
        install_ok, install_body = runtime.packs_install({"source": pack_dir})
        self.assertTrue(install_ok)
        pack_id = install_body["pack"]["pack_id"]

        approve_plan = runtime.plan_pack_lifecycle("external_pack.approve", {"pack_id": pack_id})
        approve = runtime.apply_pack_lifecycle("external_pack.approve", self._plan_apply_payload(approve_plan))
        self.assertTrue(approve["ok"])
        self.assertEqual(pack_id, approve["pack"]["pack_id"])

        enable_plan = runtime.plan_pack_lifecycle("external_pack.enable", {"pack_id": pack_id, "enabled": True})
        enabled = runtime.apply_pack_lifecycle("external_pack.enable", self._plan_apply_payload(enable_plan))
        self.assertTrue(enabled["ok"])
        self.assertTrue(enabled["pack"]["enabled"])

        selected = Path(self.tmpdir.name) / "selected.txt"
        selected.write_text("fixture", encoding="utf-8")
        grant_plan = runtime.plan_pack_lifecycle(
            "external_pack.grant",
            {
                "pack_id": pack_id,
                "requested_path": str(selected),
                "adapter": {
                    "kind": "local_file_import",
                    "purpose": "selected fixture metadata grant",
                    "path_policy": "user_selected_file_only",
                    "allowed_extensions": [".txt"],
                    "max_file_size_mb": 1,
                    "stores_local_index": False,
                    "network_allowed": False,
                },
            },
        )
        granted = runtime.apply_pack_lifecycle("external_pack.grant", self._plan_apply_payload(grant_plan))
        self.assertTrue(granted["ok"])
        self.assertFalse(granted["did_invoke_adapter"])
        self.assertFalse(granted["reads_file"])
        self.assertIn("managed_action_journal", granted)

        remove_plan = runtime.plan_pack_lifecycle("external_pack.remove", {"pack_id": pack_id})
        removed = runtime.apply_pack_lifecycle("external_pack.remove", self._plan_apply_payload(remove_plan))
        self.assertTrue(removed["ok"])
        self.assertIn("tombstone", str(removed.get("managed_action_journal", {})).lower())

    def test_external_pack_preview_list_search_are_read_only(self) -> None:
        runtime = self._runtime()
        for action_type in ("external_pack.list", "external_pack.preview", "external_pack.search", "external_pack.status"):
            payload = runtime.plan_mode_policy(action_type)
            self.assertEqual("read_only", payload["classification"], action_type)
            self.assertFalse(payload["requires_confirmation"], action_type)


class _HandlerForPlanTest:
    def __init__(self, runtime: AgentRuntime, path: str, payload: dict[str, object] | None = None) -> None:
        from tests.test_api_packs_endpoints import _HandlerForTest

        self._handler = _HandlerForTest(runtime, path, payload or {})

    @property
    def status_code(self) -> int:
        return self._handler.status_code

    def do_POST(self) -> None:
        self._handler.do_POST()

    def json_payload(self) -> dict[str, object]:
        import json

        return json.loads(self._handler.body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()

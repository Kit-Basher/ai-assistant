from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.llm.catalog import fetch_provider_catalog
from agent.config import Config
from agent.llm.autoconfig import build_autoconfig_plan
from agent.llm.model_manager import model_manager_state_path_for_runtime, save_model_manager_state
from agent.llm.self_heal import build_self_heal_plan
from agent.model_watch_catalog import write_snapshot_atomic


def _config(tmpdir: str, **overrides: object) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=os.path.join(tmpdir, "agent.db"),
        log_path=os.path.join(tmpdir, "agent.log"),
        skills_path="/tmp/skills",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="llama3",
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=15,
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url="http://127.0.0.1:11434",
        anthropic_api_key=None,
        llm_selector="single",
        llm_broker_policy_path=None,
        llm_allow_remote=True,
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=os.path.join(tmpdir, "registry.json"),
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=os.path.join(tmpdir, "usage.json"),
        llm_health_state_path=os.path.join(tmpdir, "health.json"),
        llm_automation_enabled=False,
        model_scout_state_path=os.path.join(tmpdir, "scout.json"),
        autopilot_notify_store_path=os.path.join(tmpdir, "notify.json"),
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _registry_document() -> dict[str, object]:
    return {
        "schema_version": 2,
        "providers": {
            "ollama": {"enabled": True, "local": True, "api_key_source": None},
            "openrouter": {
                "enabled": True,
                "local": False,
                "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
            },
        },
        "models": {
            "ollama:local": {
                "provider": "ollama",
                "model": "local",
                "capabilities": ["chat"],
                "quality_rank": 3,
                "cost_rank": 1,
                "enabled": True,
                "available": True,
                "max_context_tokens": 8192,
                "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
            },
            "openrouter:remote": {
                "provider": "openrouter",
                "model": "remote",
                "capabilities": ["chat"],
                "quality_rank": 8,
                "cost_rank": 2,
                "enabled": True,
                "available": True,
                "max_context_tokens": 65536,
                "pricing": {"input_per_million_tokens": 0.0, "output_per_million_tokens": 0.0},
            },
        },
        "defaults": {
            "routing_mode": "auto",
            "default_provider": None,
            "default_model": None,
            "allow_remote_fallback": True,
        },
    }


def _selection_result(*, model_id: str, provider_id: str, tier: str) -> dict[str, object]:
    candidate = {
        "model_id": model_id,
        "provider_id": provider_id,
        "local": tier == "local",
        "tier": tier,
        "utility": 0.7,
        "expected_cost_per_1m": 0.0 if tier != "cheap_remote" else 0.3,
        "quality_rank": 8,
        "context_window": 65536,
        "health_status": "ok",
    }
    return {
        "selected_candidate": candidate,
        "recommended_candidate": candidate,
        "current_candidate": None,
        "switch_recommended": True,
        "decision_reason": "current_unsuitable",
        "decision_detail": "Current default is not suitable; switch to the best policy-allowed candidate.",
        "utility_delta": 0.7,
        "rejected_candidates": [],
        "ordered_candidates": [candidate],
    }


class TestDefaultModelPolicyIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self._env_backup = dict(os.environ)
        os.environ["OPENROUTER_API_KEY"] = "sk-test"
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_autoconfig_uses_shared_policy_selector(self) -> None:
        with patch(
            "agent.llm.autoconfig.choose_best_default_chat_candidate",
            return_value=_selection_result(model_id="ollama:local", provider_id="ollama", tier="local"),
        ) as chooser:
            plan = build_autoconfig_plan(
                _registry_document(),
                {"providers": [{"id": "ollama", "status": "ok"}], "models": [{"id": "ollama:local", "status": "ok"}]},
                config=_config(self.tmpdir.name),
                env={"OPENROUTER_API_KEY": "sk-test"},
            )
        chooser.assert_called_once()
        self.assertEqual("ollama:local", plan["proposed_defaults"]["default_model"])

    def test_self_heal_uses_shared_policy_selector(self) -> None:
        document = _registry_document()
        document["defaults"]["default_model"] = "ollama:missing"  # type: ignore[index]
        with patch(
            "agent.llm.self_heal.choose_best_default_chat_candidate",
            return_value=_selection_result(model_id="ollama:local", provider_id="ollama", tier="local"),
        ) as chooser:
            plan = build_self_heal_plan(
                document,
                {"providers": [{"id": "ollama", "status": "ok"}], "models": [{"id": "ollama:local", "status": "ok"}]},
                config=_config(self.tmpdir.name),
                router_snapshot={"providers": [], "models": []},
            )
        chooser.assert_called_once()
        self.assertEqual("ollama:local", plan["proposed_defaults"]["default_model"])

    def test_bootstrap_uses_shared_policy_selector(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = _registry_document()
        runtime._save_registry_document(runtime.registry_document)
        with patch.object(
            runtime,
            "_select_chat_candidates_for_policy",
            return_value=_selection_result(model_id="ollama:local", provider_id="ollama", tier="local"),
        ) as selector:
            plan = runtime._bootstrap_plan()
        selector.assert_called_once()
        self.assertEqual("ollama:local", (plan.get("selected_candidate") or {}).get("model_id"))

    def test_runtime_selector_bridge_requires_runtime_truth_selector(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = _registry_document()
        runtime._save_registry_document(runtime.registry_document)

        with patch.object(runtime, "runtime_truth_service", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "runtime_truth_selector_unavailable"):
                runtime._select_chat_candidates_for_policy()

    def test_model_watch_uses_shared_policy_selector(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        document = _registry_document()
        document["defaults"]["default_provider"] = "ollama"  # type: ignore[index]
        document["defaults"]["default_model"] = "ollama:local"  # type: ignore[index]
        runtime.registry_document = document
        runtime._save_registry_document(runtime.registry_document)
        with patch.object(
            runtime,
            "_select_chat_candidates_for_policy",
            return_value={
                **_selection_result(model_id="openrouter:remote", provider_id="openrouter", tier="free_remote"),
                "current_candidate": {
                    "model_id": "ollama:local",
                    "provider_id": "ollama",
                    "local": True,
                    "tier": "local",
                    "utility": 0.1,
                    "utility_quality": 0.3,
                    "utility_latency": 0.25,
                    "utility_risk": 0.0,
                    "expected_cost_per_1m": 0.0,
                    "quality_rank": 3,
                    "context_window": 8192,
                    "health_status": "ok",
                },
                "recommended_candidate": {
                    "model_id": "openrouter:remote",
                    "provider_id": "openrouter",
                    "local": False,
                    "tier": "free_remote",
                    "utility": 0.6,
                    "utility_quality": 0.8,
                    "utility_latency": 0.6,
                    "utility_risk": 0.0,
                    "expected_cost_per_1m": 0.0,
                    "quality_rank": 8,
                    "context_window": 65536,
                    "health_status": "ok",
                },
                "utility_delta": 0.5,
            },
        ) as selector:
            proposal = runtime._build_model_watch_proposal(delta_rows=[{"model_id": "openrouter:remote"}])
        self.assertGreaterEqual(selector.call_count, 1)
        first_call = selector.call_args_list[0]
        self.assertEqual(["ollama:local", "openrouter:remote"], first_call.kwargs.get("candidate_model_ids"))
        self.assertEqual("openrouter:remote", proposal["to_model"])

    def test_model_policy_status_reuses_default_selection_for_target_truth(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        truth = runtime.runtime_truth_service()
        selection = {
            "policy_name": "default",
            "local_first": True,
            "tier_order": ["local", "free_remote", "cheap_remote"],
            "general_remote_cap_per_1m": 6.0,
            "cheap_remote_cap_per_1m": 0.5,
            "current_candidate": {"model_id": "ollama:local", "provider_id": "ollama"},
            "selected_candidate": {"model_id": "ollama:local", "provider_id": "ollama"},
            "recommended_candidate": {"model_id": "ollama:local", "provider_id": "ollama"},
            "switch_recommended": False,
            "decision_reason": "current_already_best",
            "decision_detail": "Current default already matches the best candidate in its tier.",
            "utility_delta": 0.0,
            "min_improvement": 0.08,
            "tier_candidates": {},
            "ordered_candidates": [],
            "rejected_candidates": [],
        }
        with patch.object(truth, "_default_chat_policy_selection", return_value=selection) as selection_reader:
            with patch.object(
                truth,
                "_configured_chat_target_status",
                return_value={
                    "provider": "ollama",
                    "model": "ollama:local",
                    "ready": True,
                    "health_status": "ok",
                    "provider_health_status": "ok",
                },
            ):
                with patch.object(
                    truth,
                    "_live_chat_target_status",
                    return_value={
                        "provider": "ollama",
                        "model": "ollama:local",
                        "ready": True,
                        "health_status": "ok",
                        "provider_health_status": "ok",
                    },
                ):
                    with patch.object(runtime, "safe_mode_target_status", return_value={"enabled": False}):
                        status = truth.model_policy_status()
        self.assertEqual(1, selection_reader.call_count)
        self.assertEqual("ollama:local", status["current_active_model"])

    def test_model_readiness_status_uses_bounded_router_snapshot_access(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        truth = runtime.runtime_truth_service()
        inventory = {
            "active_provider": "ollama",
            "active_model": "ollama:one",
            "configured_provider": "ollama",
            "configured_model": "ollama:one",
            "models": [
                {
                    "model_id": "ollama:one",
                    "provider_id": "ollama",
                    "enabled": True,
                    "available": True,
                    "local": True,
                    "active": True,
                    "lifecycle_state": "ready",
                },
                {
                    "model_id": "openrouter:two",
                    "provider_id": "openrouter",
                    "enabled": True,
                    "available": True,
                    "local": False,
                    "active": False,
                    "lifecycle_state": "ready",
                },
            ],
        }
        provider_status = {
            "configured": True,
            "connection_state": "configured_and_usable",
            "selection_state": "configured_and_usable",
            "policy_blocked": False,
            "auth_required": False,
            "secret_present": True,
            "health_status": "ok",
        }
        with patch.object(truth, "model_inventory_status", return_value=inventory):
            with patch.object(truth, "_policy_flags", return_value={"allow_install_pull": True}):
                with patch.object(truth, "_provider_status_snapshot", return_value=provider_status):
                    with patch.object(
                        truth,
                        "_router_snapshot",
                        return_value={
                            "models": [
                                {"id": "ollama:one", "health": {"status": "ok"}, "available": True},
                                {"id": "openrouter:two", "health": {"status": "ok"}, "available": True},
                            ]
                        },
                    ) as router_snapshot:
                        readiness = truth.model_readiness_status()
        self.assertEqual(3, router_snapshot.call_count)
        self.assertEqual(2, len(readiness["models"]))

    def test_model_scout_reuses_target_truth_for_policy_status(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        truth = runtime.runtime_truth_service()
        candidate_inventory = [
            {
                "model_id": "ollama:local",
                "provider_id": "ollama",
                "local": True,
                "task_types": [],
                "quality_rank": 3,
                "context_window": 8192,
                "expected_cost_per_1m": 0.0,
                "health_status": "ok",
                "usable_now": True,
                "available": True,
                "enabled": True,
                "configured": True,
                "provider_connection_state": "configured_and_usable",
                "provider_selection_state": "configured_and_usable",
                "auth_ok": True,
                "tier": "local",
            }
        ]
        selection = {
            "current_candidate": dict(candidate_inventory[0]),
            "selected_candidate": dict(candidate_inventory[0]),
            "recommended_candidate": dict(candidate_inventory[0]),
            "switch_recommended": False,
            "decision_reason": "current_already_best",
            "decision_detail": "Current default already matches the best candidate in its tier.",
            "tier_candidates": {"local": dict(candidate_inventory[0])},
            "ordered_candidates": [dict(candidate_inventory[0])],
            "rejected_candidates": [],
        }
        readiness = {
            "models": [
                {
                    **dict(candidate_inventory[0]),
                    "active": True,
                    "acquisition_state": "ready_now",
                    "availability_state": "usable_now",
                    "eligibility_state": "usable_now",
                }
            ],
            "active_model": "ollama:local",
            "active_provider": "ollama",
        }
        with patch.object(truth, "model_inventory_status", return_value={"models": []}):
            with patch.object(truth, "model_readiness_status", return_value=readiness):
                with patch.object(truth, "canonical_chat_candidate_inventory", return_value=candidate_inventory):
                    with patch.object(truth, "select_chat_candidates", return_value=selection):
                        with patch.object(
                            truth,
                            "chat_target_truth",
                            return_value={
                                "configured_model": "ollama:local",
                                "configured_provider": "ollama",
                                "effective_model": "ollama:local",
                                "effective_provider": "ollama",
                            },
                        ) as target_truth_reader:
                            truth.model_scout_v2_status(
                                task_request={"task_type": "coding", "requirements": ["chat"], "preferred_local": True},
                                included_role_keys=["best_local", "best_task_coding"],
                            )
        self.assertEqual(1, target_truth_reader.call_count)
        _args, kwargs = target_truth_reader.call_args
        self.assertIn("selection", kwargs)

    def test_model_discovery_proposals_do_not_adopt_into_recommendation_roles(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name, safe_mode_enabled=False))
        runtime.registry_document = _registry_document()
        runtime._save_registry_document(runtime.registry_document)
        runtime._model_discovery_policy_store.upsert_entry(
            "openrouter:remote",
            status="known_good",
            role_hints=["coding"],
            notes="Reviewed for discovery only.",
            reviewed_at="2026-03-30T00:00:00Z",
        )
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            before = truth.model_scout_v2_status(
                task_request={"task_type": "chat", "requirements": ["chat"], "preferred_local": True}
            )
            proposals = truth.model_discovery_proposals_status()
            after = truth.model_scout_v2_status(
                task_request={"task_type": "chat", "requirements": ["chat"], "preferred_local": True}
            )

        self.assertEqual(before.get("recommendation_roles"), after.get("recommendation_roles"))
        proposal_rows = proposals.get("proposals") if isinstance(proposals.get("proposals"), list) else []
        remote_proposal = next(
            (row for row in proposal_rows if isinstance(row, dict) and row.get("model_id") == "openrouter:remote"),
            {},
        )
        self.assertEqual("candidate_good", remote_proposal.get("proposal_kind"))
        self.assertTrue(bool(remote_proposal.get("non_canonical")))
        self.assertTrue(bool(remote_proposal.get("review_required")))
        self.assertEqual("not_adopted", remote_proposal.get("canonical_status"))

    def test_external_model_discovery_snapshot_adds_non_canonical_proposals_without_mutating_selector_truth(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name, safe_mode_enabled=False))
        runtime.registry_document = _registry_document()
        runtime._save_registry_document(runtime.registry_document)
        runtime._model_watch_catalog_path = Path(self.tmpdir.name) / "model_watch_catalog_snapshot.json"
        before_registry = copy.deepcopy(runtime.registry_document)
        before_policy = runtime.model_discovery_policy_entries()
        write_snapshot_atomic(
            runtime._model_watch_catalog_path,
            {
                "provider": "openrouter",
                "source": "openrouter_models",
                "fetched_at": 1774828800,
                "models": [
                    {
                        "id": "openrouter:vendor/cheap-text",
                        "provider_id": "openrouter",
                        "model": "vendor/cheap-text",
                        "context_length": 131072,
                        "modalities": ["text"],
                        "supports_tools": False,
                        "pricing": {
                            "prompt_per_million": 0.1,
                            "completion_per_million": 0.2,
                        },
                    }
                ],
            },
        )
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            before = truth.model_scout_v2_status(
                task_request={"task_type": "chat", "requirements": ["chat"], "preferred_local": True}
            )
            proposals = truth.model_discovery_proposals_status()
            after = truth.model_scout_v2_status(
                task_request={"task_type": "chat", "requirements": ["chat"], "preferred_local": True}
            )

        self.assertEqual(before.get("recommendation_roles"), after.get("recommendation_roles"))
        self.assertEqual(before_registry, runtime.registry_document)
        self.assertEqual(before_policy, runtime.model_discovery_policy_entries())
        self.assertEqual(1, proposals.get("external_model_count"))
        external_sources = proposals.get("external_sources") if isinstance(proposals.get("external_sources"), list) else []
        self.assertTrue(external_sources)
        self.assertTrue(bool((external_sources[0] if isinstance(external_sources[0], dict) else {}).get("ok")))
        proposal_rows = proposals.get("proposals") if isinstance(proposals.get("proposals"), list) else []
        external_proposal = next(
            (
                row
                for row in proposal_rows
                if isinstance(row, dict) and row.get("model_id") == "openrouter:vendor/cheap-text"
            ),
            {},
        )
        self.assertEqual("candidate_good", external_proposal.get("proposal_kind"))
        self.assertIn("cheap_cloud", external_proposal.get("proposed_roles") or [])
        self.assertTrue(bool(external_proposal.get("non_canonical")))
        self.assertTrue(bool(external_proposal.get("review_required")))
        self.assertEqual("not_adopted", external_proposal.get("canonical_status"))
        self.assertEqual("external_openrouter_snapshot", external_proposal.get("source"))

    def test_scout_bridge_bootstrap_and_modelops_agree_on_best_candidate(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        document = _registry_document()
        document["defaults"]["default_provider"] = "ollama"  # type: ignore[index]
        document["defaults"]["default_model"] = "ollama:local"  # type: ignore[index]
        runtime.registry_document = document
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        def _provider_health(provider_id: str | None) -> dict[str, object]:
            return {"status": "down" if str(provider_id or "").strip().lower() == "ollama" else "ok"}

        def _model_health(model_id: str | None) -> dict[str, object]:
            return {"status": "down" if str(model_id or "").strip() == "ollama:local" else "ok"}

        with patch.object(truth, "_provider_health_row", side_effect=_provider_health), patch.object(
            truth,
            "_model_health_row",
            side_effect=_model_health,
        ):
            selection = truth.select_chat_candidates(
                current_model_id="ollama:local",
                allowed_tiers=("local", "free_remote", "cheap_remote", "remote"),
                min_improvement=0.0,
                allow_remote_fallback_override=True,
            )
            scout = truth.model_scout_v2_status()
            bridge_selection = runtime._chat_preflight_bridge().select_chat_candidates(
                policy=runtime.config.default_policy,
                policy_name="default",
                current_model_id="ollama:local",
                allowed_tiers=("local", "free_remote", "cheap_remote", "remote"),
                min_improvement=0.0,
                allow_remote_fallback_override=True,
            )
            bootstrap = runtime._bootstrap_plan()
            ok_recommend, recommend = runtime.llm_models_recommend({"provider": "openrouter", "model_id": "remote"})

        self.assertEqual("openrouter:remote", (selection.get("recommended_candidate") or {}).get("model_id"))
        self.assertEqual("openrouter:remote", (scout.get("recommended_candidate") or {}).get("model_id"))
        self.assertEqual("openrouter:remote", (bridge_selection.get("recommended_candidate") or {}).get("model_id"))
        self.assertEqual("openrouter:remote", (bootstrap.get("selected_candidate") or {}).get("model_id"))
        self.assertTrue(ok_recommend)
        self.assertEqual(
            "openrouter:remote",
            ((recommend.get("envelope") or {}).get("top_for_purpose") or {}).get("canonical_model_id"),
        )
        self.assertEqual(
            scout.get("recommendation_roles"),
            ((recommend.get("envelope") or {}).get("recommendation_roles") if isinstance(recommend.get("envelope"), dict) else {}),
        )

    def test_provider_auth_missing_is_ignored_for_selection(self) -> None:
        os.environ.pop("OPENROUTER_API_KEY", None)
        runtime = AgentRuntime(_config(self.tmpdir.name))
        document = _registry_document()
        document["defaults"]["default_provider"] = "ollama"  # type: ignore[index]
        document["defaults"]["default_model"] = "ollama:local"  # type: ignore[index]
        runtime.registry_document = document
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            side_effect=lambda model_id: {"status": "ok" if str(model_id or "").strip() == "ollama:local" else "unknown"},
        ):
            provider = truth.provider_status("openrouter")
            selection = truth.select_chat_candidates(
                current_model_id="ollama:local",
                allowed_tiers=("local", "free_remote", "cheap_remote", "remote"),
                min_improvement=0.0,
                allow_remote_fallback_override=True,
            )

        self.assertEqual("configured_but_auth_missing", provider.get("connection_state"))
        self.assertEqual("configured_but_auth_missing", provider.get("selection_state"))
        self.assertEqual("ollama:local", (selection.get("selected_candidate") or {}).get("model_id"))
        self.assertNotIn(
            "openrouter:remote",
            [row.get("model_id") for row in selection.get("ordered_candidates", [])],
        )

    def test_provider_unhealthy_is_ignored_for_selection(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        document = _registry_document()
        document["defaults"]["default_provider"] = "ollama"  # type: ignore[index]
        document["defaults"]["default_model"] = "ollama:local"  # type: ignore[index]
        runtime.registry_document = document
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        def _provider_health(provider_id: str | None) -> dict[str, object]:
            return {"status": "down" if str(provider_id or "").strip().lower() == "openrouter" else "ok"}

        with patch.object(truth, "_provider_health_row", side_effect=_provider_health), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            provider = truth.provider_status("openrouter")
            selection = truth.select_chat_candidates(
                current_model_id="ollama:local",
                allowed_tiers=("local", "free_remote", "cheap_remote", "remote"),
                min_improvement=0.0,
                allow_remote_fallback_override=True,
            )

        self.assertEqual("configured_but_unhealthy", provider.get("connection_state"))
        self.assertEqual("configured_but_unhealthy", provider.get("selection_state"))
        self.assertEqual("ollama:local", (selection.get("selected_candidate") or {}).get("model_id"))
        self.assertNotIn(
            "openrouter:remote",
            [row.get("model_id") for row in selection.get("ordered_candidates", [])],
        )

    def test_provider_with_multiple_usable_models_picks_best_allowed_candidate(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "openrouter:baseline": {
                    "provider": "openrouter",
                    "model": "baseline",
                    "capabilities": ["chat"],
                    "quality_rank": 5,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": 0.0, "output_per_million_tokens": 0.0},
                },
                "openrouter:stronger": {
                    "provider": "openrouter",
                    "model": "stronger",
                    "capabilities": ["chat"],
                    "quality_rank": 9,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 0.0, "output_per_million_tokens": 0.0},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "openrouter",
                "default_model": "openrouter:baseline",
                "chat_model": "openrouter:baseline",
                "allow_remote_fallback": True,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            selection = truth.select_chat_candidates(
                current_model_id="openrouter:baseline",
                allowed_tiers=("local", "free_remote", "cheap_remote", "remote"),
                min_improvement=0.0,
                allow_remote_fallback_override=True,
            )

        ordered = [row.get("model_id") for row in selection.get("ordered_candidates", [])]
        self.assertEqual("openrouter:stronger", (selection.get("recommended_candidate") or {}).get("model_id"))
        self.assertEqual(["openrouter:stronger", "openrouter:baseline"], ordered[:2])

    def test_seed_default_registry_rows_do_not_influence_selection(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {},
            "defaults": {
                "routing_mode": "auto",
                "default_provider": None,
                "default_model": None,
                "allow_remote_fallback": True,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            inventory = truth.model_inventory_status()
            readiness = truth.model_readiness_status()
            selection = truth.select_chat_candidates(
                current_model_id=None,
                allowed_tiers=("local", "free_remote", "cheap_remote", "remote"),
                min_improvement=0.0,
                allow_remote_fallback_override=True,
            )

        self.assertEqual([], inventory.get("models"))
        self.assertEqual([], readiness.get("models"))
        self.assertEqual([], selection.get("ordered_candidates"))
        self.assertIsNone(selection.get("selected_candidate"))

    def test_local_selection_prefers_comfortable_fit_over_larger_tight_model(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
            },
            "models": {
                "ollama:qwen2.5:3b-instruct": {
                    "provider": "ollama",
                    "model": "qwen2.5:3b-instruct",
                    "capabilities": ["chat"],
                    "quality_rank": 4,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "ollama:qwen2.5:7b-instruct": {
                    "provider": "ollama",
                    "model": "qwen2.5:7b-instruct",
                    "capabilities": ["chat"],
                    "quality_rank": 7,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen2.5:3b-instruct",
                "chat_model": "ollama:qwen2.5:3b-instruct",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ), patch.object(
            truth,
            "_hardware_capacity_snapshot",
            return_value={"memory_total_bytes": 12 * 1024 * 1024 * 1024, "memory_total_gb": 12.0},
        ):
            selection = truth.select_chat_candidates(
                current_model_id="ollama:qwen2.5:3b-instruct",
                allowed_tiers=("local",),
                task_request={"task_type": "chat", "requirements": ["chat"], "preferred_local": True},
            )

        ordered = selection.get("ordered_candidates") if isinstance(selection.get("ordered_candidates"), list) else []
        self.assertEqual("ollama:qwen2.5:3b-instruct", (ordered[0] if ordered else {}).get("model_id"))
        self.assertEqual("comfortable", (ordered[0] if ordered else {}).get("local_fit_state"))
        self.assertEqual("tight", (ordered[1] if len(ordered) > 1 else {}).get("local_fit_state"))

    def test_model_scout_surfaces_cheap_cloud_and_premium_task_roles(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "openrouter:cheap-chat": {
                    "provider": "openrouter",
                    "model": "cheap-chat",
                    "capabilities": ["chat"],
                    "quality_rank": 7,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": 0.05, "output_per_million_tokens": 0.05},
                },
                "openrouter:premium-coder": {
                    "provider": "openrouter",
                    "model": "premium-coder",
                    "capabilities": ["chat"],
                    "quality_rank": 10,
                    "cost_rank": 3,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 1.0, "output_per_million_tokens": 1.5},
                },
                "openrouter:research-pro": {
                    "provider": "openrouter",
                    "model": "research-pro",
                    "capabilities": ["chat"],
                    "quality_rank": 9,
                    "cost_rank": 3,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 131072,
                    "pricing": {"input_per_million_tokens": 1.2, "output_per_million_tokens": 1.8},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": True,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            chat_status = truth.model_scout_v2_status(
                task_request={"task_type": "chat", "requirements": ["chat"], "preferred_local": True}
            )
            coding_status = truth.model_scout_v2_status(
                task_request={"task_type": "coding", "requirements": ["chat"], "preferred_local": True}
            )
            research_status = truth.model_scout_v2_status(
                task_request={"task_type": "reasoning", "requirements": ["chat", "long_context"], "preferred_local": True}
            )

        chat_roles = chat_status.get("role_candidates") if isinstance(chat_status.get("role_candidates"), dict) else {}
        chat_debug_roles = (
            chat_status.get("recommendation_roles") if isinstance(chat_status.get("recommendation_roles"), dict) else {}
        )
        coding_debug_roles = (
            coding_status.get("recommendation_roles")
            if isinstance(coding_status.get("recommendation_roles"), dict)
            else {}
        )
        research_debug_roles = (
            research_status.get("recommendation_roles")
            if isinstance(research_status.get("recommendation_roles"), dict)
            else {}
        )
        coding_task = coding_status.get("task_recommendation") if isinstance(coding_status.get("task_recommendation"), dict) else {}
        research_task = research_status.get("task_recommendation") if isinstance(research_status.get("task_recommendation"), dict) else {}
        self.assertEqual("openrouter:cheap-chat", (chat_roles.get("cheap_cloud") or {}).get("model_id"))
        self.assertEqual("cheap_remote_value", (chat_roles.get("cheap_cloud") or {}).get("recommendation_basis"))
        self.assertEqual(
            "lower-cost remote option for general use",
            (chat_roles.get("cheap_cloud") or {}).get("recommendation_explanation"),
        )
        self.assertEqual("selected", (chat_debug_roles.get("best_local") or {}).get("state"))
        self.assertEqual("ollama:qwen3.5:4b", (chat_debug_roles.get("best_local") or {}).get("model_id"))
        self.assertEqual("best_local", (chat_debug_roles.get("best_local") or {}).get("recommendation_basis"))
        self.assertEqual("lateral", (((chat_debug_roles.get("best_local") or {}).get("comparison")) or {}).get("state"))
        self.assertEqual(
            "same_as_current",
            (((chat_debug_roles.get("best_local") or {}).get("comparison")) or {}).get("basis"),
        )
        self.assertEqual(
            "not_applicable",
            ((((chat_debug_roles.get("best_local") or {}).get("advisory_actions")) or {}).get("test") or {}).get("state"),
        )
        self.assertEqual(
            "already_current_model",
            ((((chat_debug_roles.get("best_local") or {}).get("advisory_actions")) or {}).get("test") or {}).get("reason_code"),
        )
        self.assertEqual(
            "already_default_model",
            ((((chat_debug_roles.get("best_local") or {}).get("advisory_actions")) or {}).get("make_default") or {}).get("reason_code"),
        )
        self.assertEqual(
            "local_model_no_acquire_needed",
            ((((chat_debug_roles.get("best_local") or {}).get("advisory_actions")) or {}).get("acquire") or {}).get("reason_code"),
        )
        self.assertEqual("selected", (chat_debug_roles.get("cheap_cloud") or {}).get("state"))
        self.assertEqual("openrouter:cheap-chat", (chat_debug_roles.get("cheap_cloud") or {}).get("model_id"))
        self.assertEqual("cheap_remote_value", (chat_debug_roles.get("cheap_cloud") or {}).get("recommendation_basis"))
        self.assertEqual("lateral", (((chat_debug_roles.get("cheap_cloud") or {}).get("comparison")) or {}).get("state"))
        self.assertEqual(
            "lower_cost_alternative",
            (((chat_debug_roles.get("cheap_cloud") or {}).get("comparison")) or {}).get("basis"),
        )
        self.assertEqual(
            "available",
            ((((chat_debug_roles.get("cheap_cloud") or {}).get("advisory_actions")) or {}).get("test") or {}).get("state"),
        )
        self.assertEqual(
            "already_available_remote",
            ((((chat_debug_roles.get("cheap_cloud") or {}).get("advisory_actions")) or {}).get("acquire") or {}).get("reason_code"),
        )
        self.assertEqual("selected", (chat_debug_roles.get("best_task_chat") or {}).get("state"))
        self.assertEqual("ollama:qwen3.5:4b", (chat_debug_roles.get("best_task_chat") or {}).get("model_id"))
        self.assertEqual("best_task_chat", (chat_debug_roles.get("best_task_chat") or {}).get("recommendation_basis"))
        self.assertEqual(
            "lateral",
            (((chat_debug_roles.get("best_task_chat") or {}).get("comparison")) or {}).get("state"),
        )
        self.assertEqual(
            "same_as_current",
            (((chat_debug_roles.get("best_task_chat") or {}).get("comparison")) or {}).get("basis"),
        )
        self.assertEqual("openrouter:premium-coder", coding_task.get("model_id"))
        self.assertEqual("premium_coding_cloud", coding_task.get("role"))
        self.assertEqual("best_task_coding", coding_task.get("recommendation_basis"))
        self.assertEqual("strongest available option currently visible for coding", coding_task.get("recommendation_explanation"))
        self.assertEqual("selected", (coding_debug_roles.get("premium_coding") or {}).get("state"))
        self.assertEqual("openrouter:premium-coder", (coding_debug_roles.get("premium_coding") or {}).get("model_id"))
        self.assertEqual("premium_coding_tier", (coding_debug_roles.get("premium_coding") or {}).get("recommendation_basis"))
        self.assertEqual("upgrade", (((coding_debug_roles.get("premium_coding") or {}).get("comparison")) or {}).get("state"))
        self.assertEqual(
            "higher_premium_role_than_current",
            (((coding_debug_roles.get("premium_coding") or {}).get("comparison")) or {}).get("basis"),
        )
        self.assertEqual(
            "available",
            ((((coding_debug_roles.get("premium_coding") or {}).get("advisory_actions")) or {}).get("test") or {}).get("state"),
        )
        self.assertEqual(
            "available",
            ((((coding_debug_roles.get("premium_coding") or {}).get("advisory_actions")) or {}).get("switch_temporarily") or {}).get("state"),
        )
        self.assertEqual(
            "available",
            ((((coding_debug_roles.get("premium_coding") or {}).get("advisory_actions")) or {}).get("make_default") or {}).get("state"),
        )
        self.assertEqual("selected", (coding_debug_roles.get("best_task_coding") or {}).get("state"))
        self.assertEqual("openrouter:premium-coder", (coding_debug_roles.get("best_task_coding") or {}).get("model_id"))
        self.assertEqual("best_task_coding", (coding_debug_roles.get("best_task_coding") or {}).get("recommendation_basis"))
        self.assertEqual("upgrade", (((coding_debug_roles.get("best_task_coding") or {}).get("comparison")) or {}).get("state"))
        self.assertEqual(
            "stronger_task_fit_than_current",
            (((coding_debug_roles.get("best_task_coding") or {}).get("comparison")) or {}).get("basis"),
        )
        self.assertEqual("openrouter:research-pro", research_task.get("model_id"))
        self.assertEqual("premium_research_cloud", research_task.get("role"))
        self.assertEqual("best_task_research", research_task.get("recommendation_basis"))
        self.assertEqual(
            "best available research option currently visible",
            research_task.get("recommendation_explanation"),
        )
        self.assertEqual("selected", (research_debug_roles.get("premium_research") or {}).get("state"))
        self.assertEqual("openrouter:research-pro", (research_debug_roles.get("premium_research") or {}).get("model_id"))
        self.assertEqual(
            "premium_research_tier",
            (research_debug_roles.get("premium_research") or {}).get("recommendation_basis"),
        )
        self.assertEqual("upgrade", (((research_debug_roles.get("premium_research") or {}).get("comparison")) or {}).get("state"))
        self.assertEqual(
            "larger_context_research_fit_than_current",
            (((research_debug_roles.get("premium_research") or {}).get("comparison")) or {}).get("basis"),
        )
        self.assertEqual(
            "available",
            ((((research_debug_roles.get("premium_research") or {}).get("advisory_actions")) or {}).get("test") or {}).get("state"),
        )
        self.assertEqual("selected", (research_debug_roles.get("best_task_research") or {}).get("state"))
        self.assertEqual(
            "openrouter:research-pro",
            (research_debug_roles.get("best_task_research") or {}).get("model_id"),
        )
        self.assertEqual(
            "best_task_research",
            (research_debug_roles.get("best_task_research") or {}).get("recommendation_basis"),
        )
        self.assertEqual("upgrade", (((research_debug_roles.get("best_task_research") or {}).get("comparison")) or {}).get("state"))
        self.assertEqual(
            "larger_context_research_fit_than_current",
            (((research_debug_roles.get("best_task_research") or {}).get("comparison")) or {}).get("basis"),
        )
        distinct_role_models = {
            str((chat_debug_roles.get("best_local") or {}).get("model_id") or ""),
            str((chat_debug_roles.get("cheap_cloud") or {}).get("model_id") or ""),
            str((coding_debug_roles.get("premium_coding") or {}).get("model_id") or ""),
            str((research_debug_roles.get("premium_research") or {}).get("model_id") or ""),
        }
        self.assertEqual(4, len({item for item in distinct_role_models if item}))

    def test_model_scout_does_not_reuse_cheap_cloud_as_premium_role(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "openrouter:cheap-chat": {
                    "provider": "openrouter",
                    "model": "cheap-chat",
                    "capabilities": ["chat"],
                    "quality_rank": 7,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 0.05, "output_per_million_tokens": 0.05},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            coding_status = truth.model_scout_v2_status(
                task_request={"task_type": "coding", "requirements": ["chat"], "preferred_local": True}
            )
            research_status = truth.model_scout_v2_status(
                task_request={"task_type": "reasoning", "requirements": ["chat", "long_context"], "preferred_local": True}
            )

        coding_roles = coding_status.get("role_candidates") if isinstance(coding_status.get("role_candidates"), dict) else {}
        research_roles = research_status.get("role_candidates") if isinstance(research_status.get("role_candidates"), dict) else {}
        coding_debug_roles = (
            coding_status.get("recommendation_roles")
            if isinstance(coding_status.get("recommendation_roles"), dict)
            else {}
        )
        research_debug_roles = (
            research_status.get("recommendation_roles")
            if isinstance(research_status.get("recommendation_roles"), dict)
            else {}
        )
        coding_task = coding_status.get("task_recommendation") if isinstance(coding_status.get("task_recommendation"), dict) else {}
        research_task = research_status.get("task_recommendation") if isinstance(research_status.get("task_recommendation"), dict) else {}
        self.assertEqual("openrouter:cheap-chat", (coding_roles.get("cheap_cloud") or {}).get("model_id"))
        self.assertIsNone(coding_roles.get("premium_coding_cloud"))
        self.assertEqual("selected", (coding_debug_roles.get("cheap_cloud") or {}).get("state"))
        self.assertEqual("lateral", (((coding_debug_roles.get("cheap_cloud") or {}).get("comparison")) or {}).get("state"))
        self.assertEqual(
            "lower_cost_alternative",
            (((coding_debug_roles.get("cheap_cloud") or {}).get("comparison")) or {}).get("basis"),
        )
        self.assertEqual(
            "no_qualifying_candidate",
            (coding_debug_roles.get("premium_coding") or {}).get("state"),
        )
        self.assertEqual(
            "premium_coding_threshold_unmet",
            (coding_debug_roles.get("premium_coding") or {}).get("reason_code"),
        )
        self.assertNotIn("comparison", coding_debug_roles.get("premium_coding") or {})
        self.assertEqual(
            "no_selected_candidate",
            ((((coding_debug_roles.get("premium_coding") or {}).get("advisory_actions")) or {}).get("test") or {}).get("reason_code"),
        )
        self.assertEqual("best_task_coding", coding_task.get("recommendation_basis"))
        self.assertEqual("current_task", coding_task.get("role"))
        self.assertEqual("strongest available option currently visible for coding", coding_task.get("recommendation_explanation"))
        self.assertEqual("openrouter:cheap-chat", (research_roles.get("cheap_cloud") or {}).get("model_id"))
        self.assertIsNone(research_roles.get("premium_research_cloud"))
        self.assertEqual("selected", (research_debug_roles.get("cheap_cloud") or {}).get("state"))
        self.assertEqual("lateral", (((research_debug_roles.get("cheap_cloud") or {}).get("comparison")) or {}).get("state"))
        self.assertEqual(
            "lower_cost_alternative",
            (((research_debug_roles.get("cheap_cloud") or {}).get("comparison")) or {}).get("basis"),
        )
        self.assertEqual(
            "no_qualifying_candidate",
            (research_debug_roles.get("premium_research") or {}).get("state"),
        )
        self.assertEqual(
            "premium_research_threshold_unmet",
            (research_debug_roles.get("premium_research") or {}).get("reason_code"),
        )
        self.assertNotIn("comparison", research_debug_roles.get("premium_research") or {})
        self.assertEqual(
            "no_selected_candidate",
            ((((research_debug_roles.get("premium_research") or {}).get("advisory_actions")) or {}).get("test") or {}).get("reason_code"),
        )
        self.assertEqual("best_task_research", research_task.get("recommendation_basis"))
        self.assertEqual("current_task", research_task.get("role"))
        self.assertEqual("best available research option currently visible", research_task.get("recommendation_explanation"))

    def test_premium_coding_prefers_task_specialist_over_generic_remote_quality_bump(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "openrouter:generic-pro": {
                    "provider": "openrouter",
                    "model": "generic-pro",
                    "capabilities": ["chat"],
                    "quality_rank": 10,
                    "cost_rank": 3,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 1.3, "output_per_million_tokens": 1.8},
                },
                "openrouter:coder-pro": {
                    "provider": "openrouter",
                    "model": "coder-pro",
                    "capabilities": ["chat"],
                    "task_types": ["coding"],
                    "quality_rank": 9,
                    "cost_rank": 3,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 1.4, "output_per_million_tokens": 1.9},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status(
                task_request={"task_type": "coding", "requirements": ["chat"], "preferred_local": True}
            )

        role_candidates = scout.get("role_candidates") if isinstance(scout.get("role_candidates"), dict) else {}
        recommendation_roles = (
            scout.get("recommendation_roles") if isinstance(scout.get("recommendation_roles"), dict) else {}
        )
        task_recommendation = scout.get("task_recommendation") if isinstance(scout.get("task_recommendation"), dict) else {}
        self.assertEqual("openrouter:coder-pro", (role_candidates.get("premium_coding_cloud") or {}).get("model_id"))
        self.assertEqual("openrouter:coder-pro", (task_recommendation or {}).get("model_id"))
        self.assertEqual("openrouter:coder-pro", (recommendation_roles.get("premium_coding") or {}).get("model_id"))
        self.assertEqual("upgrade", (((recommendation_roles.get("premium_coding") or {}).get("comparison")) or {}).get("state"))

    def test_premium_research_prefers_reasoning_specialist_when_large_context_is_tied(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "openrouter:generic-pro": {
                    "provider": "openrouter",
                    "model": "generic-pro",
                    "capabilities": ["chat"],
                    "quality_rank": 10,
                    "cost_rank": 3,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 131072,
                    "pricing": {"input_per_million_tokens": 1.3, "output_per_million_tokens": 1.8},
                },
                "openrouter:research-pro": {
                    "provider": "openrouter",
                    "model": "research-pro",
                    "capabilities": ["chat"],
                    "task_types": ["reasoning"],
                    "quality_rank": 9,
                    "cost_rank": 3,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 131072,
                    "pricing": {"input_per_million_tokens": 1.4, "output_per_million_tokens": 1.9},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status(
                task_request={"task_type": "reasoning", "requirements": ["chat", "long_context"], "preferred_local": True}
            )

        role_candidates = scout.get("role_candidates") if isinstance(scout.get("role_candidates"), dict) else {}
        recommendation_roles = (
            scout.get("recommendation_roles") if isinstance(scout.get("recommendation_roles"), dict) else {}
        )
        task_recommendation = scout.get("task_recommendation") if isinstance(scout.get("task_recommendation"), dict) else {}
        self.assertEqual("openrouter:research-pro", (role_candidates.get("premium_research_cloud") or {}).get("model_id"))
        self.assertEqual("openrouter:research-pro", (task_recommendation or {}).get("model_id"))
        self.assertEqual("openrouter:research-pro", (recommendation_roles.get("premium_research") or {}).get("model_id"))
        self.assertEqual("upgrade", (((recommendation_roles.get("premium_research") or {}).get("comparison")) or {}).get("state"))

    def test_catalog_synced_known_openrouter_gpt_4_1_mini_enables_premium_coding_role(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.tmpdir.name,
                safe_mode_enabled=True,
                llm_catalog_path=os.path.join(self.tmpdir.name, "llm_catalog.json"),
            )
        )
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        def _fake_http(
            url: str,
            *,
            headers: dict[str, str],
            timeout_seconds: float,
            allowed_hosts: set[str],
        ) -> dict[str, object]:
            _ = timeout_seconds
            self.assertTrue(url.endswith("/models"))
            self.assertIn("openrouter.ai", allowed_hosts)
            self.assertTrue(str(headers.get("Authorization") or "").startswith("Bearer "))
            return {
                "data": [
                    {
                        "id": "openai/gpt-4o-mini",
                        "context_length": 128000,
                        "pricing": {
                            "prompt": "0.0000001500",
                            "completion": "0.0000006000",
                        },
                    },
                    {
                        "id": "openai/gpt-4.1-mini",
                        "context_length": 1047576,
                        "pricing": {
                            "prompt": "0.0000004000",
                            "completion": "0.0000016000",
                        },
                    },
                ]
            }

        catalog_result = fetch_provider_catalog(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "local": False,
                "resolved_headers": {"Authorization": "Bearer sk-test"},
            },
            _fake_http,
        )
        runtime._catalog_store.update_provider_result(
            "openrouter",
            catalog_result,
            now_epoch=123,
        )
        changed, _ = runtime._sync_catalog_into_registry()
        self.assertTrue(changed)
        synced = (runtime.registry_document.get("models") or {}).get("openrouter:openai/gpt-4.1-mini") or {}
        self.assertCountEqual(["coding", "general_chat"], synced.get("task_types") or [])
        self.assertEqual(1047576, synced.get("max_context_tokens"))
        ok_mode, _ = runtime.llm_control_mode_set({"mode": "controlled", "confirm": True, "actor": "test"})
        self.assertTrue(ok_mode)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status(
                task_request={"task_type": "coding", "requirements": ["chat"], "preferred_local": True}
            )

        role_candidates = scout.get("role_candidates") if isinstance(scout.get("role_candidates"), dict) else {}
        recommendation_roles = (
            scout.get("recommendation_roles") if isinstance(scout.get("recommendation_roles"), dict) else {}
        )
        self.assertEqual(
            "openrouter:openai/gpt-4.1-mini",
            (role_candidates.get("premium_coding_cloud") or {}).get("model_id"),
        )
        self.assertEqual(
            "openrouter:openai/gpt-4.1-mini",
            (recommendation_roles.get("premium_coding") or {}).get("model_id"),
        )
        self.assertEqual(
            "premium_coding_tier",
            (recommendation_roles.get("premium_coding") or {}).get("recommendation_basis"),
        )
        self.assertEqual(
            "selected",
            (recommendation_roles.get("premium_coding") or {}).get("state"),
        )
        self.assertNotEqual(
            "openrouter:openai/gpt-4o-mini",
            (recommendation_roles.get("premium_coding") or {}).get("model_id"),
        )

    def test_catalog_synced_known_openrouter_gpt_4_1_enables_premium_research_role(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.tmpdir.name,
                safe_mode_enabled=True,
                llm_catalog_path=os.path.join(self.tmpdir.name, "llm_catalog.json"),
            )
        )
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        def _fake_http(
            url: str,
            *,
            headers: dict[str, str],
            timeout_seconds: float,
            allowed_hosts: set[str],
        ) -> dict[str, object]:
            _ = timeout_seconds
            self.assertTrue(url.endswith("/models"))
            self.assertIn("openrouter.ai", allowed_hosts)
            self.assertTrue(str(headers.get("Authorization") or "").startswith("Bearer "))
            return {
                "data": [
                    {
                        "id": "openai/gpt-4o-mini",
                        "context_length": 128000,
                        "pricing": {
                            "prompt": "0.0000001500",
                            "completion": "0.0000006000",
                        },
                    },
                    {
                        "id": "openai/gpt-4.1",
                        "context_length": 1047576,
                        "pricing": {
                            "prompt": "0.0000010000",
                            "completion": "0.0000040000",
                        },
                    },
                ]
            }

        catalog_result = fetch_provider_catalog(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "local": False,
                "resolved_headers": {"Authorization": "Bearer sk-test"},
            },
            _fake_http,
        )
        runtime._catalog_store.update_provider_result(
            "openrouter",
            catalog_result,
            now_epoch=123,
        )
        changed, _ = runtime._sync_catalog_into_registry()
        self.assertTrue(changed)
        synced = (runtime.registry_document.get("models") or {}).get("openrouter:openai/gpt-4.1") or {}
        self.assertCountEqual(["coding", "general_chat", "reasoning"], synced.get("task_types") or [])
        self.assertEqual(1047576, synced.get("max_context_tokens"))
        ok_mode, _ = runtime.llm_control_mode_set({"mode": "controlled", "confirm": True, "actor": "test"})
        self.assertTrue(ok_mode)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status(
                task_request={"task_type": "reasoning", "requirements": ["chat", "long_context"], "preferred_local": True}
            )

        role_candidates = scout.get("role_candidates") if isinstance(scout.get("role_candidates"), dict) else {}
        recommendation_roles = (
            scout.get("recommendation_roles") if isinstance(scout.get("recommendation_roles"), dict) else {}
        )
        self.assertEqual(
            "openrouter:openai/gpt-4.1",
            (role_candidates.get("premium_research_cloud") or {}).get("model_id"),
        )
        self.assertEqual(
            "openrouter:openai/gpt-4.1",
            (recommendation_roles.get("premium_research") or {}).get("model_id"),
        )
        self.assertEqual(
            "premium_research_tier",
            (recommendation_roles.get("premium_research") or {}).get("recommendation_basis"),
        )
        self.assertEqual(
            "selected",
            (recommendation_roles.get("premium_research") or {}).get("state"),
        )
        self.assertNotEqual(
            "openrouter:openai/gpt-4o-mini",
            (recommendation_roles.get("premium_research") or {}).get("model_id"),
        )

    def test_catalog_synced_known_openrouter_claude_3_5_sonnet_enables_premium_coding_role(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.tmpdir.name,
                safe_mode_enabled=True,
                llm_catalog_path=os.path.join(self.tmpdir.name, "llm_catalog.json"),
            )
        )
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)

        def _fake_http(
            url: str,
            *,
            headers: dict[str, str],
            timeout_seconds: float,
            allowed_hosts: set[str],
        ) -> dict[str, object]:
            _ = timeout_seconds
            self.assertTrue(url.endswith("/models"))
            self.assertIn("openrouter.ai", allowed_hosts)
            self.assertTrue(str(headers.get("Authorization") or "").startswith("Bearer "))
            return {
                "data": [
                    {
                        "id": "openai/gpt-4o-mini",
                        "context_length": 128000,
                        "pricing": {
                            "prompt": "0.0000001500",
                            "completion": "0.0000006000",
                        },
                    },
                    {
                        "id": "anthropic/claude-3.5-sonnet",
                        "context_length": 200000,
                        "pricing": {
                            "prompt": "0.0000010000",
                            "completion": "0.0000040000",
                        },
                    },
                ]
            }

        catalog_result = fetch_provider_catalog(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "local": False,
                "resolved_headers": {"Authorization": "Bearer sk-test"},
            },
            _fake_http,
        )
        runtime._catalog_store.update_provider_result("openrouter", catalog_result, now_epoch=123)
        changed, _ = runtime._sync_catalog_into_registry()
        self.assertTrue(changed)
        synced = (runtime.registry_document.get("models") or {}).get("openrouter:anthropic/claude-3.5-sonnet") or {}
        self.assertCountEqual(["coding", "general_chat", "reasoning"], synced.get("task_types") or [])
        self.assertEqual(200000, synced.get("max_context_tokens"))
        self.assertEqual(8, synced.get("quality_rank"))
        ok_mode, _ = runtime.llm_control_mode_set({"mode": "controlled", "confirm": True, "actor": "test"})
        self.assertTrue(ok_mode)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status(
                task_request={"task_type": "coding", "requirements": ["chat"], "preferred_local": True}
            )

        role_candidates = scout.get("role_candidates") if isinstance(scout.get("role_candidates"), dict) else {}
        recommendation_roles = (
            scout.get("recommendation_roles") if isinstance(scout.get("recommendation_roles"), dict) else {}
        )
        self.assertEqual(
            "openrouter:anthropic/claude-3.5-sonnet",
            (role_candidates.get("premium_coding_cloud") or {}).get("model_id"),
        )
        self.assertEqual(
            "openrouter:anthropic/claude-3.5-sonnet",
            (recommendation_roles.get("premium_coding") or {}).get("model_id"),
        )
        self.assertEqual(
            "premium_coding_tier",
            (recommendation_roles.get("premium_coding") or {}).get("recommendation_basis"),
        )
        self.assertEqual(
            "selected",
            (recommendation_roles.get("premium_coding") or {}).get("state"),
        )

    def test_catalog_synced_known_openrouter_gemini_pro_1_5_enables_premium_research_role(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.tmpdir.name,
                safe_mode_enabled=True,
                llm_catalog_path=os.path.join(self.tmpdir.name, "llm_catalog.json"),
            )
        )
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)

        def _fake_http(
            url: str,
            *,
            headers: dict[str, str],
            timeout_seconds: float,
            allowed_hosts: set[str],
        ) -> dict[str, object]:
            _ = timeout_seconds
            self.assertTrue(url.endswith("/models"))
            self.assertIn("openrouter.ai", allowed_hosts)
            self.assertTrue(str(headers.get("Authorization") or "").startswith("Bearer "))
            return {
                "data": [
                    {
                        "id": "openai/gpt-4o-mini",
                        "context_length": 128000,
                        "pricing": {
                            "prompt": "0.0000001500",
                            "completion": "0.0000006000",
                        },
                    },
                    {
                        "id": "google/gemini-pro-1.5",
                        "context_length": 2097152,
                        "pricing": {
                            "prompt": "0.0000010000",
                            "completion": "0.0000040000",
                        },
                    },
                ]
            }

        catalog_result = fetch_provider_catalog(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "local": False,
                "resolved_headers": {"Authorization": "Bearer sk-test"},
            },
            _fake_http,
        )
        runtime._catalog_store.update_provider_result("openrouter", catalog_result, now_epoch=123)
        changed, _ = runtime._sync_catalog_into_registry()
        self.assertTrue(changed)
        synced = (runtime.registry_document.get("models") or {}).get("openrouter:google/gemini-pro-1.5") or {}
        self.assertCountEqual(["general_chat", "reasoning"], synced.get("task_types") or [])
        self.assertEqual(2097152, synced.get("max_context_tokens"))
        self.assertEqual(8, synced.get("quality_rank"))
        ok_mode, _ = runtime.llm_control_mode_set({"mode": "controlled", "confirm": True, "actor": "test"})
        self.assertTrue(ok_mode)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status(
                task_request={"task_type": "reasoning", "requirements": ["chat", "long_context"], "preferred_local": True}
            )

        role_candidates = scout.get("role_candidates") if isinstance(scout.get("role_candidates"), dict) else {}
        recommendation_roles = (
            scout.get("recommendation_roles") if isinstance(scout.get("recommendation_roles"), dict) else {}
        )
        self.assertEqual(
            "openrouter:google/gemini-pro-1.5",
            (role_candidates.get("premium_research_cloud") or {}).get("model_id"),
        )
        self.assertEqual(
            "openrouter:google/gemini-pro-1.5",
            (recommendation_roles.get("premium_research") or {}).get("model_id"),
        )
        self.assertEqual(
            "premium_research_tier",
            (recommendation_roles.get("premium_research") or {}).get("recommendation_basis"),
        )
        self.assertEqual(
            "selected",
            (recommendation_roles.get("premium_research") or {}).get("state"),
        )

    def test_catalog_synced_source_task_types_and_modalities_survive_into_canonical_inventory(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.tmpdir.name,
                safe_mode_enabled=True,
                llm_catalog_path=os.path.join(self.tmpdir.name, "llm_catalog.json"),
            )
        )
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {},
            "defaults": {
                "routing_mode": "auto",
                "default_provider": None,
                "default_model": None,
                "chat_model": None,
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        runtime._catalog_store.update_provider_result(
            "openrouter",
            {
                "ok": True,
                "provider_id": "openrouter",
                "source": "manual",
                "models": [
                    {
                        "id": "openrouter:vendor/coder-vision",
                        "provider_id": "openrouter",
                        "model": "vendor/coder-vision",
                        "capabilities": ["chat", "tools", "vision"],
                        "task_types": ["coding"],
                        "architecture_modality": "text+image->text",
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                        "max_context_tokens": 65536,
                        "input_cost_per_million_tokens": 1.25,
                        "output_cost_per_million_tokens": 2.5,
                        "source": "manual",
                    }
                ],
                "error_kind": None,
            },
            now_epoch=123,
        )
        changed, _ = runtime._sync_catalog_into_registry()
        self.assertTrue(changed)
        synced = (runtime.registry_document.get("models") or {}).get("openrouter:vendor/coder-vision") or {}
        self.assertEqual(["chat", "tools", "vision"], synced.get("capabilities"))
        self.assertEqual(["coding"], synced.get("task_types"))
        self.assertEqual("text+image->text", synced.get("architecture_modality"))
        self.assertEqual(["image", "text"], sorted(synced.get("input_modalities") or []))
        self.assertEqual(["text"], synced.get("output_modalities"))
        self.assertEqual(65536, synced.get("max_context_tokens"))
        self.assertEqual(1.25, ((synced.get("pricing") or {}).get("input_per_million_tokens")))
        self.assertEqual(2.5, ((synced.get("pricing") or {}).get("output_per_million_tokens")))

        truth = runtime.runtime_truth_service()
        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            inventory = truth.model_inventory_status()

        inventory_rows = inventory.get("models") if isinstance(inventory.get("models"), list) else []
        row = next(
            (
                item
                for item in inventory_rows
                if isinstance(item, dict) and item.get("model_id") == "openrouter:vendor/coder-vision"
            ),
            {},
        )
        self.assertEqual(["chat", "tools", "vision"], row.get("capabilities"))
        self.assertEqual(["coding"], row.get("task_types"))
        self.assertEqual("text+image->text", row.get("architecture_modality"))
        self.assertEqual(["image", "text"], sorted(row.get("input_modalities") or []))
        self.assertEqual(["text"], row.get("output_modalities"))
        self.assertEqual(65536, row.get("context_window"))
        self.assertEqual(1.25, row.get("price_in"))
        self.assertEqual(2.5, row.get("price_out"))

    def test_catalog_synced_remote_vision_metadata_prevents_general_chat_recommendation_drift(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.tmpdir.name,
                safe_mode_enabled=True,
                llm_catalog_path=os.path.join(self.tmpdir.name, "llm_catalog.json"),
            )
        )
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "openrouter:vendor/vision-pro": {
                    "provider": "openrouter",
                    "model": "vendor/vision-pro",
                    "capabilities": ["chat"],
                    "quality_rank": 9,
                    "cost_rank": 3,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 1.0, "output_per_million_tokens": 1.0},
                },
                "openrouter:vendor/general-pro": {
                    "provider": "openrouter",
                    "model": "vendor/general-pro",
                    "capabilities": ["chat"],
                    "quality_rank": 8,
                    "cost_rank": 3,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 1.0, "output_per_million_tokens": 1.0},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "openrouter",
                "default_model": "openrouter:vendor/vision-pro",
                "chat_model": "openrouter:vendor/vision-pro",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)

        def _fake_http(
            url: str,
            *,
            headers: dict[str, str],
            timeout_seconds: float,
            allowed_hosts: set[str],
        ) -> dict[str, object]:
            _ = timeout_seconds
            self.assertTrue(url.endswith("/models"))
            self.assertIn("openrouter.ai", allowed_hosts)
            self.assertTrue(str(headers.get("Authorization") or "").startswith("Bearer "))
            return {
                "data": [
                    {
                        "id": "vendor/vision-pro",
                        "context_length": 65536,
                        "architecture": {
                            "modality": "text+image->text",
                            "input_modalities": ["text", "image"],
                            "output_modalities": ["text"],
                        },
                        "pricing": {
                            "prompt": "0.0000010",
                            "completion": "0.0000010",
                        },
                    },
                    {
                        "id": "vendor/general-pro",
                        "context_length": 65536,
                        "pricing": {
                            "prompt": "0.0000010",
                            "completion": "0.0000010",
                        },
                    },
                ]
            }

        catalog_result = fetch_provider_catalog(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "local": False,
                "resolved_headers": {"Authorization": "Bearer sk-test"},
            },
            _fake_http,
        )
        runtime._catalog_store.update_provider_result("openrouter", catalog_result, now_epoch=123)
        changed, _ = runtime._sync_catalog_into_registry()
        self.assertTrue(changed)
        ok_mode, _ = runtime.llm_control_mode_set({"mode": "controlled", "confirm": True, "actor": "test"})
        self.assertTrue(ok_mode)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status(
                task_request={"task_type": "chat", "requirements": ["chat"], "preferred_local": False}
            )
            selection = truth.select_chat_candidates(
                current_model_id="openrouter:vendor/vision-pro",
                allowed_tiers=("remote",),
                allow_remote_fallback_override=True,
                task_request={"task_type": "chat", "requirements": ["chat"], "preferred_local": False},
                min_improvement=0.0,
            )

        recommendation_roles = (
            scout.get("recommendation_roles") if isinstance(scout.get("recommendation_roles"), dict) else {}
        )
        ordered_ids = [
            str(row.get("model_id") or "")
            for row in (selection.get("ordered_candidates") if isinstance(selection.get("ordered_candidates"), list) else [])
            if isinstance(row, dict)
        ]
        self.assertEqual("openrouter:vendor/general-pro", ordered_ids[0] if ordered_ids else None)
        self.assertIn("openrouter:vendor/vision-pro", ordered_ids)
        self.assertEqual(
            "openrouter:vendor/general-pro",
            (recommendation_roles.get("best_task_chat") or {}).get("model_id"),
        )

    def test_premium_comparison_does_not_overstate_equal_quality_remote_as_upgrade(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "ollama:qwen2.5-coder:9b": {
                    "provider": "ollama",
                    "model": "qwen2.5-coder:9b",
                    "capabilities": ["chat"],
                    "task_types": ["coding"],
                    "quality_rank": 9,
                    "cost_rank": 2,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "openrouter:premium-coder": {
                    "provider": "openrouter",
                    "model": "premium-coder",
                    "capabilities": ["chat"],
                    "task_types": ["coding"],
                    "quality_rank": 9,
                    "cost_rank": 3,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 1.0, "output_per_million_tokens": 1.5},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen2.5-coder:9b",
                "chat_model": "ollama:qwen2.5-coder:9b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status(
                task_request={"task_type": "coding", "requirements": ["chat"], "preferred_local": True}
            )

        recommendation_roles = (
            scout.get("recommendation_roles") if isinstance(scout.get("recommendation_roles"), dict) else {}
        )
        comparison = ((recommendation_roles.get("premium_coding") or {}).get("comparison")) or {}
        self.assertEqual("lateral", comparison.get("state"))
        self.assertEqual("no_meaningful_difference", comparison.get("basis"))

    def test_controlled_mode_can_recommend_remote_candidates_without_remote_fallback(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "openrouter:cheap-chat": {
                    "provider": "openrouter",
                    "model": "cheap-chat",
                    "capabilities": ["chat"],
                    "quality_rank": 8,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 0.0, "output_per_million_tokens": 0.0},
                },
                "openrouter:premium-coder": {
                    "provider": "openrouter",
                    "model": "premium-coder",
                    "capabilities": ["chat"],
                    "quality_rank": 10,
                    "cost_rank": 3,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 1.0, "output_per_million_tokens": 1.5},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()
        seen_path = os.path.join(self.tmpdir.name, "controlled_mode_seen_models.json")
        os.environ["AGENT_MODELOPS_SEEN_MODELS_PATH"] = seen_path

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            policy = truth.model_controller_policy_status()
            scout = truth.model_scout_v2_status(
                task_request={"task_type": "coding", "requirements": ["chat"], "preferred_local": True}
            )
            ok_check, check = runtime.llm_models_check({"purposes": ["chat"]})
            ok_recommend, recommend = runtime.llm_models_recommend(
                {"provider": "openrouter", "model_id": "premium-coder"}
            )

        self.assertTrue(policy.get("allow_remote_recommendation"))
        self.assertFalse(policy.get("allow_remote_fallback"))
        self.assertEqual(
            "ollama:qwen3.5:4b",
            ((scout.get("role_candidates") or {}).get("comfortable_local_default") or {}).get("model_id"),
        )
        self.assertEqual(
            "openrouter:cheap-chat",
            ((scout.get("role_candidates") or {}).get("cheap_cloud") or {}).get("model_id"),
        )
        self.assertEqual(
            "openrouter:premium-coder",
            ((scout.get("role_candidates") or {}).get("premium_coding_cloud") or {}).get("model_id"),
        )
        ordered_ids = [
            str(row.get("model_id") or "")
            for row in (((scout.get("selection") or {}).get("ordered_candidates")) or [])
            if isinstance(row, dict)
        ]
        self.assertIn("openrouter:premium-coder", ordered_ids)
        self.assertIn("openrouter:cheap-chat", ordered_ids)
        self.assertTrue(ok_check)
        envelope = check.get("envelope") if isinstance(check.get("envelope"), dict) else {}
        recommendations = (envelope.get("recommendations_by_purpose") or {}).get("chat") if isinstance(envelope, dict) else []
        self.assertEqual(
            scout.get("recommendation_roles"),
            envelope.get("recommendation_roles"),
        )
        self.assertEqual(
            "upgrade",
            (((envelope.get("recommendation_roles") or {}).get("premium_coding") or {}).get("comparison") or {}).get("state"),
        )
        self.assertTrue(isinstance(recommendations, list))
        self.assertEqual(
            "openrouter:premium-coder",
            (recommendations[0] if recommendations else {}).get("canonical_model_id"),
        )
        self.assertEqual("premium_coding", (recommendations[0] if recommendations else {}).get("source_role"))
        self.assertTrue(bool((recommendations[0] if recommendations else {}).get("compat_only")))
        self.assertEqual(
            (((envelope.get("recommendation_roles") or {}).get("premium_coding")) or {}).get("recommendation_basis"),
            (recommendations[0] if recommendations else {}).get("recommendation_basis"),
        )
        self.assertTrue(ok_recommend)
        recommend_envelope = recommend.get("envelope") if isinstance(recommend.get("envelope"), dict) else {}
        selected_summary = (recommend_envelope.get("selected") or {}) if isinstance(recommend_envelope, dict) else {}
        top_summary = (recommend_envelope.get("top_for_purpose") or {}) if isinstance(recommend_envelope, dict) else {}
        self.assertEqual(
            "openrouter:premium-coder",
            selected_summary.get("canonical_model_id"),
        )
        self.assertEqual("premium_coding", selected_summary.get("source_role"))
        self.assertTrue(bool(selected_summary.get("compat_only")))
        self.assertEqual(
            (((recommend_envelope.get("recommendation_roles") or {}).get("premium_coding")) or {}).get("recommendation_basis"),
            selected_summary.get("recommendation_basis"),
        )
        self.assertEqual(
            ((((recommend_envelope.get("recommendation_roles") or {}).get("premium_coding")) or {}).get("comparison")) or {},
            selected_summary.get("comparison"),
        )
        self.assertEqual(
            (((((recommend_envelope.get("recommendation_roles") or {}).get("premium_coding")) or {}).get("advisory_actions")) or {}),
            selected_summary.get("advisory_actions"),
        )
        self.assertEqual(
            selected_summary.get("canonical_model_id"),
            top_summary.get("canonical_model_id"),
        )
        self.assertEqual(
            envelope.get("recommendation_roles"),
            recommend_envelope.get("recommendation_roles"),
        )
        self.assertEqual(
            "upgrade",
            ((((recommend_envelope.get("recommendation_roles") or {}).get("premium_coding")) or {}).get("comparison") or {}).get("state"),
        )
        self.assertEqual(
            "available",
            (((((recommend_envelope.get("recommendation_roles") or {}).get("premium_coding")) or {}).get("advisory_actions")) or {}).get("test", {}).get("state"),
        )

    def test_chat_recommendation_does_not_prefer_local_vision_specialist(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name, safe_mode_enabled=True))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "ollama:llava:7b": {
                    "provider": "ollama",
                    "model": "llava:7b",
                    "capabilities": ["chat", "vision"],
                    "quality_rank": 9,
                    "cost_rank": 2,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status(
                task_request={"task_type": "chat", "requirements": ["chat"], "preferred_local": True}
            )
            selection = truth.select_chat_candidates(
                current_model_id="ollama:qwen3.5:4b",
                allowed_tiers=("local",),
                task_request={"task_type": "chat", "requirements": ["chat"], "preferred_local": True},
                min_improvement=0.0,
            )

        local_role = (scout.get("role_candidates") or {}).get("comfortable_local_default") if isinstance(scout.get("role_candidates"), dict) else {}
        ordered_ids = [
            str(row.get("model_id") or "")
            for row in (selection.get("ordered_candidates") if isinstance(selection.get("ordered_candidates"), list) else [])
            if isinstance(row, dict)
        ]
        self.assertEqual("ollama:qwen3.5:4b", (local_role or {}).get("model_id"))
        self.assertEqual("ollama:qwen3.5:4b", ordered_ids[0] if ordered_ids else None)
        self.assertIn("ollama:llava:7b", ordered_ids)

    def test_coding_recommendation_does_not_prefer_local_vision_specialist(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name, safe_mode_enabled=True))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "ollama:qwen2.5-coder:7b": {
                    "provider": "ollama",
                    "model": "qwen2.5-coder:7b",
                    "capabilities": ["chat"],
                    "quality_rank": 8,
                    "cost_rank": 2,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "ollama:llava:7b": {
                    "provider": "ollama",
                    "model": "llava:7b",
                    "capabilities": ["chat", "vision"],
                    "quality_rank": 9,
                    "cost_rank": 2,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status(
                task_request={"task_type": "coding", "requirements": ["chat"], "preferred_local": True}
            )
            selection = truth.select_chat_candidates(
                current_model_id="ollama:qwen3.5:4b",
                allowed_tiers=("local",),
                task_request={"task_type": "coding", "requirements": ["chat"], "preferred_local": True},
                min_improvement=0.0,
            )

        local_role = (scout.get("role_candidates") or {}).get("comfortable_local_default") if isinstance(scout.get("role_candidates"), dict) else {}
        ordered_ids = [
            str(row.get("model_id") or "")
            for row in (selection.get("ordered_candidates") if isinstance(selection.get("ordered_candidates"), list) else [])
            if isinstance(row, dict)
        ]
        self.assertEqual("ollama:qwen2.5-coder:7b", (local_role or {}).get("model_id"))
        self.assertEqual("ollama:qwen2.5-coder:7b", ordered_ids[0] if ordered_ids else None)
        self.assertIn("ollama:llava:7b", ordered_ids)

    def test_operator_check_and_scout_share_selection_when_not_ready_model_exists(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "ollama:qwen2.5:7b-instruct": {
                    "provider": "ollama",
                    "model": "qwen2.5:7b-instruct",
                    "capabilities": ["chat"],
                    "quality_rank": 9,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "ollama:llama3:latest": {
                    "provider": "ollama",
                    "model": "llama3:latest",
                    "capabilities": ["chat"],
                    "quality_rank": 10,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 8192,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()
        seen_path = os.path.join(self.tmpdir.name, "modelops_seen_models.json")
        os.environ["AGENT_MODELOPS_SEEN_MODELS_PATH"] = seen_path

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            side_effect=lambda model_id: {
                "status": "down" if str(model_id or "").strip() == "ollama:llama3:latest" else "ok"
            },
        ), patch("agent.api_server.list_models_ollama", return_value=[]):
            scout = truth.model_scout_v2_status()
            selection, _current = runtime._modelops_chat_selection()
            bridge_selection = runtime._chat_preflight_bridge().select_chat_candidates(
                policy=runtime.config.default_policy,
                policy_name="default",
                current_model_id="ollama:qwen3.5:4b",
                allowed_tiers=("local",),
                min_improvement=0.0,
                allow_remote_fallback_override=False,
            )
            ok_check, check = runtime.llm_models_check({"purposes": ["chat"]})

        self.assertTrue(ok_check)
        ordered_ids = [row.get("model_id") for row in selection.get("ordered_candidates", [])]
        self.assertEqual(
            ordered_ids,
            [row.get("model_id") for row in ((scout.get("selection") or {}).get("ordered_candidates") or [])],
        )
        self.assertEqual(
            ordered_ids,
            [row.get("model_id") for row in bridge_selection.get("ordered_candidates", [])],
        )
        self.assertEqual(
            ordered_ids,
            [row.get("model_id") for row in scout.get("candidate_rows", [])],
        )
        self.assertEqual("ollama:qwen2.5:7b-instruct", (scout.get("recommended_candidate") or {}).get("model_id"))
        envelope = check.get("envelope") if isinstance(check.get("envelope"), dict) else {}
        recommendations = (envelope.get("recommendations_by_purpose") or {}).get("chat") if isinstance(envelope, dict) else []
        self.assertTrue(isinstance(recommendations, list))
        self.assertEqual("ollama:qwen2.5:7b-instruct", (recommendations[0] if recommendations else {}).get("canonical_model_id"))
        self.assertEqual("best_local", (recommendations[0] if recommendations else {}).get("source_role"))
        self.assertTrue(bool((recommendations[0] if recommendations else {}).get("compat_only")))
        self.assertNotIn(
            "ollama:llama3:latest",
            [str(row.get("canonical_model_id") or "") for row in recommendations if isinstance(row, dict)],
        )

    def test_runtime_proven_local_readiness_overrides_stale_installed_not_ready_state_for_selection(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "ollama:qwen2.5:7b-instruct": {
                    "provider": "ollama",
                    "model": "qwen2.5:7b-instruct",
                    "capabilities": ["chat"],
                    "quality_rank": 9,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()
        seen_path = os.path.join(self.tmpdir.name, "modelops_seen_models_stale_ready.json")
        os.environ["AGENT_MODELOPS_SEEN_MODELS_PATH"] = seen_path

        state_path = model_manager_state_path_for_runtime(runtime)
        save_model_manager_state(
            state_path,
            {
                "schema_version": 1,
                "targets": {
                    "ollama:qwen2.5:7b-instruct": {
                        "target_key": "ollama:qwen2.5:7b-instruct",
                        "target_type": "model",
                        "provider_id": "ollama",
                        "model_id": "ollama:qwen2.5:7b-instruct",
                        "state": "installed_not_ready",
                        "message": "Old readiness state",
                    }
                },
            },
        )

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ), patch("agent.api_server.list_models_ollama", return_value=[]):
            readiness = truth.model_readiness_status()
            selection = truth.select_chat_candidates(
                current_model_id="ollama:qwen3.5:4b",
                allowed_tiers=("local",),
                min_improvement=0.0,
            )
            scout = truth.model_scout_v2_status()
            ok_check, check = runtime.llm_models_check({"purposes": ["chat"]})

        readiness_row = next(
            row
            for row in readiness.get("models", [])
            if row.get("model_id") == "ollama:qwen2.5:7b-instruct"
        )
        self.assertTrue(bool(readiness_row.get("usable_now")))
        self.assertEqual("ready_now", readiness_row.get("acquisition_state"))
        self.assertEqual("ollama:qwen2.5:7b-instruct", (selection.get("recommended_candidate") or {}).get("model_id"))
        self.assertEqual("ollama:qwen2.5:7b-instruct", (scout.get("recommended_candidate") or {}).get("model_id"))
        envelope = check.get("envelope") if isinstance(check.get("envelope"), dict) else {}
        recommendations = (envelope.get("recommendations_by_purpose") or {}).get("chat") if isinstance(envelope, dict) else []
        self.assertTrue(ok_check)
        self.assertEqual("ollama:qwen2.5:7b-instruct", (recommendations[0] if recommendations else {}).get("canonical_model_id"))
        self.assertEqual("best_local", (recommendations[0] if recommendations else {}).get("source_role"))
        self.assertTrue(bool((recommendations[0] if recommendations else {}).get("compat_only")))

    def test_safe_mode_blocks_remote_selection_and_switch_boundary(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name, safe_mode_enabled=True))
        runtime.registry_document = _registry_document()
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        def _provider_health(provider_id: str | None) -> dict[str, object]:
            return {"status": "down" if str(provider_id or "").strip().lower() == "ollama" else "ok"}

        def _model_health(model_id: str | None) -> dict[str, object]:
            return {"status": "down" if str(model_id or "").strip() == "ollama:local" else "ok"}

        with patch.object(truth, "_provider_health_row", side_effect=_provider_health), patch.object(
            truth,
            "_model_health_row",
            side_effect=_model_health,
        ):
            selection = truth.select_chat_candidates(
                current_model_id="ollama:local",
                allowed_tiers=("local", "free_remote", "cheap_remote", "remote"),
                min_improvement=0.0,
                allow_remote_fallback_override=True,
            )

        self.assertIsNone(selection.get("recommended_candidate"))
        self.assertFalse(bool(selection.get("allow_remote_fallback")))

        ok_default, default_body = runtime.set_default_chat_model("openrouter:remote")
        ok_confirmed, confirmed_body = runtime.set_confirmed_chat_model_target("openrouter:remote")
        ok_temporary, temporary_body = runtime.set_temporary_chat_model_target("openrouter:remote")

        self.assertFalse(ok_default)
        self.assertFalse(ok_confirmed)
        self.assertFalse(ok_temporary)
        self.assertEqual("safe_mode_remote_switch_blocked", default_body.get("error"))
        self.assertEqual("safe_mode_remote_switch_blocked", confirmed_body.get("error"))
        self.assertEqual("safe_mode_remote_switch_blocked", temporary_body.get("error"))

        scout = truth.model_scout_v2_status()
        self.assertFalse(bool((scout.get("policy") or {}).get("allow_remote_recommendation")))
        self.assertIsNone(((scout.get("role_candidates") or {}).get("cheap_cloud")))
        recommendation_roles = (
            scout.get("recommendation_roles") if isinstance(scout.get("recommendation_roles"), dict) else {}
        )
        self.assertEqual("blocked_by_mode", (recommendation_roles.get("cheap_cloud") or {}).get("state"))
        self.assertEqual("safe_mode_remote_block", (recommendation_roles.get("cheap_cloud") or {}).get("reason_code"))
        self.assertNotIn("comparison", recommendation_roles.get("cheap_cloud") or {})
        self.assertEqual(
            "blocked",
            ((((recommendation_roles.get("cheap_cloud") or {}).get("advisory_actions")) or {}).get("test") or {}).get("state"),
        )
        self.assertEqual(
            "safe_mode_remote_block",
            ((((recommendation_roles.get("cheap_cloud") or {}).get("advisory_actions")) or {}).get("test") or {}).get("reason_code"),
        )
        self.assertEqual("blocked_by_mode", (recommendation_roles.get("premium_coding") or {}).get("state"))
        self.assertEqual("safe_mode_remote_block", (recommendation_roles.get("premium_coding") or {}).get("reason_code"))
        self.assertNotIn("comparison", recommendation_roles.get("premium_coding") or {})
        self.assertEqual("blocked_by_mode", (recommendation_roles.get("premium_research") or {}).get("state"))
        self.assertEqual("safe_mode_remote_block", (recommendation_roles.get("premium_research") or {}).get("reason_code"))
        self.assertNotIn("comparison", recommendation_roles.get("premium_research") or {})

        ok_check, check = runtime.llm_models_check({"purposes": ["chat"]})
        self.assertTrue(ok_check)
        recommendations = (
            (((check.get("envelope") or {}).get("recommendations_by_purpose")) or {}).get("chat")
            if isinstance(check.get("envelope"), dict)
            else []
        )
        self.assertTrue(
            all(
                str((row or {}).get("source_role") or "").strip().lower()
                not in {"cheap_cloud", "premium_coding", "premium_research"}
                for row in recommendations
                if isinstance(row, dict)
            )
        )

    def test_llm_models_recommend_surfaces_no_qualifying_candidate_roles_consistently(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "ollama": {"enabled": True, "local": True, "api_key_source": None},
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "quality_rank": 6,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                },
                "openrouter:cheap-chat": {
                    "provider": "openrouter",
                    "model": "cheap-chat",
                    "capabilities": ["chat"],
                    "quality_rank": 7,
                    "cost_rank": 1,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 0.05, "output_per_million_tokens": 0.05},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status()
            ok_recommend, recommend = runtime.llm_models_recommend(
                {"provider": "openrouter", "model_id": "cheap-chat"}
            )

        self.assertTrue(ok_recommend)
        recommend_roles = (
            ((recommend.get("envelope") or {}).get("recommendation_roles"))
            if isinstance(recommend.get("envelope"), dict)
            else {}
        )
        selected_summary = (((recommend.get("envelope") or {}).get("selected")) or {}) if isinstance(recommend.get("envelope"), dict) else {}
        self.assertEqual(scout.get("recommendation_roles"), recommend_roles)
        self.assertEqual(
            "no_qualifying_candidate",
            (recommend_roles.get("premium_coding") or {}).get("state"),
        )
        self.assertEqual(
            "premium_coding_threshold_unmet",
            (recommend_roles.get("premium_coding") or {}).get("reason_code"),
        )
        self.assertEqual(
            "no_selected_candidate",
            ((((recommend_roles.get("premium_coding") or {}).get("advisory_actions")) or {}).get("test") or {}).get("reason_code"),
        )
        self.assertEqual("openrouter:cheap-chat", selected_summary.get("canonical_model_id"))
        self.assertEqual("cheap_cloud", selected_summary.get("source_role"))
        self.assertEqual(
            (recommend_roles.get("cheap_cloud") or {}).get("recommendation_basis"),
            selected_summary.get("recommendation_basis"),
        )
        ok_check, check = runtime.llm_models_check({"purposes": ["chat"]})
        self.assertTrue(ok_check)
        recommendations = (
            (((check.get("envelope") or {}).get("recommendations_by_purpose")) or {}).get("chat")
            if isinstance(check.get("envelope"), dict)
            else []
        )
        self.assertEqual("openrouter:cheap-chat", (recommendations[0] if recommendations else {}).get("canonical_model_id"))
        self.assertEqual("cheap_cloud", (recommendations[0] if recommendations else {}).get("source_role"))

    def test_llm_models_recommend_surfaces_blocked_by_mode_roles_consistently(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name, safe_mode_enabled=True))
        runtime.registry_document = _registry_document()
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()

        with patch.object(truth, "_provider_health_row", return_value={"status": "ok"}), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            scout = truth.model_scout_v2_status()
            ok_recommend, recommend = runtime.llm_models_recommend({"provider": "ollama", "model_id": "local"})

        self.assertTrue(ok_recommend)
        recommend_roles = (
            ((recommend.get("envelope") or {}).get("recommendation_roles"))
            if isinstance(recommend.get("envelope"), dict)
            else {}
        )
        top_summary = (((recommend.get("envelope") or {}).get("top_for_purpose")) or {}) if isinstance(recommend.get("envelope"), dict) else {}
        self.assertEqual(scout.get("recommendation_roles"), recommend_roles)
        self.assertEqual("blocked_by_mode", (recommend_roles.get("cheap_cloud") or {}).get("state"))
        self.assertEqual("safe_mode_remote_block", (recommend_roles.get("cheap_cloud") or {}).get("reason_code"))
        self.assertEqual(
            "blocked",
            ((((recommend_roles.get("cheap_cloud") or {}).get("advisory_actions")) or {}).get("test") or {}).get("state"),
        )
        self.assertEqual("best_task_chat", top_summary.get("source_role"))
        self.assertEqual("ollama:local", top_summary.get("canonical_model_id"))

    def test_safe_mode_clears_remote_explicit_override_from_effective_target(self) -> None:
        runtime = AgentRuntime(
            _config(self.tmpdir.name, safe_mode_enabled=True, safe_mode_chat_model="ollama:local")
        )
        runtime.registry_document = _registry_document()
        runtime._save_registry_document(runtime.registry_document)
        runtime._safe_mode_explicit_chat_target_override = {
            "provider": "openrouter",
            "model": "openrouter:remote",
        }

        def _fake_resolve(
            model_id: str,
            *,
            provider_id: str | None = None,
            require_local: bool | None = None,
            require_available: bool = False,
        ) -> tuple[bool, dict[str, object]]:
            _ = provider_id
            _ = require_available
            candidate = str(model_id or "").strip()
            if candidate == "openrouter:remote":
                if require_local is True:
                    return False, {"ok": False, "error": "switch_target_unavailable"}
                return True, {"ok": True, "provider": "openrouter", "model_id": "openrouter:remote", "local": False}
            if candidate == "ollama:local":
                return True, {"ok": True, "provider": "ollama", "model_id": "ollama:local", "local": True}
            return False, {"ok": False, "error": "switch_target_unavailable"}

        with patch.object(runtime, "_resolve_exact_chat_target_live", side_effect=_fake_resolve):
            status = runtime.safe_mode_target_status()

        self.assertEqual("ollama:local", status.get("effective_model"))
        self.assertEqual("ollama", status.get("effective_provider"))
        self.assertTrue(bool(status.get("effective_local")))
        self.assertFalse(bool(status.get("explicit_override_active")))
        self.assertIsNone(runtime._safe_mode_explicit_chat_target_override)

    def test_controlled_mode_temporary_switch_preserves_default_until_restore(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name, safe_mode_enabled=False))
        document = _registry_document()
        document["models"]["ollama:trial"] = {  # type: ignore[index]
            "provider": "ollama",
            "model": "trial",
            "capabilities": ["chat"],
            "quality_rank": 5,
            "cost_rank": 1,
            "enabled": True,
            "available": True,
            "max_context_tokens": 8192,
            "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
        }
        runtime.registry_document = document
        runtime._save_registry_document(runtime.registry_document)
        runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:local",
                "allow_remote_fallback": False,
            }
        )
        runtime._health_monitor.state = {
            "providers": {
                "ollama": {"status": "ok", "last_checked_at": 123},
                "openrouter": {"status": "down", "last_checked_at": 123},
            },
            "models": {
                "ollama:local": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "ollama:trial": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "openrouter:remote": {"provider_id": "openrouter", "status": "down", "last_checked_at": 123},
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        ok_temporary, temporary_body = runtime.set_temporary_chat_model_target("ollama:trial")
        defaults_after_temporary = runtime.get_defaults()

        self.assertTrue(ok_temporary)
        self.assertEqual("Temporarily using ollama:trial for chat.", temporary_body.get("message"))
        self.assertEqual("ollama:local", ((runtime.registry_document.get("defaults") or {}).get("chat_model")))
        self.assertEqual("ollama:trial", defaults_after_temporary.get("resolved_default_model"))
        self.assertEqual("ollama", defaults_after_temporary.get("default_provider"))
        self.assertEqual({"provider": "ollama", "model": "ollama:trial"}, runtime._temporary_chat_target_override)

        ok_restore, restore_body = runtime.restore_temporary_chat_model_target("ollama:local")
        defaults_after_restore = runtime.get_defaults()

        self.assertTrue(ok_restore)
        self.assertEqual("Now using ollama:local for chat.", restore_body.get("message"))
        self.assertIsNone(runtime._temporary_chat_target_override)
        self.assertEqual("ollama:local", defaults_after_restore.get("resolved_default_model"))
        self.assertEqual("ollama:local", ((runtime.registry_document.get("defaults") or {}).get("chat_model")))

    def test_controlled_mode_blocks_unusable_remote_confirmed_switch(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name, safe_mode_enabled=False))
        runtime.registry_document = _registry_document()
        runtime._save_registry_document(runtime.registry_document)
        runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:local",
                "allow_remote_fallback": False,
            }
        )
        runtime._health_monitor.state = {
            "providers": {
                "ollama": {"status": "ok", "last_checked_at": 123},
                "openrouter": {"status": "down", "last_checked_at": 123},
            },
            "models": {
                "ollama:local": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "openrouter:remote": {"provider_id": "openrouter", "status": "down", "last_checked_at": 123},
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        ok_confirmed, confirmed_body = runtime.set_confirmed_chat_model_target("openrouter:remote")
        current = runtime.runtime_truth_service().current_chat_target_status()

        self.assertFalse(ok_confirmed)
        self.assertEqual("switch_target_not_usable", confirmed_body.get("error"))
        self.assertIn("provider is down", str(confirmed_body.get("message") or ""))
        self.assertEqual("ollama:local", current.get("model"))
        self.assertIsNone(runtime._temporary_chat_target_override)

    def test_controlled_mode_allows_explicit_remote_confirmed_switch_when_remote_fallback_is_disabled(self) -> None:
        runtime = AgentRuntime(_config(self.tmpdir.name, safe_mode_enabled=False))
        runtime.registry_document = _registry_document()
        runtime._save_registry_document(runtime.registry_document)
        runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:local",
                "allow_remote_fallback": False,
            }
        )
        runtime._health_monitor.state = {
            "providers": {
                "ollama": {"status": "ok", "last_checked_at": 123},
                "openrouter": {"status": "ok", "last_checked_at": 123},
            },
            "models": {
                "ollama:local": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "openrouter:remote": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        ok_confirmed, confirmed_body = runtime.set_confirmed_chat_model_target("openrouter:remote")
        current = runtime.runtime_truth_service().current_chat_target_status()

        self.assertTrue(ok_confirmed)
        self.assertEqual("openrouter:remote", confirmed_body.get("model_id"))
        self.assertIn("Now using openrouter:remote for chat.", str(confirmed_body.get("message") or ""))
        self.assertEqual("openrouter:remote", current.get("model"))
        self.assertIsNone(runtime._temporary_chat_target_override)


if __name__ == "__main__":
    unittest.main()

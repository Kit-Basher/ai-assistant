from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.model_watch_hf import (
    build_hf_local_download_proposal,
    hf_watch_state_path_for_runtime,
    load_hf_watch_state,
    scan_hf_watch,
)
from telegram_adapter.bot import maybe_handle_llm_fixit_reply


def _config(registry_path: str, db_path: str, **overrides: object) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
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
        llm_registry_path=registry_path,
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=os.path.join(os.path.dirname(db_path), "usage_stats.json"),
        llm_health_state_path=os.path.join(os.path.dirname(db_path), "llm_health_state.json"),
        llm_automation_enabled=False,
        model_scout_state_path=os.path.join(os.path.dirname(db_path), "model_scout_state.json"),
        autopilot_notify_store_path=os.path.join(os.path.dirname(db_path), "llm_notifications.json"),
    )
    return base.__class__(**{**base.__dict__, **overrides})


class _FakeHFClient:
    def __init__(self, rows: dict[str, dict[str, object]]) -> None:
        self._rows = rows

    def list_models(self, author: str) -> list[dict[str, str]]:
        _ = author
        return []

    def model_info(self, repo_id: str, files_metadata: bool = True) -> dict[str, object]:
        _ = files_metadata
        return self._rows[repo_id]


class TestModelWatchHF(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_hf_scan_state_persists_and_detects_revision_changes(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                model_watch_hf_enabled=True,
                model_watch_hf_allowlist_repos=("acme/model-a",),
            )
        )
        client_v1 = _FakeHFClient(
            {
                "acme/model-a": {
                    "sha": "rev-1",
                    "siblings": [
                        {"rfilename": "README.md", "size": 100},
                        {"rfilename": "model-q4_k_m.gguf", "size": 1024},
                    ],
                }
            }
        )
        first = scan_hf_watch(runtime, client=client_v1, now_epoch=1_700_000_100)
        self.assertTrue(bool(first.get("ok")))
        self.assertEqual(1, int(first.get("scanned_repos") or 0))
        self.assertEqual(1, int(first.get("discovered_count") or 0))
        self.assertEqual("new", str((first.get("updates") or [{}])[0].get("status") or ""))

        second = scan_hf_watch(runtime, client=client_v1, now_epoch=1_700_000_200)
        self.assertTrue(bool(second.get("ok")))
        self.assertEqual(0, int(second.get("discovered_count") or 0))

        client_v2 = _FakeHFClient(
            {
                "acme/model-a": {
                    "sha": "rev-2",
                    "siblings": [
                        {"rfilename": "README.md", "size": 100},
                        {"rfilename": "model-q4_k_m.gguf", "size": 1024},
                    ],
                }
            }
        )
        third = scan_hf_watch(runtime, client=client_v2, now_epoch=1_700_000_300)
        self.assertTrue(bool(third.get("ok")))
        self.assertEqual(1, int(third.get("discovered_count") or 0))
        self.assertEqual("changed", str((third.get("updates") or [{}])[0].get("status") or ""))

        state_path = hf_watch_state_path_for_runtime(runtime)
        persisted = load_hf_watch_state(state_path)
        repos = persisted.get("repos") if isinstance(persisted.get("repos"), dict) else {}
        self.assertIn("acme/model-a", repos)
        row = repos.get("acme/model-a") if isinstance(repos.get("acme/model-a"), dict) else {}
        self.assertEqual("rev-2", row.get("revision"))
        self.assertEqual(1_700_000_100, int(row.get("first_seen_ts") or 0))
        self.assertEqual(1_700_000_300, int(row.get("last_seen_ts") or 0))

    def test_hf_proposal_local_download_and_confirm_gating(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                model_watch_hf_enabled=True,
                model_watch_hf_allowlist_repos=("acme/model-b",),
                model_watch_hf_download_base_path=os.path.join(self.tmpdir.name, "hf_models"),
            )
        )
        scan_payload = {
            "ok": True,
            "enabled": True,
            "scanned_repos": 1,
            "discovered_count": 1,
            "updates": [
                {
                    "status": "new",
                    "repo_id": "acme/model-b",
                    "revision": "abcd1234",
                    "interesting_files": [
                        {"path": "model-q4_k_m.gguf", "size": 2048, "kind": "gguf"},
                    ],
                    "interesting_files_count": 1,
                    "selected_gguf": "model-q4_k_m.gguf",
                    "installability": "installable_ollama",
                    "estimated_total_bytes": 2048,
                    "estimated_total_human": "0.00 GiB",
                    "over_size_limit": False,
                    "recommended_action": "download_install",
                    "meta_hash": "hash-a",
                }
            ],
            "last_run_ts": 1_700_000_400,
            "state_path": os.path.join(self.tmpdir.name, "model_watch_hf_state.json"),
        }
        proposal = build_hf_local_download_proposal(runtime, scan_payload=scan_payload)
        self.assertIsNotNone(proposal)
        proposal = proposal or {}
        self.assertEqual("local_download", str(proposal.get("proposal_type") or ""))
        self.assertEqual("download_install_local", str((proposal.get("choices") or [{}])[0].get("id") or ""))

        with patch("agent.api_server.scan_hf_watch", return_value=scan_payload), patch.object(
            runtime,
            "telegram_status",
            return_value={"state": "running"},
        ), patch.object(
            runtime,
            "_resolve_telegram_target",
            return_value=("token", "123456789"),
        ), patch.object(runtime, "_send_telegram_message", return_value=None):
            ok_scan, body_scan = runtime.model_watch_hf_scan(trigger="manual")

        self.assertTrue(ok_scan)
        self.assertTrue(bool(body_scan.get("proposal_created")))
        wizard_state = runtime._llm_fixit_store.state  # type: ignore[attr-defined]
        self.assertTrue(bool(wizard_state.get("active")))
        self.assertEqual("awaiting_choice", str(wizard_state.get("step") or ""))
        self.assertEqual("local_download", str(wizard_state.get("proposal_type") or ""))

        with patch("agent.api_server.hf_snapshot_download", return_value="/tmp/hf"), patch(
            "agent.api_server.subprocess.run",
            return_value=Mock(returncode=0, stderr="", stdout=""),
        ) as subprocess_mock:
            reply = maybe_handle_llm_fixit_reply(
                llm_fixit_fn=lambda payload: runtime.llm_fixit(payload),
                wizard_store=runtime._llm_fixit_store,  # type: ignore[attr-defined]
                audit_log=None,
                chat_id="123456789",
                text="1",
                log_path=runtime.config.log_path,
            )
            self.assertIsNotNone(reply)
            self.assertIn("Reply YES", str(reply))
            self.assertEqual("awaiting_confirm", str(runtime._llm_fixit_store.state.get("step")))  # type: ignore[attr-defined]
            subprocess_mock.assert_not_called()

        pending_plan = runtime._llm_fixit_store.state.get("pending_plan") if isinstance(runtime._llm_fixit_store.state, dict) else []
        download_step = (
            pending_plan[0]
            if isinstance(pending_plan, list) and pending_plan and isinstance(pending_plan[0], dict)
            else {}
        )
        download_params = download_step.get("params") if isinstance(download_step.get("params"), dict) else {}
        target_dir = str(download_params.get("target_dir") or "").strip()
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
            with open(os.path.join(target_dir, "model-q4_k_m.gguf"), "w", encoding="utf-8") as handle:
                handle.write("fake-gguf")

        with patch("agent.api_server.hf_snapshot_download", return_value="/tmp/hf") as download_mock, patch(
            "agent.api_server.subprocess.run",
            return_value=Mock(returncode=0, stderr="", stdout=""),
        ) as subprocess_mock, patch.object(
            runtime,
            "refresh_models",
            return_value=(True, {"ok": True}),
        ) as refresh_mock:
            ok_apply, body_apply = runtime.llm_fixit({"confirm": True})

        self.assertTrue(ok_apply)
        self.assertTrue(bool(body_apply.get("ok")))
        download_mock.assert_called_once()
        subprocess_mock.assert_called_once()
        refresh_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()

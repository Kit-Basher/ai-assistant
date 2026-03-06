from __future__ import annotations

from collections import deque
import copy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import hashlib
import ipaddress
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from agent.config import Config, load_config
from agent.error_kind import classify_error_kind
from agent.error_response_ux import (
    bad_request_next_question,
    deterministic_error_message,
    friendly_error_message,
)
from agent.fallback_ladder import run_with_fallback
from agent.identity import get_public_identity
from agent.startup_checks import run_startup_checks
from agent.intent.assessment import (
    IntentAssessment,
    IntentCandidate,
    assess_intent_deterministic,
    rebuild_assessment_from_candidates,
)
from agent.intent.clarification import build_clarification_plan, build_thread_integrity_plan
from agent.intent.llm_rerank import rerank_intents_with_llm
from agent.intent.low_confidence import detect_low_confidence
from agent.intent.thread_integrity import detect_thread_drift, normalize_text as normalize_thread_text
from agent.logging_utils import log_event
from agent.model_watch import (
    CatalogDelta,
    ModelWatchStore,
    latest_model_watch_batch,
    map_buzz_leads_to_catalog,
    model_watch_last_run_at,
    normalize_model_watch_state,
    buzz_scan as model_watch_buzz_scan,
    scan_provider_catalogs,
    summarize_model_watch_batch,
)
from agent.model_watch_catalog import (
    build_openrouter_snapshot,
    load_latest_snapshot as load_model_watch_catalog_snapshot,
    snapshot_path_for_config as model_watch_catalog_path_for_config,
    write_snapshot_atomic as write_model_watch_catalog_snapshot,
)
from agent.model_watch_skill import run_watch_once_for_config
from agent.model_watch_hf import (
    build_hf_local_download_proposal,
    hf_snapshot_download,
    hf_status as model_watch_hf_status,
    scan_hf_watch,
)
from agent.response_envelope import failure, ok_result, validate_envelope
from agent.safe_mode_ux import build_safe_mode_paused_message
from agent.model_scout import build_model_scout
from agent.telegram_runner import TelegramRunner
from agent.audit_log import AuditLog, redact as redact_audit_value
from agent.bootstrap.snapshot import collect_bootstrap_snapshot
from agent.llm.action_ledger import ActionLedgerStore
from agent.llm.autopilot_safety import AutopilotSafetyStateStore, detect_autopilot_churn
from agent.llm.autoconfig import apply_autoconfig_plan, build_autoconfig_plan
from agent.llm.capabilities import (
    apply_capabilities_reconcile_plan,
    build_capabilities_reconcile_plan,
    capability_list_from_inference,
    infer_capabilities_from_catalog,
)
from agent.llm.catalog import (
    CatalogStore,
    _http_get_json_with_policy as catalog_http_get_json_with_policy,
    fetch_provider_catalog,
)
from agent.llm.cleanup import apply_registry_cleanup_plan, build_registry_cleanup_plan
from agent.llm.default_model_guard import validate_default_model
from agent.llm.health import HealthProbeSettings, HealthStateStore, LLMHealthMonitor
from agent.llm.hygiene import apply_hygiene_plan, build_hygiene_plan
from agent.llm.notifications import (
    NotificationStore,
    build_notification_from_diff,
    build_notification_from_state_diff,
    sanitize_notification_text,
    should_send,
)
from agent.llm.ollama_endpoints import normalize_ollama_base_urls
from agent.llm.notify_delivery import DeliveryResult, LocalTarget, TelegramTarget
from agent.llm.probes import probe_model, probe_provider
from agent.llm.provider_validation import validate_provider_call_format
from agent.llm.remediation import build_llm_remediation_plan
from agent.llm.registry_txn import RegistrySnapshotStore, apply_with_rollback
from agent.llm.self_heal import (
    apply_self_heal_plan,
    build_drift_report,
    build_self_heal_plan,
)
from agent.llm.support import (
    build_model_diagnosis,
    build_provider_diagnosis,
    sanitize_support_payload,
)
from agent.llm.value_policy import (
    ValuePolicy,
    detect_premium_escalation_triggers,
    normalize_policy,
    rank_candidates_by_utility,
    utility_delta,
)
from agent.llm.registry import RegistryStore
from agent.llm.router import LLMRouter
from agent.llm.types import LLMError, Message, Request
from agent.modelops import ModelOpsExecutor, ModelOpsPlanner, SafeRunner
from agent.modelops.discovery import ModelInfo, list_models_ollama, list_models_openrouter
from agent.modelops.recommend import (
    load_seen_model_ids,
    recommend_models,
    recommendation_to_dict,
    save_seen_model_ids,
)
from agent.ux.llm_fixit_wizard import (
    LLMFixitWizardStore,
    WizardChoice,
    WizardDecision,
    build_plan_for_choice as wizard_build_plan_for_choice,
    confirm_token_for_plan as wizard_confirm_token_for_plan,
    confirm_token_for_plan_rows as wizard_confirm_token_for_plan_rows,
    decision_issue_hash as wizard_decision_issue_hash,
    decision_to_json as wizard_decision_to_json,
    evaluate_wizard_decision,
    failure_streak_threshold_crossed,
    parse_choice_answer as wizard_parse_choice_answer,
    provider_or_model_ok_down_transition,
    render_wizard_prompt,
    summarize_notification_message,
)
from agent.ux.clarify_suggest import (
    build_clarify_message,
    build_suggest_message,
    classify_ambiguity,
    parse_recovery_choice,
    recovery_options,
)
from agent.memory_v2.inject import with_built_context
from agent.memory_v2.ingest import ingest_bootstrap_snapshot
from agent.memory_v2.retrieval import select_memory
from agent.memory_v2.storage import SQLiteMemoryStore
from agent.memory_v2.types import MemoryLevel, MemoryQuery
from agent.packs.manifest import PackManifestError, compute_permissions_hash, load_manifest, normalize_permissions
from agent.packs.policy import is_iface_allowed
from agent.packs.store import PackStore
from agent.orchestrator import classify_authoritative_domain, has_local_observations_block
from agent.perception import analyze_snapshot, collect_snapshot, summarize_inventory
from agent.permissions import PermissionPolicy, PermissionRequest, PermissionStore
from agent.secret_store import SecretStore
from agent.skills_loader import SkillLoader
from memory.db import MemoryDB


_PROVIDER_ID_RE = re.compile(r"^[a-z0-9_-]{2,64}$")
_TELEGRAM_BOT_TOKEN_SECRET_KEY = "telegram:bot_token"
_MEMORY_V2_BOOTSTRAP_COMPLETED_KEY = "memory_v2.bootstrap_completed"
_MEMORY_V2_BOOTSTRAP_COMPLETED_AT_KEY = "memory_v2.bootstrap_completed_at"
_MEMORY_V2_LAST_BOOTSTRAP_TS_KEY = "memory_v2.last_bootstrap_ts"
_MEMORY_V2_GREETED_ONCE_KEY = "memory_v2.greeted_once"
_BOOTSTRAP_GREETING_TEXT = "Hi — I’m here and ready to help. What can I do for you?"
_AUTOPILOT_APPLY_ACTIONS = {
    "llm.autoconfig.apply",
    "llm.hygiene.apply",
    "llm.cleanup.apply",
    "llm.self_heal.apply",
    "llm.capabilities.reconcile.apply",
    "llm.autopilot.bootstrap.apply",
}
_SAFE_MODE_BLOCKED_DEDUPE_SECONDS = 600
_SAFE_MODE_PAUSED_HEALTH_NOTIFY_SECONDS = 21600
_MODEL_SCOUT_PACK_ID = "model_scout"
_MODEL_SCOUT_IFACES = (
    "model_scout.status",
    "model_scout.suggestions",
    "model_scout.run",
    "model_scout.dismiss",
    "model_scout.mark_installed",
)
_HEALTH_STATUSES = {"ok", "degraded", "down", "unknown", "not_applicable"}
_OLLAMA_PULL_ALLOWLIST = (
    "qwen2.5:3b-instruct",
    "qwen2.5:7b-instruct",
)
_OLLAMA_PULL_ALLOWLIST_SET = set(_OLLAMA_PULL_ALLOWLIST)


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _startup_stdout_event(step: str, **fields: Any) -> None:
    payload: dict[str, Any] = {
        "step": str(step or "").strip() or "unknown",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    for key, value in fields.items():
        payload[str(key)] = value
    try:
        print(f"startup.event {json.dumps(payload, ensure_ascii=True, sort_keys=True)}", flush=True)
    except Exception:
        return


def compute_notification_test_policy(runtime: "AgentRuntime") -> dict[str, Any]:
    parsed = urllib.parse.urlparse(str(runtime.listening_url or ""))
    bind_host = str(parsed.hostname or "").strip()
    is_loopback = runtime._host_is_loopback(bind_host)
    explicit = runtime.config.llm_notifications_allow_test
    if explicit is True:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_test_effective": True,
            "allow_reason": "explicit_true",
        }
    if explicit is False:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_test_effective": False,
            "allow_reason": "explicit_false",
        }
    if is_loopback:
        return {
            "bind_host": bind_host,
            "is_loopback": True,
            "allow_test_effective": True,
            "allow_reason": "loopback_auto",
        }
    return {
        "bind_host": bind_host,
        "is_loopback": False,
        "allow_test_effective": False,
        "allow_reason": "permission_required",
    }


def compute_notification_send_policy(runtime: "AgentRuntime") -> dict[str, Any]:
    parsed = urllib.parse.urlparse(str(runtime.listening_url or ""))
    bind_host = str(parsed.hostname or "").strip()
    is_loopback = runtime._host_is_loopback(bind_host)
    explicit = runtime.config.llm_notifications_allow_send
    if explicit is True:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_send_effective": True,
            "allow_reason": "explicit_true",
        }
    if explicit is False:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_send_effective": False,
            "allow_reason": "explicit_false",
        }
    if is_loopback:
        return {
            "bind_host": bind_host,
            "is_loopback": True,
            "allow_send_effective": True,
            "allow_reason": "loopback_auto",
        }
    return {
        "bind_host": bind_host,
        "is_loopback": False,
        "allow_send_effective": False,
        "allow_reason": "permission_required",
    }


def compute_self_heal_apply_policy(runtime: "AgentRuntime") -> dict[str, Any]:
    parsed = urllib.parse.urlparse(str(runtime.listening_url or ""))
    bind_host = str(parsed.hostname or "").strip()
    is_loopback = runtime._host_is_loopback(bind_host)
    explicit = runtime.config.llm_self_heal_allow_apply
    if explicit is True:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_apply_effective": True,
            "allow_reason": "explicit_true",
        }
    if explicit is False:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_apply_effective": False,
            "allow_reason": "explicit_false",
        }
    if is_loopback:
        return {
            "bind_host": bind_host,
            "is_loopback": True,
            "allow_apply_effective": True,
            "allow_reason": "loopback_auto",
        }
    return {
        "bind_host": bind_host,
        "is_loopback": False,
        "allow_apply_effective": False,
        "allow_reason": "permission_required",
    }


def compute_capabilities_reconcile_apply_policy(runtime: "AgentRuntime") -> dict[str, Any]:
    parsed = urllib.parse.urlparse(str(runtime.listening_url or ""))
    bind_host = str(parsed.hostname or "").strip()
    is_loopback = runtime._host_is_loopback(bind_host)
    explicit = runtime.config.llm_capabilities_reconcile_allow_apply
    if explicit is True:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_apply_effective": True,
            "allow_reason": "explicit_true",
        }
    if explicit is False:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_apply_effective": False,
            "allow_reason": "explicit_false",
        }
    if is_loopback:
        return {
            "bind_host": bind_host,
            "is_loopback": True,
            "allow_apply_effective": True,
            "allow_reason": "loopback_auto",
        }
    return {
        "bind_host": bind_host,
        "is_loopback": False,
        "allow_apply_effective": False,
        "allow_reason": "permission_required",
    }


def compute_registry_prune_apply_policy(runtime: "AgentRuntime") -> dict[str, Any]:
    parsed = urllib.parse.urlparse(str(runtime.listening_url or ""))
    bind_host = str(parsed.hostname or "").strip()
    is_loopback = runtime._host_is_loopback(bind_host)
    explicit = runtime.config.llm_registry_prune_allow_apply
    if explicit is True:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_apply_effective": True,
            "allow_reason": "explicit_true",
        }
    if explicit is False:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_apply_effective": False,
            "allow_reason": "explicit_false",
        }
    if is_loopback:
        return {
            "bind_host": bind_host,
            "is_loopback": True,
            "allow_apply_effective": True,
            "allow_reason": "loopback_auto",
        }
    return {
        "bind_host": bind_host,
        "is_loopback": False,
        "allow_apply_effective": False,
        "allow_reason": "permission_required",
    }


def compute_registry_rollback_policy(runtime: "AgentRuntime") -> dict[str, Any]:
    parsed = urllib.parse.urlparse(str(runtime.listening_url or ""))
    bind_host = str(parsed.hostname or "").strip()
    is_loopback = runtime._host_is_loopback(bind_host)
    explicit = runtime.config.llm_registry_rollback_allow
    if explicit is True:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_rollback_effective": True,
            "allow_reason": "explicit_true",
        }
    if explicit is False:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_rollback_effective": False,
            "allow_reason": "explicit_false",
        }
    if is_loopback:
        return {
            "bind_host": bind_host,
            "is_loopback": True,
            "allow_rollback_effective": True,
            "allow_reason": "loopback_auto",
        }
    return {
        "bind_host": bind_host,
        "is_loopback": False,
        "allow_rollback_effective": False,
        "allow_reason": "permission_required",
    }


def compute_autopilot_bootstrap_apply_policy(runtime: "AgentRuntime") -> dict[str, Any]:
    parsed = urllib.parse.urlparse(str(runtime.listening_url or ""))
    bind_host = str(parsed.hostname or "").strip()
    is_loopback = runtime._host_is_loopback(bind_host)
    explicit = runtime.config.llm_autopilot_bootstrap_allow_apply
    if explicit is True:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_apply_effective": True,
            "allow_reason": "explicit_true",
        }
    if explicit is False:
        return {
            "bind_host": bind_host,
            "is_loopback": is_loopback,
            "allow_apply_effective": False,
            "allow_reason": "explicit_false",
        }
    if is_loopback:
        return {
            "bind_host": bind_host,
            "is_loopback": True,
            "allow_apply_effective": True,
            "allow_reason": "loopback_auto",
        }
    return {
        "bind_host": bind_host,
        "is_loopback": False,
        "allow_apply_effective": False,
        "allow_reason": "permission_required",
    }


class AgentRuntime:
    @staticmethod
    def _runtime_state_path(
        config: Config,
        configured_path: str | None,
        filename: str,
    ) -> str | None:
        explicit = str(configured_path or "").strip()
        if explicit:
            return explicit
        # Tests and ephemeral runs often use a temp DB path. Co-locate state there
        # so stale global state cannot bleed across isolated runtimes.
        try:
            db_parent = Path(config.db_path).expanduser().resolve().parent
        except (OSError, RuntimeError, ValueError):
            return None
        db_parent_value = str(db_parent)
        if db_parent_value == "/tmp" or db_parent_value.startswith("/tmp/"):
            return str((db_parent / filename).resolve())
        return None

    def __init__(self, config: Config, *, defer_bootstrap_warmup: bool = False) -> None:
        init_started = time.monotonic()

        def _mark(step: str, **fields: Any) -> None:
            payload: dict[str, Any] = {
                "elapsed_ms": int((time.monotonic() - init_started) * 1000),
            }
            payload.update(fields)
            _startup_stdout_event(step, **payload)

        self.config = config
        self._defer_bootstrap_warmup = bool(defer_bootstrap_warmup)
        self._registry_lock = threading.RLock()
        self.startup_phase = "starting"
        self._startup_warmup_lock = threading.Lock()
        self._startup_warmup_remaining: list[str] = []
        self._startup_warmup_thread: threading.Thread | None = None
        self._startup_warmup_started = False
        self._startup_last_error: str | None = None
        _mark("runtime.init.begin", pid=os.getpid())
        self.secret_store = SecretStore(path=os.getenv("AGENT_SECRET_STORE_PATH", "").strip() or None)
        _mark("runtime.init.secret_store")
        self._telegram_configured_cached = False
        self._telegram_token_source_cached = "none"
        self._refresh_telegram_config_cache()
        self._repo_root = Path(__file__).resolve().parents[1]
        self.started_at = datetime.now(timezone.utc)
        self.started_at_iso = self.started_at.isoformat()
        self.pid = os.getpid()
        self.version = self._read_version()
        self.git_commit = self._read_git_commit()

        registry_path = config.llm_registry_path
        if not registry_path:
            registry_path = str(self._repo_root / "llm_registry.json")
        self.registry_store = RegistryStore(registry_path)
        _mark("runtime.init.registry_store", registry_path=self.registry_store.path)
        self.permission_store = PermissionStore(path=os.getenv("AGENT_PERMISSIONS_PATH", "").strip() or None)
        self.permission_policy = PermissionPolicy()
        self.pack_store = PackStore(self.config.db_path)
        self.audit_log = AuditLog(path=os.getenv("AGENT_AUDIT_LOG_PATH", "").strip() or None)
        self.webui_dist_path = Path(
            os.getenv("AGENT_WEBUI_DIST_PATH", "").strip() or str(self._repo_root / "agent" / "webui" / "dist")
        ).resolve()
        self.webui_dev_proxy = _is_truthy(os.getenv("WEBUI_DEV_PROXY"))
        self.webui_dev_url = os.getenv("WEBUI_DEV_URL", "http://127.0.0.1:1420").strip() or "http://127.0.0.1:1420"
        self.listening_url = self._default_listening_url()

        self.router: LLMRouter | None = None
        self.registry_document: dict[str, Any] = {}
        self.model_scout = build_model_scout(config)
        model_scout_notify_state_path = self._runtime_state_path(
            config,
            os.getenv("AGENT_MODEL_SCOUT_NOTIFY_STATE_PATH", "").strip() or None,
            "model_scout_notify_state.json",
        )
        if model_scout_notify_state_path:
            self._model_scout_notify_state_path = Path(model_scout_notify_state_path).expanduser().resolve()
        else:
            self._model_scout_notify_state_path = (
                Path.home() / ".local" / "share" / "personal-agent" / "model_scout_notify_state.json"
            ).resolve()
        model_watch_state_path = self._runtime_state_path(
            config,
            self.config.model_watch_state_path,
            "model_watch_state.json",
        )
        self._model_watch_store = ModelWatchStore(path=model_watch_state_path)
        self._model_watch_catalog_path = model_watch_catalog_path_for_config(self.config)
        model_watch_hf_state_path = self._runtime_state_path(
            config,
            self.config.model_watch_hf_state_path,
            "model_watch_hf_state.json",
        )
        if model_watch_hf_state_path:
            self._model_watch_hf_state_path = Path(model_watch_hf_state_path).expanduser().resolve()
        else:
            self._model_watch_hf_state_path = (
                Path.home() / ".local" / "share" / "personal-agent" / "model_watch_hf_state.json"
            ).resolve()
        provider_catalog_state_path = self._runtime_state_path(
            config,
            self.config.provider_catalog_state_path,
            "provider_catalog_state.json",
        )
        if provider_catalog_state_path:
            self._provider_catalog_state_path = Path(provider_catalog_state_path).expanduser().resolve()
        else:
            self._provider_catalog_state_path = (
                Path.home() / ".local" / "share" / "personal-agent" / "provider_catalog_state.json"
            ).resolve()
        installer_script = self._repo_root / "agent" / "modelops" / "install_ollama.sh"
        self.modelops_planner = ModelOpsPlanner(installer_script_path=str(installer_script))
        self.modelops_executor = ModelOpsExecutor(
            safe_runner=SafeRunner(str(installer_script)),
            apply_defaults=self._modelops_apply_defaults,
            toggle_enabled=self._modelops_toggle_enabled,
        )
        self._memory_v2_store: SQLiteMemoryStore | None = None
        if bool(self.config.memory_v2_enabled):
            try:
                self._memory_v2_store = SQLiteMemoryStore(self.config.db_path)
            except Exception as exc:  # pragma: no cover - defensive initialization path
                self._memory_v2_store = None
                log_event(
                    self.config.log_path,
                    "memory_v2_init_failed",
                    {
                        "error": exc.__class__.__name__,
                    },
                )

        self._scheduler_stop = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_next_run: dict[str, float] = {}
        self._telegram_runner: TelegramRunner | None = None
        self._health_monitor = LLMHealthMonitor(
            HealthProbeSettings(
                interval_seconds=max(1, int(self.config.llm_health_interval_seconds)),
                max_probes_per_run=max(1, int(self.config.llm_health_max_probes_per_run)),
                probe_timeout_seconds=max(0.1, float(self.config.llm_health_probe_timeout_seconds)),
            ),
            store=HealthStateStore(path=self.config.llm_health_state_path),
            probe_fn=self._probe_llm_candidate,
        )
        _mark("runtime.init.health_monitor")
        self._catalog_store = CatalogStore(path=self.config.llm_catalog_path)
        _mark("runtime.init.catalog_store", catalog_path=self.config.llm_catalog_path)
        snapshots_dir = self._runtime_state_path(
            self.config,
            self.config.llm_registry_snapshots_dir,
            "registry_snapshots",
        )
        self._registry_snapshot_store = RegistrySnapshotStore(
            path=snapshots_dir,
            max_items=max(1, int(self.config.llm_registry_snapshot_max_items)),
        )
        ledger_path = self._runtime_state_path(
            self.config,
            self.config.llm_autopilot_ledger_path,
            "autopilot_action_ledger.json",
        )
        self._action_ledger = ActionLedgerStore(
            path=ledger_path,
            max_items=max(1, int(self.config.llm_autopilot_ledger_max_items)),
        )
        autopilot_state_path = self._runtime_state_path(
            self.config,
            self.config.llm_autopilot_state_path,
            "autopilot_state.json",
        )
        self._autopilot_safety_state = AutopilotSafetyStateStore(
            path=autopilot_state_path,
            max_recent_apply_ids=max(1, int(self.config.llm_autopilot_churn_recent_limit)),
        )
        self._notification_store = NotificationStore(
            path=self.config.autopilot_notify_store_path,
            max_recent=max(50, int(self.config.llm_notifications_max_items)),
            max_items=max(1, int(self.config.llm_notifications_max_items)),
            max_age_days=max(0, int(self.config.llm_notifications_max_age_days)),
            compact=bool(self.config.llm_notifications_compact),
        )
        llm_fixit_state_path = self._runtime_state_path(
            self.config,
            os.getenv("AGENT_LLM_FIXIT_WIZARD_STATE_PATH", "").strip() or None,
            "llm_fixit_wizard_state.json",
        )
        self._llm_fixit_store = LLMFixitWizardStore(path=llm_fixit_state_path)
        self._model_watch_hf_last_status: dict[str, Any] = {
            "enabled": bool(self.config.model_watch_hf_enabled),
            "last_run_ts": None,
            "last_error": None,
            "discovered_count": 0,
            "tracked_repos": 0,
            "state_path": str(self._model_watch_hf_state_path),
        }
        self._safe_mode_last_blocked_reason: str | None = None
        self._safe_mode_last_escalation_reason: str | None = (
            str(self._autopilot_safety_state.status().get("last_churn_reason") or "").strip() or None
        )
        latest_notification_rows = self._notification_store.recent(limit=1)
        latest_notification = latest_notification_rows[0] if latest_notification_rows else {}
        self._last_notify_status: dict[str, Any] = {
            "outcome": str((latest_notification or {}).get("outcome") or "unknown"),
            "reason": str((latest_notification or {}).get("reason") or "unknown"),
            "dedupe_hash": str(
                (latest_notification or {}).get("dedupe_hash")
                or self._notification_store.state.get("last_sent_hash")
                or ""
            ).strip()
            or None,
            "ts": (latest_notification or {}).get("ts") or self._notification_store.state.get("last_sent_ts"),
            "changed_defaults": 0,
            "changed_providers": 0,
            "changed_models": 0,
        }

        self._request_log: deque[dict[str, Any]] = deque(maxlen=100)
        self._clarify_recovery_state: dict[str, Any] = {
            "active": False,
            "source": None,
            "reason": None,
            "choices": [],
            "created_ts": None,
            "expires_ts": None,
        }
        self._premium_override_once = False
        self._premium_override_until_ts: int | None = None
        self._model_watch_last_proposal_evaluation: dict[str, Any] | None = None
        self._ollama_probe_state: dict[str, Any] = {
            "configured_base_url": str(self.config.ollama_base_url or self.config.ollama_host or "").strip(),
            "native_base": str(
                normalize_ollama_base_urls(str(self.config.ollama_base_url or self.config.ollama_host or ""))
                .get("native_base")
                or ""
            ).strip(),
            "openai_base": str(
                normalize_ollama_base_urls(str(self.config.ollama_base_url or self.config.ollama_host or ""))
                .get("openai_base")
                or ""
            ).strip(),
            "native_ok": False,
            "openai_compat_ok": False,
            "last_error_kind": None,
            "last_status_code": None,
            "last_checked_at": None,
        }
        self._memory_v2_bootstrap_status: dict[str, Any] = {"ran": False, "ok": False}
        if self._defer_bootstrap_warmup:
            warmup_tasks = self._build_startup_warmup_tasks()
            with self._startup_warmup_lock:
                self._startup_warmup_remaining = list(warmup_tasks)
            self._memory_v2_bootstrap_status = {"ran": False, "ok": False, "reason": "deferred_until_listening"}
            _mark("runtime.init.warmup_deferred", warmup_remaining=list(warmup_tasks))
            _mark(
                "runtime.init.scheduler_init",
                scheduler_enabled=False,
                deferred=True,
            )
        else:
            self._ensure_native_packs_registered()
            _mark("runtime.init.native_packs")
            self._reload_router()
            _mark("runtime.init.router_reload")
            if (
                bool(self.config.memory_v2_enabled)
                and self._memory_v2_store is not None
                and not self._memory_v2_bootstrap_completed()
            ):
                self._initialize_memory_v2_bootstrap()
                _mark("runtime.init.memory_bootstrap_ran")
            else:
                self._memory_v2_bootstrap_status = {"ran": False, "ok": True, "reason": "not_required_or_completed"}
                _mark("runtime.init.memory_bootstrap_not_required")
            self._router.set_external_health_state(self._health_monitor.state)
            self._start_background_scheduler_if_enabled()
            _mark(
                "runtime.init.scheduler_init",
                scheduler_enabled=bool(self._scheduler_thread is not None),
                model_catalog_refresh="deferred",
            )
        try:
            self._set_model_watch_hf_status_cache(model_watch_hf_status(self))
        except Exception:
            pass
        _mark("runtime.init.complete")

    def _default_listening_url(self) -> str:
        host = os.getenv("AGENT_API_HOST", "127.0.0.1").strip() or "127.0.0.1"
        port = os.getenv("AGENT_API_PORT", "8765").strip() or "8765"
        return f"http://{host}:{port}"

    def set_listening(self, host: str, port: int) -> None:
        self.listening_url = f"http://{host}:{int(port)}"

    @staticmethod
    def _host_is_loopback(host: str | None) -> bool:
        value = str(host or "").strip()
        if not value:
            return False
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].strip()
        if value.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(value).is_loopback
        except ValueError:
            return False

    def _allow_notifications_test_without_permission(self) -> bool:
        policy = compute_notification_test_policy(self)
        return bool(policy.get("allow_test_effective"))

    def _safe_mode_status(self) -> dict[str, Any]:
        return self._autopilot_safety_state.status()

    def _effective_safe_mode(self) -> bool:
        if not bool(self.config.llm_autopilot_safe_mode):
            return False
        return self._autopilot_safety_state.effective_safe_mode(bool(self.config.llm_autopilot_safe_mode))

    def _autopilot_apply_pause_enabled(self) -> bool:
        if not bool(self.config.llm_autopilot_safe_mode):
            return False
        return self._autopilot_safety_state.apply_pause_enabled()

    def _safe_mode_health_payload(self) -> dict[str, Any]:
        state = self._safe_mode_status()
        paused = bool(self._autopilot_apply_pause_enabled())
        reason = (
            str(state.get("safe_mode_reason") or "").strip()
            if paused
            else "not_paused"
        ) or "not_paused"
        last_transition_at = None
        safe_mode_entered_ts_raw = state.get("safe_mode_entered_ts")
        last_churn_event_ts_raw = state.get("last_churn_event_ts")
        try:
            safe_mode_entered_ts = int(safe_mode_entered_ts_raw or 0)
        except (TypeError, ValueError):
            safe_mode_entered_ts = 0
        try:
            last_churn_event_ts = int(last_churn_event_ts_raw or 0)
        except (TypeError, ValueError):
            last_churn_event_ts = 0
        if safe_mode_entered_ts > 0:
            last_transition_at = safe_mode_entered_ts
        elif last_churn_event_ts > 0:
            last_transition_at = last_churn_event_ts
        next_retry = None
        if paused:
            try:
                next_retry_raw = self._scheduler_next_run.get("autoconfig")
                next_retry_value = int(float(next_retry_raw)) if next_retry_raw is not None else 0
            except (TypeError, ValueError):
                next_retry_value = 0
            if next_retry_value > 0:
                next_retry = next_retry_value
        return {
            "paused": paused,
            "reason": reason,
            "cooldown_until": None,
            "next_retry": next_retry,
            "last_transition_at": last_transition_at,
        }

    def _read_version(self) -> str:
        version_file = self._repo_root / "VERSION"
        try:
            if version_file.is_file():
                version = version_file.read_text(encoding="utf-8").strip()
                if version:
                    return version
        except (OSError, UnicodeError):
            pass
        return "unknown"

    def _read_git_commit(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(self._repo_root), "rev-parse", "--short", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                commit = (result.stdout or "").strip()
                if commit:
                    return commit
        except (OSError, subprocess.SubprocessError):
            pass
        return None

    def _ensure_native_packs_registered(self) -> None:
        try:
            loaded_skills = SkillLoader(self.config.skills_path).load_all()
        except Exception as exc:  # pragma: no cover - defensive startup path
            log_event(
                self.config.log_path,
                "pack_seed_failed",
                {"error": exc.__class__.__name__},
            )
            return
        for skill in loaded_skills.values():
            if str(skill.pack_trust or "").strip().lower() != "native":
                continue
            permissions = skill.pack_permissions if isinstance(skill.pack_permissions, dict) else {"ifaces": []}
            self.pack_store.ensure_native_pack(
                pack_id=str(skill.pack_id or skill.name),
                version=str(skill.version or "0.1.0"),
                permissions=permissions,
                manifest_path=skill.pack_manifest_path,
            )
        self.pack_store.ensure_native_pack(
            pack_id=_MODEL_SCOUT_PACK_ID,
            version=str(self.version or "0.1.0"),
            permissions=self._model_scout_pack_permissions(),
            manifest_path=None,
        )

    @staticmethod
    def _model_scout_pack_permissions() -> dict[str, Any]:
        return normalize_permissions({"ifaces": list(_MODEL_SCOUT_IFACES)})

    def _model_scout_pack_gate(self, *, iface: str) -> tuple[bool, dict[str, Any] | None]:
        iface_name = str(iface or "").strip() or "model_scout.run"
        function_name = iface_name.split(".")[-1]
        decision = is_iface_allowed(
            pack_id=_MODEL_SCOUT_PACK_ID,
            iface=iface_name,
            fallback_iface=function_name,
            pack_record=self.pack_store.get_pack(_MODEL_SCOUT_PACK_ID),
            trust="native",
            expected_permissions_hash=compute_permissions_hash(self._model_scout_pack_permissions()),
        )
        if decision.allowed:
            return True, None
        message = (
            f"This skill pack is not allowed to call {function_name}. "
            f"Approve pack {_MODEL_SCOUT_PACK_ID} for {function_name}?"
        )
        return (
            False,
            {
                "ok": False,
                "error": "pack_permission_denied",
                "error_kind": "bad_request",
                "message": message,
                "next_question": f"Approve pack {_MODEL_SCOUT_PACK_ID} for {function_name}?",
                "errors": ["pack_permission_denied"],
                "pack_id": _MODEL_SCOUT_PACK_ID,
                "iface": iface_name,
                "reason": decision.reason,
            },
        )

    @staticmethod
    def _pack_error(
        *,
        error: str,
        message: str,
        error_kind: str = "bad_request",
        next_question: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        body: dict[str, Any] = {
            "ok": False,
            "error": error,
            "error_kind": error_kind,
            "message": message,
            "errors": [error],
        }
        if next_question:
            body["next_question"] = next_question
        return False, body

    @staticmethod
    def _pack_clarification(*, intent: str, message: str, trace_id: str = "packs") -> tuple[bool, dict[str, Any]]:
        envelope = validate_envelope(
            {
                "ok": True,
                "intent": intent,
                "confidence": 0.0,
                "did_work": False,
                "error_kind": "needs_clarification",
                "message": message,
                "next_question": message,
                "actions": [],
                "errors": ["needs_clarification"],
                "trace_id": trace_id,
            }
        )
        return True, {
            "ok": bool(envelope.get("ok", True)),
            "intent": str(envelope.get("intent") or intent),
            "confidence": float(envelope.get("confidence", 0.0)),
            "did_work": bool(envelope.get("did_work", False)),
            "error_kind": str(envelope.get("error_kind") or "needs_clarification"),
            "message": str(envelope.get("message") or message),
            "next_question": envelope.get("next_question"),
            "actions": envelope.get("actions", []),
            "errors": envelope.get("errors", ["needs_clarification"]),
            "trace_id": envelope.get("trace_id") or trace_id,
            "envelope": envelope,
        }

    def list_packs(self) -> dict[str, Any]:
        return {
            "ok": True,
            "packs": self.pack_store.list_packs(),
        }

    def packs_install(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        source = str(payload.get("source") or "").strip()
        pack_dir = str(payload.get("path") or "").strip()
        if not pack_dir:
            if source and source not in {"path", "local"} and "://" not in source:
                pack_dir = source
            elif source in {"path", "local"}:
                pack_dir = str(payload.get("pack_path") or "").strip()
        if not pack_dir:
            return self._pack_error(
                error="bad_request",
                error_kind="bad_request",
                message="pack source path is required",
                next_question="Provide a local pack directory path containing pack.json.",
            )
        if "://" in source and source not in {"path", "local"}:
            return self._pack_error(
                error="bad_request",
                error_kind="bad_request",
                message="only local path installs are supported",
                next_question="Use a local filesystem path for now.",
            )
        manifest_path = os.path.join(pack_dir, "pack.json")
        try:
            manifest = load_manifest(manifest_path)
        except PackManifestError as exc:
            return self._pack_error(
                error="bad_request",
                error_kind="bad_request",
                message=f"invalid pack manifest: {exc}",
                next_question="Fix pack.json and retry install.",
            )
        requested_pack_id = str(payload.get("pack_id") or "").strip()
        if requested_pack_id and requested_pack_id != manifest.pack_id:
            return self._pack_error(
                error="bad_request",
                error_kind="bad_request",
                message="pack_id does not match manifest",
                next_question=f"Use pack_id {manifest.pack_id} or update pack.json.",
            )
        enable = bool(payload.get("enable", False))
        pack_row = self.pack_store.install_pack(manifest, manifest_path=manifest_path, enable=enable)
        log_event(
            self.config.log_path,
            "pack_installed",
            {
                "pack_id": manifest.pack_id,
                "trust": manifest.trust,
                "enabled": bool(pack_row.get("enabled")),
            },
        )
        message = f"Installed pack {manifest.pack_id}."
        if manifest.trust != "native":
            message += f" Approve pack {manifest.pack_id} before running it?"
        return True, {
            "ok": True,
            "pack": pack_row,
            "message": message,
            "requires_approval": manifest.trust != "native",
        }

    def packs_approve(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        pack_id = str(payload.get("pack_id") or "").strip()
        if not pack_id:
            return self._pack_error(
                error="bad_request",
                error_kind="bad_request",
                message="pack_id is required",
                next_question="Which pack_id should be approved?",
            )
        current = self.pack_store.get_pack(pack_id)
        if current is None:
            return self._pack_error(
                error="pack_not_found",
                error_kind="bad_request",
                message=f"pack not found: {pack_id}",
                next_question="Install the pack first, then approve it.",
            )
        if payload.get("approve") is not True:
            return self._pack_clarification(
                intent="packs.approve",
                message=f"Approve pack {pack_id} for its declared permissions?",
            )
        approved = self.pack_store.set_approval_hash(pack_id, str(current.get("permissions_hash") or ""))
        if approved is None:
            return self._pack_error(
                error="pack_not_found",
                error_kind="bad_request",
                message=f"pack not found: {pack_id}",
                next_question="Install the pack first, then approve it.",
            )
        if "enable" in payload:
            approved = self.pack_store.set_enabled(pack_id, bool(payload.get("enable")))
        log_event(
            self.config.log_path,
            "pack_approved",
            {
                "pack_id": pack_id,
                "enabled": bool((approved or {}).get("enabled", False)),
            },
        )
        return True, {
            "ok": True,
            "pack": approved,
            "message": f"Approved permissions for pack {pack_id}.",
        }

    def packs_enable(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        pack_id = str(payload.get("pack_id") or "").strip()
        if not pack_id:
            return self._pack_error(
                error="bad_request",
                error_kind="bad_request",
                message="pack_id is required",
                next_question="Which pack_id should be enabled or disabled?",
            )
        if "enabled" not in payload:
            return self._pack_error(
                error="bad_request",
                error_kind="bad_request",
                message="enabled flag is required",
                next_question='Include {"enabled": true} or {"enabled": false}.',
            )
        updated = self.pack_store.set_enabled(pack_id, bool(payload.get("enabled")))
        if updated is None:
            return self._pack_error(
                error="pack_not_found",
                error_kind="bad_request",
                message=f"pack not found: {pack_id}",
                next_question="Install the pack first, then enable it.",
            )
        log_event(
            self.config.log_path,
            "pack_enabled_updated",
            {
                "pack_id": pack_id,
                "enabled": bool(updated.get("enabled")),
            },
        )
        return True, {
            "ok": True,
            "pack": updated,
            "message": f"Pack {pack_id} {'enabled' if bool(updated.get('enabled')) else 'disabled'}.",
        }

    def _reload_router(self) -> None:
        with self._registry_lock:
            self.registry_document = self.registry_store.read_document()
            registry = self.registry_store.load(self.config)
            self.router = LLMRouter(
                self.config,
                registry=registry,
                log_path=self.config.log_path,
                secret_store=self.secret_store,
            )
            self.router.set_external_health_state(self._health_monitor.state)

    @property
    def _router(self) -> LLMRouter:
        assert self.router is not None
        return self.router

    def _set_startup_phase(self, phase: str) -> None:
        normalized = str(phase or "").strip().lower() or "starting"
        if normalized not in {"starting", "listening", "warming", "ready", "degraded"}:
            normalized = "starting"
        self.startup_phase = normalized
        _startup_stdout_event("runtime.startup_phase", phase=normalized)

    def _warmup_remaining_snapshot(self) -> list[str]:
        with self._startup_warmup_lock:
            return list(self._startup_warmup_remaining)

    def mark_server_listening(self) -> None:
        self._startup_warmup_started = True
        self._set_startup_phase("listening")
        self._start_startup_warmup_thread()

    def _build_startup_warmup_tasks(self) -> list[str]:
        tasks: list[str] = [
            "native_packs",
            "router_reload",
        ]
        if (
            bool(self.config.memory_v2_enabled)
            and self._memory_v2_store is not None
            and not self._memory_v2_bootstrap_completed()
        ):
            tasks.append("memory_bootstrap")
        tasks.append("model_catalog_refresh")
        return tasks

    def _start_startup_warmup_thread(self) -> None:
        pending = self._warmup_remaining_snapshot()
        if not pending:
            self._start_background_scheduler_if_enabled()
            self._set_startup_phase("ready")
            return
        self._set_startup_phase("warming")
        with self._startup_warmup_lock:
            if self._startup_warmup_thread is not None and self._startup_warmup_thread.is_alive():
                return
            self._startup_warmup_thread = threading.Thread(
                target=self._run_startup_warmup,
                name="startup-warmup",
                daemon=True,
            )
            self._startup_warmup_thread.start()

    def _run_startup_warmup_task(self, task: str) -> None:
        if task == "native_packs":
            self._ensure_native_packs_registered()
            return
        if task == "router_reload":
            self._reload_router()
            if self.router is not None:
                self.router.set_external_health_state(self._health_monitor.state)
            return
        if task == "memory_bootstrap":
            if (
                bool(self.config.memory_v2_enabled)
                and self._memory_v2_store is not None
                and not self._memory_v2_bootstrap_completed()
            ):
                self._initialize_memory_v2_bootstrap()
            else:
                _startup_stdout_event("runtime.warmup.task.skip", task=task, reason="not_required")
            return
        if task == "model_catalog_refresh":
            if not bool(self.config.llm_automation_enabled):
                _startup_stdout_event("runtime.warmup.task.skip", task=task, reason="automation_disabled")
                return
            ok, body = self.run_llm_catalog_refresh(trigger="startup_warmup")
            if not ok:
                reason = (
                    str((body or {}).get("error") or "").strip()
                    or str((body or {}).get("message") or "").strip()
                    or "catalog_refresh_failed"
                )
                raise RuntimeError(reason)
            return
        _startup_stdout_event("runtime.warmup.task.skip", task=task, reason="unknown_task")

    def _run_startup_warmup(self) -> None:
        had_failure = False
        while True:
            with self._startup_warmup_lock:
                if not self._startup_warmup_remaining:
                    break
                task = str(self._startup_warmup_remaining[0] or "").strip()
            if not task:
                with self._startup_warmup_lock:
                    if self._startup_warmup_remaining:
                        self._startup_warmup_remaining.pop(0)
                continue
            started = time.monotonic()
            _startup_stdout_event("runtime.warmup.task.start", task=task)
            try:
                self._run_startup_warmup_task(task)
            except Exception as exc:  # pragma: no cover - defensive warmup path
                had_failure = True
                self._startup_last_error = f"{task}:{exc.__class__.__name__}"
                _startup_stdout_event(
                    "runtime.warmup.task.error",
                    task=task,
                    error_type=exc.__class__.__name__,
                    error=str(exc),
                )
            finally:
                with self._startup_warmup_lock:
                    if self._startup_warmup_remaining and self._startup_warmup_remaining[0] == task:
                        self._startup_warmup_remaining.pop(0)
                    else:
                        self._startup_warmup_remaining = [row for row in self._startup_warmup_remaining if row != task]
                _startup_stdout_event(
                    "runtime.warmup.task.done",
                    task=task,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
        if not had_failure:
            try:
                self._start_background_scheduler_if_enabled()
            except Exception as exc:  # pragma: no cover - defensive path
                had_failure = True
                self._startup_last_error = f"scheduler:{exc.__class__.__name__}"
                _startup_stdout_event(
                    "runtime.warmup.task.error",
                    task="scheduler_start",
                    error_type=exc.__class__.__name__,
                    error=str(exc),
                )
        self._set_startup_phase("degraded" if had_failure else "ready")

    def close(self) -> None:
        self.stop_embedded_telegram()
        warmup_thread = self._startup_warmup_thread
        if warmup_thread is not None and warmup_thread.is_alive():
            warmup_thread.join(timeout=1.0)
        self._scheduler_stop.set()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=2.0)
        self.model_scout.close()

    def start_embedded_telegram(
        self,
        *,
        app_factory: Callable[..., Any] | None = None,
        token_resolver: Callable[[], tuple[str | None, str] | str | None] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> bool:
        self._refresh_telegram_config_cache()
        print(f"telegram.embedded: start called pid={self.pid}", flush=True)
        if self._telegram_runner is not None:
            existing_source = "none"
            try:
                existing_source = str(self._telegram_runner.status().get("token_source") or "none")
            except Exception:
                existing_source = "none"
            print(
                f"telegram.embedded: start result=true token_source={existing_source}",
                flush=True,
            )
            return True
        runner = TelegramRunner(
            runtime=self,
            log_path=self.config.log_path,
            audit_log=self.audit_log,
            app_factory=app_factory,
            token_resolver=token_resolver,
            sleep_fn=sleep_fn,
        )
        started = runner.start()
        if started:
            self._telegram_runner = runner
        token_source = "none"
        try:
            token_source = str(runner.status().get("token_source") or "none")
        except Exception:
            token_source = "none"
        print(
            f"telegram.embedded: start result={'true' if started else 'false'} token_source={token_source}",
            flush=True,
        )
        return bool(started)

    def stop_embedded_telegram(self) -> None:
        runner = self._telegram_runner
        self._telegram_runner = None
        if runner is None:
            return
        try:
            runner.stop()
        except Exception:
            pass

    def _start_background_scheduler_if_enabled(self) -> None:
        if not bool(self.config.llm_automation_enabled):
            return
        if self._scheduler_thread is not None:
            return
        now = time.time()
        self._scheduler_next_run = {
            "refresh": now + 5.0,
            "bootstrap": now + 6.0,
            "catalog": now + 8.0,
            "capabilities_reconcile": now + 9.0,
            "health": now + 10.0,
            "hygiene": now + max(30.0, float(self.config.llm_hygiene_interval_seconds)),
            "cleanup": now + max(45.0, float(self.config.llm_hygiene_interval_seconds)),
            "self_heal": now + 15.0,
            "model_scout": now + max(30.0, float(self.config.llm_model_scout_interval_seconds)),
            "autoconfig": now
            + (
                15.0
                if bool(self.config.llm_autoconfig_run_on_startup)
                else max(60.0, float(self.config.llm_autoconfig_interval_seconds))
            ),
        }
        if bool(self.config.model_watch_enabled):
            self._scheduler_next_run["model_watch"] = now + max(
                60.0,
                float(self.config.model_watch_startup_grace_seconds),
            )
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="llm-automation-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

    def _scheduler_loop(
        self,
        *,
        sleep_fn: Callable[[float], None] | None = None,
        stop_event: threading.Event | None = None,
        max_iters: int | None = None,
    ) -> None:
        latest_inventory: dict[str, Any] = {}
        stop = stop_event or self._scheduler_stop
        iterations = 0
        consecutive_failures = 0
        last_failure_ts: int | None = None
        job_error_last_log_at: dict[str, int] = {}

        def _record_job_error(*, job: str, exc: Exception) -> None:
            now_epoch = int(time.time())
            last_logged = int(job_error_last_log_at.get(job) or 0)
            if now_epoch - last_logged < 60:
                return
            job_error_last_log_at[job] = now_epoch
            self._safe_log_event(
                "scheduler_job_error",
                {
                    "job": str(job),
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )

        while True:
            if max_iters is not None and iterations >= int(max_iters):
                break
            if self._scheduler_wait(stop_event=stop, seconds=1.0, sleep_fn=sleep_fn):
                break
            iterations += 1

            try:
                now = time.time()
                cycle_started = False
                cycle_before_state: dict[str, Any] | None = None
                cycle_reasons: set[str] = set()
                cycle_extra_changes: list[str] = []

                def _start_cycle() -> None:
                    nonlocal cycle_started, cycle_before_state
                    if not cycle_started:
                        cycle_before_state = self._autopilot_notify_state_snapshot()
                        cycle_started = True

                def _collect_safe_mode_block(body: dict[str, Any] | None) -> None:
                    payload = body if isinstance(body, dict) else {}
                    reason = str(payload.get("safe_mode_blocked_reason") or "").strip()
                    if not reason:
                        return
                    cycle_reasons.add("safe_mode_blocked")
                    cycle_extra_changes.append(f"Safe mode blocked: {reason}")

                try:
                    if now >= float(self._scheduler_next_run.get("refresh", 0.0)):
                        _start_cycle()
                        ok, body = self.refresh_models({"actor": "scheduler"})
                        if ok and isinstance(body, dict) and isinstance(body.get("inventory"), dict):
                            latest_inventory = body.get("inventory") or {}
                        if ok and bool((body or {}).get("changed")):
                            cycle_reasons.add("refreshed_provider_inventory")
                        self._scheduler_next_run["refresh"] = now + max(
                            30.0,
                            float(self.config.llm_health_interval_seconds),
                        )
                except Exception as exc:
                    _record_job_error(job="refresh", exc=exc)
                    self._scheduler_next_run["refresh"] = now + max(
                        60.0,
                        float(self.config.llm_health_interval_seconds),
                    )

                try:
                    if "bootstrap" in self._scheduler_next_run and now >= float(self._scheduler_next_run.get("bootstrap", 0.0)):
                        _start_cycle()
                        ok, body = self.llm_autopilot_bootstrap(
                            {"actor": "scheduler", "confirm": False},
                            trigger="scheduler",
                        )
                        if ok and bool((body or {}).get("applied")):
                            plan = body.get("plan") if isinstance(body, dict) else {}
                            cycle_reasons.update(self._plan_reasons(plan if isinstance(plan, dict) else {}))
                        _collect_safe_mode_block(body if isinstance(body, dict) else None)
                        self._scheduler_next_run["bootstrap"] = now + max(
                            86400.0,
                            float(self.config.llm_self_heal_interval_seconds),
                        )
                except Exception as exc:
                    _record_job_error(job="bootstrap", exc=exc)
                    self._scheduler_next_run["bootstrap"] = now + max(
                        86400.0,
                        float(self.config.llm_self_heal_interval_seconds),
                    )

                try:
                    if "catalog" in self._scheduler_next_run and now >= float(self._scheduler_next_run.get("catalog", 0.0)):
                        _start_cycle()
                        ok, body = self.run_llm_catalog_refresh(trigger="scheduler")
                        if ok and bool((body or {}).get("changed")):
                            cycle_reasons.add("refreshed_model_catalog")
                            for line in (body or {}).get("notable_changes") or []:
                                text = str(line or "").strip()
                                if text:
                                    cycle_extra_changes.append(text)
                        self._scheduler_next_run["catalog"] = now + max(
                            60.0,
                            float(self.config.llm_catalog_refresh_interval_seconds),
                        )
                except Exception as exc:
                    _record_job_error(job="catalog", exc=exc)
                    self._scheduler_next_run["catalog"] = now + max(
                        300.0,
                        float(self.config.llm_catalog_refresh_interval_seconds),
                    )

                try:
                    if (
                        "capabilities_reconcile" in self._scheduler_next_run
                        and now >= float(self._scheduler_next_run.get("capabilities_reconcile", 0.0))
                    ):
                        _start_cycle()
                        ok, body = self.llm_capabilities_reconcile_apply(
                            {
                                "actor": "scheduler",
                                "confirm": False,
                            },
                            trigger="scheduler",
                        )
                        if ok and bool((body or {}).get("applied")):
                            plan = body.get("plan") if isinstance(body, dict) else {}
                            cycle_reasons.update(self._plan_reasons(plan if isinstance(plan, dict) else {}))
                            changes = (
                                plan.get("changes")
                                if isinstance(plan, dict) and isinstance(plan.get("changes"), list)
                                else []
                            )
                            for row in changes[:3]:
                                if not isinstance(row, dict):
                                    continue
                                model_id = str(row.get("id") or "").strip()
                                field = str(row.get("field") or "").strip()
                                if not model_id or not field:
                                    continue
                                cycle_extra_changes.append(f"Capabilities: {model_id} updated {field}")
                        self._scheduler_next_run["capabilities_reconcile"] = now + max(
                            30.0,
                            float(self.config.llm_health_interval_seconds),
                        )
                except Exception as exc:
                    _record_job_error(job="capabilities_reconcile", exc=exc)
                    self._scheduler_next_run["capabilities_reconcile"] = now + max(
                        120.0,
                        float(self.config.llm_health_interval_seconds),
                    )

                try:
                    if now >= float(self._scheduler_next_run.get("health", 0.0)):
                        _start_cycle()
                        self.run_llm_health(trigger="scheduler")
                        self._scheduler_next_run["health"] = now + max(
                            1.0,
                            float(self.config.llm_health_interval_seconds),
                        )
                except Exception as exc:
                    _record_job_error(job="health", exc=exc)
                    self._scheduler_next_run["health"] = now + max(
                        30.0,
                        float(self.config.llm_health_interval_seconds),
                    )

                try:
                    if now >= float(self._scheduler_next_run.get("hygiene", 0.0)):
                        _start_cycle()
                        ok, body = self.llm_hygiene_apply(
                            {
                                "actor": "scheduler",
                                "confirm": False,
                                "provider_inventory": latest_inventory,
                            }
                        )
                        if ok and bool((body or {}).get("applied")):
                            plan = body.get("plan") if isinstance(body, dict) else {}
                            cycle_reasons.update(self._plan_reasons(plan if isinstance(plan, dict) else {}))
                        _collect_safe_mode_block(body if isinstance(body, dict) else None)
                        self._scheduler_next_run["hygiene"] = now + max(
                            300.0,
                            float(self.config.llm_hygiene_interval_seconds),
                        )
                except Exception as exc:
                    _record_job_error(job="hygiene", exc=exc)
                    self._scheduler_next_run["hygiene"] = now + max(
                        600.0,
                        float(self.config.llm_hygiene_interval_seconds),
                    )

                try:
                    if "cleanup" in self._scheduler_next_run and now >= float(self._scheduler_next_run.get("cleanup", 0.0)):
                        _start_cycle()
                        ok, body = self.llm_cleanup_apply(
                            {
                                "actor": "scheduler",
                                "confirm": False,
                                "provider_failure_streak": self.config.llm_hygiene_provider_failure_streak,
                            },
                            trigger="scheduler",
                        )
                        if ok and bool((body or {}).get("applied")):
                            plan = body.get("plan") if isinstance(body, dict) else {}
                            cycle_reasons.update(self._plan_reasons(plan if isinstance(plan, dict) else {}))
                        _collect_safe_mode_block(body if isinstance(body, dict) else None)
                        self._scheduler_next_run["cleanup"] = now + max(
                            300.0,
                            float(self.config.llm_hygiene_interval_seconds),
                        )
                except Exception as exc:
                    _record_job_error(job="cleanup", exc=exc)
                    self._scheduler_next_run["cleanup"] = now + max(
                        600.0,
                        float(self.config.llm_hygiene_interval_seconds),
                    )

                try:
                    if now >= float(self._scheduler_next_run.get("self_heal", 0.0)):
                        _start_cycle()
                        drift = self._current_drift_report()
                        if bool(drift.get("has_drift")):
                            ok, body = self.llm_self_heal_apply(
                                {
                                    "actor": "scheduler",
                                    "confirm": False,
                                },
                                trigger="scheduler",
                            )
                            if ok and bool((body or {}).get("applied")):
                                plan = body.get("plan") if isinstance(body, dict) else {}
                                cycle_reasons.update(self._plan_reasons(plan if isinstance(plan, dict) else {}))
                            _collect_safe_mode_block(body if isinstance(body, dict) else None)
                        self._scheduler_next_run["self_heal"] = now + max(
                            300.0,
                            float(self.config.llm_self_heal_interval_seconds),
                        )
                except Exception as exc:
                    _record_job_error(job="self_heal", exc=exc)
                    self._scheduler_next_run["self_heal"] = now + max(
                        600.0,
                        float(self.config.llm_self_heal_interval_seconds),
                    )

                try:
                    if now >= float(self._scheduler_next_run.get("autoconfig", 0.0)):
                        _start_cycle()
                        ok, body = self.llm_autoconfig_apply(
                            {
                                "actor": "scheduler",
                                "confirm": False,
                                "disable_auth_failed_providers": True,
                            }
                        )
                        if ok and bool((body or {}).get("applied")):
                            plan = body.get("plan") if isinstance(body, dict) else {}
                            cycle_reasons.update(self._plan_reasons(plan if isinstance(plan, dict) else {}))
                        _collect_safe_mode_block(body if isinstance(body, dict) else None)
                        self._scheduler_next_run["autoconfig"] = now + max(
                            300.0,
                            float(self.config.llm_autoconfig_interval_seconds),
                        )
                except Exception as exc:
                    _record_job_error(job="autoconfig", exc=exc)
                    self._scheduler_next_run["autoconfig"] = now + max(
                        600.0,
                        float(self.config.llm_autoconfig_interval_seconds),
                    )

                try:
                    if now >= float(self._scheduler_next_run.get("model_scout", 0.0)):
                        self.run_model_scout(trigger="scheduler")
                        self._scheduler_next_run["model_scout"] = now + max(
                            60.0,
                            float(self.config.llm_model_scout_interval_seconds),
                        )
                except Exception as exc:
                    _record_job_error(job="model_scout", exc=exc)
                    self._scheduler_next_run["model_scout"] = now + max(
                        300.0,
                        float(self.config.llm_model_scout_interval_seconds),
                    )

                if "model_watch" in self._scheduler_next_run:
                    try:
                        if now >= float(self._scheduler_next_run.get("model_watch", 0.0)):
                            ok_watch, watch_body = self.run_model_watch_once(trigger="scheduler")
                            next_after = int(watch_body.get("next_check_after_seconds") or 0)
                            if next_after <= 0:
                                next_after = int(self.config.model_watch_interval_seconds)
                            self._scheduler_next_run["model_watch"] = now + max(60.0, float(next_after))
                            if not ok_watch:
                                self._safe_log_event(
                                    "model_watch_scheduler_error",
                                    {"error": str(watch_body.get("error") or "run_failed")},
                                )
                    except Exception as exc:
                        _record_job_error(job="model_watch", exc=exc)
                        self._safe_log_event(
                            "model_watch_scheduler_error",
                            {"error": str(exc)},
                        )
                        self._scheduler_next_run["model_watch"] = now + max(
                            300.0,
                            float(self.config.model_watch_interval_seconds),
                        )

                if cycle_started and cycle_before_state is not None:
                    churn = self._evaluate_autopilot_churn(now_epoch=int(now), trigger="scheduler")
                    if bool(churn.get("entered_safe_mode")):
                        cycle_reasons.add("safe_mode_churn_detected")
                        cycle_extra_changes.append(str(churn.get("notification_line") or ""))
                    self._process_scheduler_notification_cycle(
                        before_state=cycle_before_state,
                        after_state=self._autopilot_notify_state_snapshot(),
                        reasons=sorted(cycle_reasons),
                        extra_changes=sorted(
                            {str(item).strip() for item in cycle_extra_changes if str(item).strip()}
                        ),
                        trigger="scheduler",
                    )

                consecutive_failures = 0
                last_failure_ts = None
            except Exception as exc:  # pragma: no cover - fail-safe loop guard
                consecutive_failures += 1
                last_failure_ts = int(time.time())
                if consecutive_failures <= 5:
                    backoff_seconds = 5.0
                else:
                    backoff_seconds = min(60.0, 5.0 * float(consecutive_failures - 4))
                self._safe_log_event(
                    "scheduler_loop_error",
                    {
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "consecutive_failures": int(consecutive_failures),
                        "last_failure_ts": int(last_failure_ts),
                        "backoff_seconds": int(backoff_seconds),
                    },
                )
                if self._scheduler_wait(stop_event=stop, seconds=backoff_seconds, sleep_fn=sleep_fn):
                    break

    def _schedule_autoconfig_soon(self, seconds: float = 10.0) -> None:
        if not bool(self.config.llm_automation_enabled):
            return
        next_at = time.time() + max(1.0, float(seconds))
        current = float(self._scheduler_next_run.get("autoconfig", next_at))
        self._scheduler_next_run["autoconfig"] = min(current, next_at)

    def _probe_provider_cfg(self, provider_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        providers = (
            self.registry_document.get("providers")
            if isinstance(self.registry_document.get("providers"), dict)
            else {}
        )
        provider_payload = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
        if not isinstance(provider_payload, dict) or not provider_payload:
            return None

        headers = self._provider_request_headers(provider_payload)
        defaults = self._ensure_defaults(self.registry_document)
        provider_impl = self._router._providers.get(provider_id)  # type: ignore[attr-defined]
        provider_available: bool | None = None if provider_impl is not None else False

        provider_cfg = {
            "id": provider_id,
            "provider_id": provider_id,
            "provider_type": provider_payload.get("provider_type"),
            "base_url": provider_payload.get("base_url"),
            "chat_path": provider_payload.get("chat_path"),
            "enabled": bool(provider_payload.get("enabled", True)),
            "local": bool(provider_payload.get("local", False)),
            "allow_remote_fallback": bool(defaults.get("allow_remote_fallback", True)),
            "api_key_source": provider_payload.get("api_key_source"),
            "_resolved_api_key_present": bool(self._provider_api_key(provider_payload)),
            "headers": headers,
            "available": provider_available,
        }
        return provider_cfg, provider_payload

    def _probe_llm_provider(self, provider_id: str, timeout_seconds: float) -> dict[str, Any]:
        built = self._probe_provider_cfg(provider_id)
        if built is None:
            return {
                "status": "down",
                "error_kind": "provider_not_found",
                "status_code": None,
                "detail": "provider not found",
                "duration_ms": 0,
            }
        provider_cfg, provider_payload = built
        validation = self._validate_provider_for_probe(
            provider_id,
            provider_payload,
            headers=provider_cfg.get("headers") if isinstance(provider_cfg.get("headers"), dict) else {},
            trigger="scheduler",
        )
        if not bool(validation.get("ok")):
            return {
                "status": "down",
                "error_kind": str(validation.get("error_kind") or "provider_invalid"),
                "status_code": None,
                "detail": str(validation.get("message") or "provider validation failed"),
                "duration_ms": 0,
            }
        result = probe_provider(
            provider_cfg,
            timeout_seconds=float(timeout_seconds),
            http_get_json=self._http_get_json,
        )
        if str(provider_id or "").strip().lower() == "ollama":
            self._ollama_probe_state = {
                "configured_base_url": str(result.get("configured_base_url") or provider_cfg.get("base_url") or "").strip(),
                "native_base": str(result.get("native_base") or "").strip(),
                "openai_base": str(result.get("openai_base") or "").strip(),
                "native_ok": bool(result.get("native_ok", False)),
                "openai_compat_ok": bool(result.get("openai_compat_ok", False)),
                "last_error_kind": str(result.get("last_error_kind") or "").strip().lower() or None,
                "last_status_code": (
                    int(result.get("last_status_code"))
                    if isinstance(result.get("last_status_code"), int)
                    else None
                ),
                "last_checked_at": int(time.time()),
            }
        return result

    def _ensure_fresh_ollama_probe_state(
        self,
        ttl_seconds: int = 30,
        *,
        timeout_seconds: float = 2.0,
    ) -> dict[str, Any]:
        cached = self._ollama_probe_state if isinstance(self._ollama_probe_state, dict) else {}
        if self._probe_provider_cfg("ollama") is None:
            return dict(cached)
        last_checked = cached.get("last_checked_at")
        now = int(time.time())
        if isinstance(last_checked, int) and now - last_checked <= int(ttl_seconds):
            return dict(cached)
        try:
            self._probe_llm_provider("ollama", float(timeout_seconds))
        except Exception:
            return dict(self._ollama_probe_state) if isinstance(self._ollama_probe_state, dict) else {}
        return dict(self._ollama_probe_state) if isinstance(self._ollama_probe_state, dict) else {}

    def _effective_provider_health(
        self,
        provider_id: str,
        snapshot_row_health: dict[str, Any] | None,
    ) -> dict[str, Any]:
        health_payload = dict(snapshot_row_health) if isinstance(snapshot_row_health, dict) else {}
        provider_key = str(provider_id or "").strip().lower()
        if provider_key != "ollama":
            return health_payload

        probe_state = self._ensure_fresh_ollama_probe_state(ttl_seconds=30, timeout_seconds=2.0)
        if bool(probe_state.get("native_ok", False)):
            now_epoch = int(time.time())
            return {
                "status": "ok",
                "last_error_kind": None,
                "status_code": None,
                "last_status_code": None,
                "failure_streak": 0,
                "cooldown_until": None,
                "down_since": None,
                "last_checked_at": now_epoch,
                "last_ts": float(now_epoch),
            }

        if health_payload:
            return health_payload

        fallback_checked = probe_state.get("last_checked_at")
        checked_at = int(fallback_checked) if isinstance(fallback_checked, int) else int(time.time())
        last_status_code = int(probe_state.get("last_status_code")) if isinstance(probe_state.get("last_status_code"), int) else None
        last_error_kind = str(probe_state.get("last_error_kind") or "").strip().lower() or None
        return {
            "status": "down" if last_error_kind else "unknown",
            "last_error_kind": last_error_kind,
            "status_code": last_status_code,
            "last_status_code": last_status_code,
            "failure_streak": 0,
            "cooldown_until": None,
            "down_since": None,
            "last_checked_at": checked_at,
            "last_ts": float(checked_at),
        }

    def _maybe_probe_ollama_on_demand(
        self,
        *,
        timeout_seconds: float = 2.0,
        max_age_seconds: int = 30,
    ) -> None:
        self._ensure_fresh_ollama_probe_state(
            ttl_seconds=int(max_age_seconds),
            timeout_seconds=float(timeout_seconds),
        )

    @staticmethod
    def _health_epoch(value: Any) -> int | None:
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _health_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _health_iso(epoch: int | None) -> str | None:
        if epoch is None or int(epoch) <= 0:
            return None
        try:
            return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            return None

    @staticmethod
    def _health_status(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in _HEALTH_STATUSES else "unknown"

    def _normalize_health_record(
        self,
        raw: dict[str, Any] | None,
        *,
        status_hint: str | None = None,
        now_epoch: int | None = None,
    ) -> dict[str, Any]:
        payload = dict(raw) if isinstance(raw, dict) else {}
        now_value = int(now_epoch) if isinstance(now_epoch, int) and now_epoch > 0 else int(time.time())
        status = self._health_status(payload.get("status") if payload else status_hint)
        last_checked_at = self._health_epoch(payload.get("last_checked_at"))
        last_ts = self._health_float(payload.get("last_ts"))
        if last_checked_at is None:
            inferred_checked = self._health_epoch(last_ts)
            if inferred_checked is not None:
                last_checked_at = inferred_checked
        # Down/degraded/ok records must always carry a check timestamp to avoid
        # ambiguous "down + null timestamps" payloads in user-facing status.
        if status in {"ok", "degraded", "down"} and last_checked_at is None:
            last_checked_at = now_value
        if last_ts is None and last_checked_at is not None:
            last_ts = float(last_checked_at)
        status_code = self._health_epoch(payload.get("status_code"))
        last_status_code = self._health_epoch(payload.get("last_status_code"))
        if last_status_code is None:
            last_status_code = status_code
        cooldown_until = self._health_epoch(payload.get("cooldown_until"))
        down_since = self._health_epoch(payload.get("down_since"))
        if status == "down" and down_since is None and last_checked_at is not None:
            down_since = last_checked_at
        last_error_kind = str(payload.get("last_error_kind") or "").strip().lower() or None
        try:
            successes = max(0, int(payload.get("successes") or 0))
        except (TypeError, ValueError):
            successes = 0
        try:
            failures = max(0, int(payload.get("failures") or 0))
        except (TypeError, ValueError):
            failures = 0
        try:
            failure_streak = max(0, int(payload.get("failure_streak") or 0))
        except (TypeError, ValueError):
            failure_streak = 0
        if status == "ok":
            last_error_kind = None
            status_code = None
            last_status_code = None
            cooldown_until = None
            down_since = None
            failure_streak = 0
        return {
            "status": status,
            "last_checked_at": last_checked_at,
            "last_checked_at_iso": self._health_iso(last_checked_at),
            "last_error_kind": last_error_kind,
            "status_code": status_code,
            "last_status_code": last_status_code,
            "cooldown_until": cooldown_until,
            "cooldown_until_iso": self._health_iso(cooldown_until),
            "down_since": down_since,
            "down_since_iso": self._health_iso(down_since),
            "successes": successes,
            "failures": failures,
            "failure_streak": failure_streak,
            "last_ts": last_ts,
            "last_ts_iso": self._health_iso(self._health_epoch(last_ts)),
        }

    def _normalize_llm_status_rows(
        self,
        *,
        provider_rows: list[dict[str, Any]],
        model_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        now_epoch = int(time.time())
        normalized_providers: list[dict[str, Any]] = []
        disabled_provider_ids: set[str] = set()
        disabled_provider_status: dict[str, tuple[int | None, int | None]] = {}
        for row in provider_rows:
            if not isinstance(row, dict):
                continue
            cloned = dict(row)
            provider_id = str(cloned.get("id") or "").strip().lower()
            if not provider_id:
                continue
            health_raw = cloned.get("health") if isinstance(cloned.get("health"), dict) else {}
            if provider_id == "ollama":
                health_raw = self._effective_provider_health(provider_id, health_raw if isinstance(health_raw, dict) else {})
            provider_health_error = (
                str((health_raw or {}).get("last_error_kind") or "").strip().lower()
                if isinstance(health_raw, dict)
                else ""
            )
            provider_is_disabled = (not bool(cloned.get("enabled", True))) or provider_health_error == "provider_disabled"
            if provider_is_disabled:
                disabled_payload = dict(health_raw) if isinstance(health_raw, dict) else {}
                provider_status_code = (
                    int(disabled_payload.get("status_code"))
                    if isinstance(disabled_payload.get("status_code"), int)
                    else None
                )
                provider_last_status_code = (
                    int(disabled_payload.get("last_status_code"))
                    if isinstance(disabled_payload.get("last_status_code"), int)
                    else provider_status_code
                )
                disabled_payload.update(
                    {
                        "status": "down",
                        "last_error_kind": "provider_disabled",
                        "status_code": provider_status_code,
                        "last_status_code": provider_last_status_code,
                        "last_checked_at": now_epoch,
                        "last_ts": float(now_epoch),
                    }
                )
                health_raw = disabled_payload
            normalized_health = self._normalize_health_record(
                health_raw if isinstance(health_raw, dict) else {},
                now_epoch=now_epoch,
            )
            if provider_is_disabled:
                disabled_provider_ids.add(provider_id)
                provider_status_code = (
                    int(normalized_health.get("status_code"))
                    if isinstance(normalized_health.get("status_code"), int)
                    else None
                )
                provider_last_status_code = (
                    int(normalized_health.get("last_status_code"))
                    if isinstance(normalized_health.get("last_status_code"), int)
                    else provider_status_code
                )
                disabled_provider_status[provider_id] = (provider_status_code, provider_last_status_code)
            cloned["health"] = normalized_health
            normalized_providers.append(cloned)

        normalized_models: list[dict[str, Any]] = []
        for row in model_rows:
            if not isinstance(row, dict):
                continue
            cloned = dict(row)
            model_id = str(cloned.get("id") or "").strip()
            if not model_id:
                continue
            provider_id = str(cloned.get("provider") or "").strip().lower()
            if not provider_id and ":" in model_id:
                provider_id = model_id.split(":", 1)[0].strip().lower()
            health_raw = cloned.get("health") if isinstance(cloned.get("health"), dict) else {}
            if provider_id in disabled_provider_ids:
                cloned["routable"] = False
                disabled_payload = dict(health_raw) if isinstance(health_raw, dict) else {}
                provider_status_code, provider_last_status_code = disabled_provider_status.get(provider_id, (None, None))
                existing_status_code = (
                    int(disabled_payload.get("status_code"))
                    if isinstance(disabled_payload.get("status_code"), int)
                    else None
                )
                existing_last_status_code = (
                    int(disabled_payload.get("last_status_code"))
                    if isinstance(disabled_payload.get("last_status_code"), int)
                    else existing_status_code
                )
                disabled_payload.update(
                    {
                        "status": "down",
                        "last_error_kind": "provider_disabled",
                        "status_code": provider_status_code if provider_status_code is not None else existing_status_code,
                        "last_status_code": (
                            provider_last_status_code
                            if provider_last_status_code is not None
                            else existing_last_status_code
                        ),
                        "last_checked_at": now_epoch,
                        "last_ts": float(now_epoch),
                    }
                )
                health_raw = disabled_payload
            cloned["health"] = self._normalize_health_record(
                health_raw if isinstance(health_raw, dict) else {},
                now_epoch=now_epoch,
            )
            normalized_models.append(cloned)
        return normalized_providers, normalized_models

    def _probe_llm_model(self, provider_id: str, model_id: str, timeout_seconds: float) -> dict[str, Any]:
        built = self._probe_provider_cfg(provider_id)
        if built is None:
            return {
                "status": "down",
                "error_kind": "provider_not_found",
                "status_code": None,
                "detail": "provider not found",
                "duration_ms": 0,
            }
        provider_cfg, _provider_payload = built
        models = self.registry_document.get("models") if isinstance(self.registry_document.get("models"), dict) else {}
        model_payload = models.get(model_id) if isinstance(models.get(model_id), dict) else {}
        model_name = str(model_payload.get("model") or "").strip() or (
            str(model_id).split(":", 1)[1] if ":" in str(model_id) else str(model_id)
        )
        return probe_model(
            provider_cfg,
            model_name,
            timeout_seconds=float(timeout_seconds),
            model_capabilities=list(model_payload.get("capabilities") or []),
            http_post_json=self._http_post_json,
        )

    def _probe_llm_candidate(self, provider_id: str, model_id: str, timeout_seconds: float) -> dict[str, Any]:
        provider_probe = self._probe_llm_provider(provider_id, timeout_seconds)
        provider_error = str(provider_probe.get("error_kind") or "").strip().lower()
        if provider_error == "not_applicable":
            return {
                "ok": True,
                "error_kind": "not_applicable",
                "status_code": None,
                "message": str(provider_probe.get("detail") or ""),
            }
        if str(provider_probe.get("status") or "").strip().lower() != "ok":
            return {
                "ok": False,
                "status": str(provider_probe.get("status") or "").strip().lower() or "degraded",
                "error_kind": provider_error or "provider_error",
                "status_code": provider_probe.get("status_code"),
                "message": str(provider_probe.get("detail") or ""),
            }
        if str(provider_id or "").strip().lower() == "ollama" and bool(provider_probe.get("native_ok", False)):
            return {
                "ok": True,
                "provider": provider_id,
                "model": model_id,
                "message": "ollama native probe ok",
            }

        model_probe = self._probe_llm_model(provider_id, model_id, timeout_seconds)
        model_error = str(model_probe.get("error_kind") or "").strip().lower()
        if model_error == "not_applicable":
            return {
                "ok": True,
                "error_kind": "not_applicable",
                "status_code": None,
                "message": str(model_probe.get("detail") or ""),
            }
        if str(model_probe.get("status") or "").strip().lower() == "ok":
            return {
                "ok": True,
                "provider": provider_id,
                "model": model_id,
            }
        return {
            "ok": False,
            "status": str(model_probe.get("status") or "").strip().lower() or "degraded",
            "error_kind": model_error or "provider_error",
            "status_code": model_probe.get("status_code"),
            "message": str(model_probe.get("detail") or ""),
        }

    def _save_registry_document(self, document: dict[str, Any]) -> None:
        with self._registry_lock:
            self._normalize_defaults_for_persistence(document)
            try:
                self.registry_store.write_document(document)
            except OSError as exc:
                raise RuntimeError(f"registry_path not writable: {self.registry_store.path}") from exc
            self._reload_router()

    def _persist_registry_document(self, document: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        try:
            self._save_registry_document(document)
        except RuntimeError as exc:
            return False, {"ok": False, "error": str(exc)}
        return True, None

    @staticmethod
    def _registry_hash(document: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _persist_registry_document_transactional(
        self,
        plan_apply_fn: Any,
    ) -> tuple[bool, dict[str, Any]]:
        def _wrapped_plan_apply(current: dict[str, Any]) -> dict[str, Any]:
            updated = plan_apply_fn(current)
            document = updated if isinstance(updated, dict) else {}
            self._normalize_defaults_for_persistence(document)
            return document

        with self._registry_lock:
            result = apply_with_rollback(
                registry_path=self.registry_store.path,
                snapshot_store=self._registry_snapshot_store,
                plan_apply_fn=_wrapped_plan_apply,
            )
            self._reload_router()
            snapshot_id_after: str | None = None
            if bool(result.get("ok")):
                try:
                    after_snapshot = self._registry_snapshot_store.create_snapshot(
                        self.registry_store.path,
                        self.registry_document if isinstance(self.registry_document, dict) else {},
                    )
                    snapshot_id_after = str(after_snapshot.get("snapshot_id") or "").strip() or None
                except Exception:
                    snapshot_id_after = None
        if not bool(result.get("ok")):
            error_kind = str(result.get("error_kind") or "registry_write_failed")
            return False, {
                "ok": False,
                "error": error_kind,
                "snapshot_id": result.get("snapshot_id"),
                "verify_error": result.get("verify_error"),
            }
        return True, {
            "ok": True,
            "snapshot_id": result.get("snapshot_id"),
            "snapshot_id_after": snapshot_id_after,
            "resulting_registry_hash": result.get("resulting_registry_hash"),
        }

    def _record_action_ledger(
        self,
        *,
        action: str,
        actor: str,
        decision: str,
        outcome: str,
        reason: str,
        trigger: str | None,
        snapshot_id: str | None,
        snapshot_id_after: str | None = None,
        resulting_registry_hash: str | None = None,
        changed_ids: list[str] | None = None,
    ) -> None:
        try:
            self._action_ledger.append(
                ts=int(time.time()),
                action=action,
                actor=actor,
                decision=decision,
                outcome=outcome,
                reason=reason,
                trigger=trigger,
                snapshot_id=snapshot_id,
                snapshot_id_after=snapshot_id_after,
                resulting_registry_hash=resulting_registry_hash,
                changed_ids=sorted(
                    {
                        str(item).strip()
                        for item in (changed_ids or [])
                        if str(item).strip()
                    }
                ),
            )
        except Exception:
            return

    def _is_remote_provider_id(self, provider_id: str) -> bool:
        providers = self.registry_document.get("providers") if isinstance(self.registry_document.get("providers"), dict) else {}
        payload = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
        return not bool(payload.get("local", False))

    def _is_remote_model_id(self, model_id: str) -> bool:
        models = self.registry_document.get("models") if isinstance(self.registry_document.get("models"), dict) else {}
        model_payload = models.get(model_id) if isinstance(models.get(model_id), dict) else {}
        provider_id = str(model_payload.get("provider") or "").strip().lower()
        if not provider_id:
            return True
        return self._is_remote_provider_id(provider_id)

    def _apply_safe_mode_to_plan(
        self,
        *,
        action: str,
        plan: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        if not bool(self.config.llm_autopilot_safe_mode):
            return plan, []
        paused = self._autopilot_apply_pause_enabled()
        changes = plan.get("changes") if isinstance(plan.get("changes"), list) else []
        blocked_lines: list[str] = []
        kept_changes: list[dict[str, Any]] = []
        for row in sorted((item for item in changes if isinstance(item, dict)), key=self._plan_change_sort_key):
            kind = str(row.get("kind") or "").strip().lower()
            field = str(row.get("field") or "").strip()
            after_value = row.get("after")
            blocked_reason: str | None = None

            if kind == "provider" and field == "enabled" and bool(after_value):
                provider_id = str(row.get("id") or "").strip().lower()
                if provider_id and self._is_remote_provider_id(provider_id):
                    blocked_reason = f"{action}: blocked enabling remote provider {provider_id}"
            elif kind == "model" and field in {"enabled", "available"} and bool(after_value):
                model_id = str(row.get("id") or "").strip()
                if model_id and self._is_remote_model_id(model_id):
                    blocked_reason = f"{action}: blocked enabling remote model {model_id}"
            elif kind == "defaults" and field == "allow_remote_fallback":
                before_value = bool(row.get("before", False))
                if not before_value and bool(after_value):
                    blocked_reason = f"{action}: blocked enabling remote fallback"
            elif kind == "defaults" and field == "default_provider":
                provider_id = str(after_value or "").strip().lower()
                if provider_id and self._is_remote_provider_id(provider_id):
                    blocked_reason = f"{action}: blocked switching default provider to remote {provider_id}"
            elif kind == "defaults" and field == "default_model":
                model_id = str(after_value or "").strip()
                if model_id and self._is_remote_model_id(model_id):
                    blocked_reason = f"{action}: blocked switching default model to remote {model_id}"

            if blocked_reason:
                blocked_lines.append(blocked_reason)
            elif not paused:
                kept_changes.append(row)

        if paused:
            filtered = copy.deepcopy(plan if isinstance(plan, dict) else {})
            filtered["changes"] = []
            impact = filtered.get("impact") if isinstance(filtered.get("impact"), dict) else {}
            impact["changes_count"] = 0
            filtered["impact"] = impact
            reasons = filtered.get("reasons") if isinstance(filtered.get("reasons"), list) else []
            reasons_set = {str(item).strip() for item in reasons if str(item).strip()}
            reasons_set.add("safe_mode_paused")
            reasons_set.add("safe_mode_blocked")
            filtered["reasons"] = sorted(reasons_set)
            paused_reason = str(self._safe_mode_status().get("safe_mode_reason") or "churn_detected").strip() or "churn_detected"
            if blocked_lines:
                dedup_blocked = sorted({str(item).strip() for item in blocked_lines if str(item).strip()})
                blocked_reason = build_safe_mode_paused_message(
                    reason=paused_reason,
                    blocked_detail=dedup_blocked[0],
                )
                self._safe_mode_last_blocked_reason = blocked_reason
                return filtered, [blocked_reason]
            blocked_reason = build_safe_mode_paused_message(reason=paused_reason, blocked_detail=action)
            self._safe_mode_last_blocked_reason = blocked_reason
            return filtered, [blocked_reason]

        if not blocked_lines:
            return plan, []
        dedup_blocked = sorted({str(item).strip() for item in blocked_lines if str(item).strip()})
        filtered = copy.deepcopy(plan)
        filtered["changes"] = kept_changes
        impact = filtered.get("impact") if isinstance(filtered.get("impact"), dict) else {}
        impact["changes_count"] = len(kept_changes)
        filtered["impact"] = impact
        reasons = filtered.get("reasons") if isinstance(filtered.get("reasons"), list) else []
        reasons_set = {str(item).strip() for item in reasons if str(item).strip()}
        reasons_set.add("safe_mode_blocked")
        filtered["reasons"] = sorted(reasons_set)
        self._safe_mode_last_blocked_reason = dedup_blocked[0]
        return filtered, dedup_blocked

    @staticmethod
    def _plan_change_sort_key(change: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(change.get("kind") or "").strip().lower(),
            str(change.get("id") or "").strip(),
            str(change.get("field") or "").strip(),
        )

    def _log_request(self, endpoint: str, ok: bool, payload: dict[str, Any]) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "ok": bool(ok),
            "payload": payload,
        }
        self._request_log.appendleft(record)

    def _safe_log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = str(event_type or "").strip() or "runtime_event"
        body = payload if isinstance(payload, dict) else {"value": str(payload)}
        try:
            log_event(self.config.log_path, event, body)
        except Exception:
            pass

    @staticmethod
    def _scheduler_wait(
        *,
        stop_event: threading.Event,
        seconds: float,
        sleep_fn: Callable[[float], None] | None,
    ) -> bool:
        duration = max(0.0, float(seconds))
        if sleep_fn is None:
            return bool(stop_event.wait(duration))
        try:
            sleep_fn(duration)
        except Exception:
            return bool(stop_event.is_set())
        return bool(stop_event.is_set())

    @staticmethod
    def _provider_public_payload(provider_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        api_source = payload.get("api_key_source") if isinstance(payload.get("api_key_source"), dict) else None
        return {
            "id": provider_id,
            "provider_type": payload.get("provider_type"),
            "base_url": payload.get("base_url"),
            "chat_path": payload.get("chat_path"),
            "enabled": bool(payload.get("enabled", True)),
            "local": bool(payload.get("local", False)),
            "api_key_source": {
                "type": (api_source or {}).get("type"),
                "name": (api_source or {}).get("name"),
            }
            if api_source
            else None,
            "default_headers": payload.get("default_headers") or {},
            "default_query_params": payload.get("default_query_params") or {},
        }

    def _sorted_provider_ids(self) -> list[str]:
        providers = self.registry_document.get("providers") if isinstance(self.registry_document.get("providers"), dict) else {}
        return sorted(str(provider_id) for provider_id in providers.keys())

    def _models_for_provider(self, provider_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        models = self.registry_document.get("models") if isinstance(self.registry_document.get("models"), dict) else {}
        for model_id, payload in sorted(models.items()):
            if not isinstance(payload, dict):
                continue
            if str(payload.get("provider") or "").strip().lower() != provider_id:
                continue
            rows.append({"id": model_id, **payload})
        return rows

    def _ensure_defaults(self, document: dict[str, Any]) -> dict[str, Any]:
        defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
        if not isinstance(defaults, dict):
            defaults = {}
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        defaults.setdefault("routing_mode", "auto")
        defaults.setdefault("default_provider", None)
        defaults.setdefault("chat_model", None)
        defaults.setdefault("embed_model", None)
        defaults.setdefault("default_model", None)
        defaults.setdefault("last_chat_model", None)
        defaults.setdefault("allow_remote_fallback", True)
        defaults.setdefault("fallback_chain", [])
        chat_model = str(defaults.get("chat_model") or "").strip() or None
        legacy_default_model = str(defaults.get("default_model") or "").strip() or None
        if chat_model is None and legacy_default_model:
            chat_model = legacy_default_model
        if chat_model:
            defaults["chat_model"] = chat_model
            defaults["default_model"] = chat_model
        else:
            defaults["chat_model"] = None
            defaults["default_model"] = None

        last_chat_model = str(defaults.get("last_chat_model") or "").strip() or None
        defaults["last_chat_model"] = last_chat_model

        embed_model = str(defaults.get("embed_model") or "").strip() or None
        if embed_model is None:
            embed_model = self._best_local_embedding_model(models)
        defaults["embed_model"] = embed_model
        document["defaults"] = defaults
        return defaults

    @staticmethod
    def _best_local_embedding_model(models: dict[str, Any]) -> str | None:
        candidates: list[str] = []
        for model_id, payload in sorted(models.items()):
            if not isinstance(payload, dict):
                continue
            provider = str(payload.get("provider") or "").strip().lower()
            if provider != "ollama":
                continue
            valid_default, validated_model, _validation_error = validate_default_model(
                str(model_id),
                models,
                purpose="embedding",
            )
            if not valid_default or not validated_model:
                continue
            candidates.append(validated_model)
        if not candidates:
            return None
        preferred = [item for item in candidates if "nomic-embed-text" in item]
        if preferred:
            return sorted(preferred)[0]
        return sorted(candidates)[0]

    def _normalize_defaults_for_persistence(self, document: dict[str, Any]) -> None:
        defaults = self._ensure_defaults(document)
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        provider_ids = {str(provider_id).strip().lower() for provider_id in providers.keys()}
        provider_for_model = str(defaults.get("default_provider") or "").strip().lower() or None

        raw_chat_model = (
            str(defaults.get("chat_model") or "").strip()
            or str(defaults.get("default_model") or "").strip()
            or None
        )
        canonical_chat_model = None
        if raw_chat_model is not None:
            canonical_chat_model = self._normalize_default_model_id(
                raw_chat_model,
                provider_for_model=provider_for_model,
                models=models,
                provider_ids=provider_ids,
            )
        if canonical_chat_model is None and raw_chat_model is None:
            defaults["chat_model"] = None
            defaults["default_model"] = None
        else:
            valid_chat, validated_chat, _chat_error = validate_default_model(
                canonical_chat_model,
                models,
                purpose="chat",
            )
            if valid_chat:
                defaults["chat_model"] = validated_chat
                defaults["default_model"] = validated_chat
            else:
                defaults["chat_model"] = None
                defaults["default_model"] = None

        raw_last_chat_model = str(defaults.get("last_chat_model") or "").strip() or None
        canonical_last_chat_model = None
        if raw_last_chat_model is not None:
            canonical_last_chat_model = self._normalize_default_model_id(
                raw_last_chat_model,
                provider_for_model=provider_for_model,
                models=models,
                provider_ids=provider_ids,
            )
        defaults["last_chat_model"] = canonical_last_chat_model or raw_last_chat_model

        raw_embed_model = str(defaults.get("embed_model") or "").strip() or None
        canonical_embed_model = None
        if raw_embed_model is not None:
            canonical_embed_model = self._normalize_default_model_id(
                raw_embed_model,
                provider_for_model=provider_for_model,
                models=models,
                provider_ids=provider_ids,
            )
        valid_embed, validated_embed, _embed_error = validate_default_model(
            canonical_embed_model,
            models,
            purpose="embedding",
        )
        if valid_embed:
            defaults["embed_model"] = validated_embed
        else:
            defaults["embed_model"] = self._best_local_embedding_model(models)
        document["defaults"] = defaults

    def health(self) -> dict[str, Any]:
        snapshot = self._router.doctor_snapshot()
        return {
            "ok": True,
            "service": "personal-agent-api",
            "time": datetime.now(timezone.utc).isoformat(),
            "routing_mode": snapshot.get("routing_mode"),
            "configured_providers": [item.get("id") for item in snapshot.get("providers") or []],
            "registry_path": self.registry_store.path,
            "safe_mode": self._safe_mode_health_payload(),
        }

    def models(self) -> dict[str, Any]:
        snapshot = self._router.doctor_snapshot()
        return {
            "providers": snapshot.get("providers") or [],
            "models": snapshot.get("models") or [],
            "routing_mode": snapshot.get("routing_mode"),
            "defaults": snapshot.get("defaults") or {},
            "circuits": snapshot.get("circuits") or {},
        }

    def version_info(self) -> dict[str, Any]:
        return {
            "ok": True,
            "version": self.version,
            "git_commit": self.git_commit,
            "started_at": self.started_at_iso,
            "pid": self.pid,
            "listening": self.listening_url,
        }

    def list_providers(self) -> dict[str, Any]:
        snapshot = self._router.doctor_snapshot()
        provider_health = {
            str(item.get("id") or ""): item.get("health") or {}
            for item in (snapshot.get("providers") or [])
            if isinstance(item, dict)
        }
        model_health = {
            str(item.get("id") or ""): item.get("health") or {}
            for item in (snapshot.get("models") or [])
            if isinstance(item, dict)
        }
        providers = self.registry_document.get("providers") if isinstance(self.registry_document.get("providers"), dict) else {}
        rows = [
            {
                **self._provider_public_payload(provider_id, payload),
                "health": provider_health.get(provider_id, {}),
                "models": [
                    {
                        **model_row,
                        "health": model_health.get(str(model_row.get("id") or ""), {}),
                    }
                    for model_row in self._models_for_provider(provider_id)
                ],
            }
            for provider_id, payload in sorted(providers.items())
            if isinstance(payload, dict)
        ]
        return {"providers": rows}

    def _refresh_telegram_config_cache(self) -> None:
        token = (self.secret_store.get_secret(_TELEGRAM_BOT_TOKEN_SECRET_KEY) or "").strip()
        token_source = "secret_store" if token else "none"
        if not token:
            env_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
            if env_token:
                token = env_token
                token_source = "env"
        self._telegram_configured_cached = bool(token)
        self._telegram_token_source_cached = token_source

    def telegram_status(self) -> dict[str, Any]:
        runner_status = (
            self._telegram_runner.status()
            if self._telegram_runner is not None and hasattr(self._telegram_runner, "status")
            else {}
        )
        state = str(runner_status.get("state") or "").strip() or "stopped"
        if not self._telegram_configured_cached and state == "stopped":
            state = "disabled_missing_token"
        return {
            "ok": True,
            "configured": bool(self._telegram_configured_cached),
            "token_source": str(self._telegram_token_source_cached or "none"),
            "state": state,
            "embedded_running": bool(runner_status.get("embedded_running", False)),
            "last_event": str(runner_status.get("last_event") or ""),
            "last_error": str(runner_status.get("last_error") or "") or None,
            "last_ts": float(runner_status.get("last_ts") or 0.0),
            "last_ts_iso": str(runner_status.get("last_ts_iso") or "") or None,
            "consecutive_failures": int(runner_status.get("consecutive_failures") or 0),
        }

    def ready_status(self) -> dict[str, Any]:
        telegram = self.telegram_status()
        telegram_state = str(telegram.get("state") or "stopped")
        hf_status = self._model_watch_hf_status_snapshot()
        phase = str(self.startup_phase or "starting").strip().lower() or "starting"
        warmup_remaining = self._warmup_remaining_snapshot()
        if phase == "starting" and not self._startup_warmup_started and not warmup_remaining:
            # Runtime can be instantiated directly in tests/tools without going through run_server().
            # Treat that mode as immediately ready.
            phase = "ready"
        telegram_ready = telegram_state in {"running", "disabled_missing_token"}
        ready = bool(phase == "ready" and telegram_ready)
        uptime_seconds = max(0, int((datetime.now(timezone.utc) - self.started_at).total_seconds()))
        recent_messages = self._ready_recent_telegram_messages(limit=5)
        if phase == "degraded":
            message = "Startup warmup degraded. Check logs and retry."
        elif phase in {"starting", "listening", "warming"}:
            message = "Starting up... retrying. Try /ready again in a moment."
        elif ready and telegram_state == "running":
            message = "Agent is ready. Telegram is running."
        elif ready and telegram_state == "disabled_missing_token":
            message = "Agent is ready. Telegram is disabled (missing token)."
        else:
            message = "Starting up... retrying. Try /ready again in a moment."
        return {
            "ok": True,
            "ready": bool(ready),
            "phase": phase,
            "warmup_remaining": list(warmup_remaining),
            "last_error": str(self._startup_last_error or "") or None,
            "api": {
                "version": self.version,
                "git_commit": self.git_commit,
                "pid": self.pid,
                "started_at": self.started_at_iso,
                "uptime_seconds": uptime_seconds,
            },
            "telegram": {
                "configured": bool(telegram.get("configured", False)),
                "token_source": str(telegram.get("token_source") or "none"),
                "state": telegram_state,
                "status": telegram_state,
                "embedded_running": bool(telegram.get("embedded_running", False)),
                "consecutive_failures": int(telegram.get("consecutive_failures") or 0),
                "last_event": str(telegram.get("last_event") or ""),
                "last_error": str(telegram.get("last_error") or "") or None,
                "last_ts": float(telegram.get("last_ts") or 0.0),
                "last_ts_iso": str(telegram.get("last_ts_iso") or "") or None,
                "recent_messages": recent_messages,
            },
            "model_watch": {
                "hf": hf_status,
            },
            "message": message,
        }

    def _model_watch_hf_status_snapshot(self) -> dict[str, Any]:
        status = self._model_watch_hf_last_status if isinstance(self._model_watch_hf_last_status, dict) else {}
        last_run_ts_raw = status.get("last_run_ts")
        try:
            last_run_ts = int(last_run_ts_raw) if last_run_ts_raw is not None else None
        except (TypeError, ValueError):
            last_run_ts = None
        return {
            "enabled": bool(status.get("enabled", False)),
            "last_run_ts": last_run_ts,
            "last_error": str(status.get("last_error") or "").strip() or None,
            "discovered_count": int(status.get("discovered_count") or 0),
            "tracked_repos": int(status.get("tracked_repos") or 0),
        }

    @staticmethod
    def _ready_sanitize_params_redacted(params: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for raw_key, raw_value in (params or {}).items():
            key = str(raw_key or "").strip()
            lowered = key.lower()
            if not key:
                continue
            if lowered in {"text", "message", "content", "prompt", "chat_id", "raw_chat_id"}:
                continue
            if "chat_id" in lowered and "redacted" not in lowered:
                continue
            cleaned[key] = raw_value
        return cleaned

    def _ready_recent_telegram_messages(self, *, limit: int = 5) -> list[dict[str, Any]]:
        max_rows = max(1, int(limit))
        rows: list[dict[str, Any]] = []
        for entry in self.audit_log.recent(limit=100):
            if not isinstance(entry, dict):
                continue
            action = str(entry.get("action") or "").strip()
            if not action.startswith("telegram.message."):
                continue
            params = entry.get("params_redacted") if isinstance(entry.get("params_redacted"), dict) else {}
            rows.append(
                {
                    "ts": str(entry.get("ts") or ""),
                    "action": action,
                    "outcome": str(entry.get("outcome") or ""),
                    "reason": str(entry.get("reason") or ""),
                    "error_kind": str(entry.get("error_kind") or "") or None,
                    "params_redacted": self._ready_sanitize_params_redacted(params),
                }
            )
            if len(rows) >= max_rows:
                break
        return rows

    def _resolve_telegram_target(self) -> tuple[str | None, str | None]:
        token = (self.secret_store.get_secret(_TELEGRAM_BOT_TOKEN_SECRET_KEY) or "").strip()
        if not token:
            token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id: str | None = None
        db = self._open_perception_db()
        try:
            if db is not None:
                chat_id = (db.get_preference("telegram_chat_id") or "").strip() or None
        finally:
            if db is not None:
                db.close()
        return (token or None), chat_id

    @staticmethod
    def _send_telegram_message(token: str, chat_id: str, text: str) -> None:
        payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw or "{}")
        if not isinstance(parsed, dict) or not bool(parsed.get("ok")):
            raise RuntimeError(str((parsed or {}).get("description") or "telegram_send_failed"))

    def set_telegram_secret(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        token = str(payload.get("bot_token") or "").strip()
        if not token:
            return False, {"ok": False, "error": "bot_token is required"}
        self.secret_store.set_secret(_TELEGRAM_BOT_TOKEN_SECRET_KEY, token)
        self._refresh_telegram_config_cache()
        return True, {"ok": True}

    def test_telegram(self) -> tuple[bool, dict[str, Any]]:
        token = (self.secret_store.get_secret(_TELEGRAM_BOT_TOKEN_SECRET_KEY) or "").strip()
        if not token:
            return False, {"ok": False, "error": "telegram token not configured"}

        url = f"https://api.telegram.org/bot{token}/getMe"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=8.0) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw or "{}")
        except urllib.error.HTTPError as exc:
            body_bytes = exc.read() if hasattr(exc, "read") else b""
            body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
            error_message = "telegram_api_error"
            try:
                parsed_body = json.loads(body_text or "{}")
                if isinstance(parsed_body, dict):
                    error_message = str(parsed_body.get("description") or error_message)
            except json.JSONDecodeError:
                pass
            return False, {"ok": False, "error": "telegram_api_error", "message": error_message}
        except (OSError, TimeoutError, ValueError, UnicodeError, json.JSONDecodeError, urllib.error.URLError) as exc:
            return False, {"ok": False, "error": "telegram_request_failed", "message": str(exc) or "request_failed"}

        if not isinstance(parsed, dict):
            return False, {"ok": False, "error": "telegram_invalid_response"}

        if not bool(parsed.get("ok")):
            return False, {
                "ok": False,
                "error": "telegram_api_error",
                "message": str(parsed.get("description") or "request_failed"),
            }

        result = parsed.get("result") if isinstance(parsed.get("result"), dict) else {}
        return True, {
            "ok": True,
            "telegram_user": {
                "id": result.get("id"),
                "username": result.get("username"),
                "first_name": result.get("first_name"),
                "is_bot": bool(result.get("is_bot")),
            },
        }

    @staticmethod
    def _normalize_model_payload(provider_id: str, raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        model_name = str(raw.get("model") or raw.get("name") or "").strip()
        if not model_name:
            raise ValueError("model is required")
        model_id = str(raw.get("id") or f"{provider_id}:{model_name}").strip()
        capabilities = raw.get("capabilities") if isinstance(raw.get("capabilities"), list) else ["chat"]
        pricing_raw = raw.get("pricing") if isinstance(raw.get("pricing"), dict) else {}
        payload = {
            "provider": provider_id,
            "model": model_name,
            "capabilities": [str(item).strip().lower() for item in capabilities if str(item).strip()],
            "quality_rank": int(raw.get("quality_rank", 5) or 5),
            "cost_rank": int(raw.get("cost_rank", 5) or 5),
            "default_for": [str(item) for item in (raw.get("default_for") or ["chat"])],
            "enabled": bool(raw.get("enabled", True)),
            "available": bool(raw.get("available", True)),
            "pricing": {
                "input_per_million_tokens": pricing_raw.get("input_per_million_tokens"),
                "output_per_million_tokens": pricing_raw.get("output_per_million_tokens"),
            },
            "max_context_tokens": raw.get("max_context_tokens"),
        }
        return model_id, payload

    def add_provider(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        provider_id = str(payload.get("id") or payload.get("provider_id") or "").strip().lower()
        if not _PROVIDER_ID_RE.match(provider_id):
            return False, {"ok": False, "error": "provider id must match [a-z0-9_-]{2,64}"}

        provider_type = str(payload.get("provider_type") or "openai_compat").strip().lower()
        if provider_type not in {"openai_compat"}:
            return False, {"ok": False, "error": "provider_type must be openai_compat"}

        base_url = str(payload.get("base_url") or "").strip()
        if not base_url:
            return False, {"ok": False, "error": "base_url is required"}

        chat_path = str(payload.get("chat_path") or "/v1/chat/completions").strip() or "/v1/chat/completions"
        if not chat_path.startswith("/"):
            chat_path = "/" + chat_path

        document = self.registry_document
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        if provider_id in providers:
            return False, {"ok": False, "error": f"provider already exists: {provider_id}"}

        api_key_source = payload.get("api_key_source") if isinstance(payload.get("api_key_source"), dict) else None
        if api_key_source is None:
            auth_env_var = str(payload.get("auth_env_var") or "").strip()
            if auth_env_var:
                api_key_source = {"type": "env", "name": auth_env_var}
            elif bool(payload.get("requires_api_key", True)):
                api_key_source = {"type": "secret", "name": f"provider:{provider_id}:api_key"}

        providers[provider_id] = {
            "provider_type": provider_type,
            "base_url": base_url,
            "chat_path": chat_path,
            "api_key_source": api_key_source,
            "default_headers": payload.get("default_headers") if isinstance(payload.get("default_headers"), dict) else {},
            "default_query_params": payload.get("default_query_params")
            if isinstance(payload.get("default_query_params"), dict)
            else {},
            "enabled": bool(payload.get("enabled", True)),
            "local": bool(payload.get("local", False)),
        }

        model_items = payload.get("models") if isinstance(payload.get("models"), list) else []
        single_model = str(payload.get("model") or "").strip()
        if single_model:
            model_items.append({"model": single_model, "id": payload.get("model_id")})

        for model_raw in model_items:
            if not isinstance(model_raw, dict):
                continue
            try:
                model_id, model_payload = self._normalize_model_payload(provider_id, model_raw)
            except ValueError:
                continue
            models[model_id] = model_payload

        document["providers"] = providers
        document["models"] = models
        self._ensure_defaults(document)
        saved, error = self._persist_registry_document(document)
        if not saved:
            assert error is not None
            return False, error
        self._schedule_autoconfig_soon()

        return True, {
            "ok": True,
            "provider": self._provider_public_payload(provider_id, providers[provider_id]),
            "models": self._models_for_provider(provider_id),
        }

    def update_provider(self, provider_id: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        provider_key = provider_id.strip().lower()
        document = self.registry_document
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        if provider_key not in providers:
            return False, {"ok": False, "error": "provider not found"}

        current = dict(providers[provider_key])
        if "base_url" in payload:
            current["base_url"] = str(payload.get("base_url") or "").strip() or current.get("base_url")
        if "chat_path" in payload:
            chat_path = str(payload.get("chat_path") or "").strip() or current.get("chat_path")
            if not str(chat_path).startswith("/"):
                chat_path = "/" + str(chat_path)
            current["chat_path"] = chat_path
        if "enabled" in payload:
            current["enabled"] = bool(payload.get("enabled"))
        if "local" in payload:
            current["local"] = bool(payload.get("local"))
        if isinstance(payload.get("default_headers"), dict):
            current["default_headers"] = payload.get("default_headers")
        if isinstance(payload.get("default_query_params"), dict):
            current["default_query_params"] = payload.get("default_query_params")
        if isinstance(payload.get("api_key_source"), dict):
            current["api_key_source"] = payload.get("api_key_source")

        providers[provider_key] = current
        document["providers"] = providers
        saved, error = self._persist_registry_document(document)
        if not saved:
            assert error is not None
            return False, error
        self._schedule_autoconfig_soon()
        return True, {
            "ok": True,
            "provider": self._provider_public_payload(provider_key, current),
        }

    def add_provider_model(self, provider_id: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        provider_key = provider_id.strip().lower()
        document = self.registry_document
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        if provider_key not in providers:
            return False, {"ok": False, "error": "provider not found"}

        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        try:
            model_id, model_payload = self._normalize_model_payload(provider_key, payload)
        except ValueError as exc:
            return False, {"ok": False, "error": str(exc)}

        if ":" in model_id:
            prefix = model_id.split(":", 1)[0].strip().lower()
            if prefix != provider_key:
                return False, {"ok": False, "error": "model id must be scoped to provider"}
        else:
            model_id = f"{provider_key}:{model_id}"

        models[model_id] = model_payload
        document["models"] = models
        saved, error = self._persist_registry_document(document)
        if not saved:
            assert error is not None
            return False, error
        self._schedule_autoconfig_soon()

        return True, {"ok": True, "model": {"id": model_id, **model_payload}}

    def delete_provider(self, provider_id: str) -> tuple[bool, dict[str, Any]]:
        provider_key = provider_id.strip().lower()
        document = self.registry_document
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        defaults = self._ensure_defaults(document)

        if provider_key not in providers:
            return False, {"ok": False, "error": "provider not found"}

        warning = None
        if str(defaults.get("default_provider") or "").strip().lower() == provider_key:
            warning = "removed provider was default_provider"
            defaults["default_provider"] = None
            if str(defaults.get("default_model") or "").startswith(f"{provider_key}:"):
                defaults["default_model"] = None

        providers.pop(provider_key, None)
        for model_id in list(models.keys()):
            model_payload = models.get(model_id)
            if isinstance(model_payload, dict) and str(model_payload.get("provider") or "").strip().lower() == provider_key:
                models.pop(model_id, None)

        document["providers"] = providers
        document["models"] = models
        document["defaults"] = defaults
        saved, error = self._persist_registry_document(document)
        if not saved:
            assert error is not None
            return False, error
        self._schedule_autoconfig_soon()

        response = {"ok": True, "deleted": provider_key}
        if warning:
            response["warning"] = warning
        return True, response

    def set_provider_secret(self, provider_id: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        provider_key = provider_id.strip().lower()
        api_key = str(payload.get("api_key") or "").strip()
        if not api_key:
            return False, {"ok": False, "error": "api_key is required"}

        document = self.registry_document
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        provider_payload = providers.get(provider_key)
        if not isinstance(provider_payload, dict):
            return False, {"ok": False, "error": "provider not found"}

        secret_key = f"provider:{provider_key}:api_key"
        desired_source = {"type": "secret", "name": secret_key}
        source = provider_payload.get("api_key_source") if isinstance(provider_payload.get("api_key_source"), dict) else None
        if source != desired_source:
            provider_payload["api_key_source"] = desired_source
            providers[provider_key] = provider_payload
            document["providers"] = providers
            saved, error = self._persist_registry_document(document)
            if not saved:
                assert error is not None
                return False, error

        self.secret_store.set_secret(secret_key, api_key)
        self._router.set_provider_api_key(provider_key, api_key)
        self._schedule_autoconfig_soon()
        return True, {"ok": True, "provider": provider_key}

    def _provider_default_model(self, provider_id: str) -> str | None:
        models = self.registry_document.get("models") if isinstance(self.registry_document.get("models"), dict) else {}
        for model_id, payload in sorted(models.items()):
            if not isinstance(payload, dict):
                continue
            if str(payload.get("provider") or "").strip().lower() == provider_id and bool(payload.get("enabled", True)):
                return str(model_id)
        return None

    @staticmethod
    def _normalize_provider_test_error(kind: str | None, status_code: int | None) -> str:
        normalized = str(kind or "").strip().lower()
        if status_code == 402:
            return "payment_required"
        if normalized in {
            "payment_required",
            "credits_insufficient",
            "insufficient_credits",
        }:
            return "payment_required"
        if normalized in {
            "auth_error",
            "rate_limit",
            "server_error",
            "bad_request",
            "misconfigured_path",
            "missing_auth",
            "bad_base_url",
        }:
            return normalized
        if status_code in {401, 403}:
            return "auth_error"
        if status_code == 429:
            return "rate_limit"
        if status_code is not None and 500 <= int(status_code) <= 599:
            return "server_error"
        return "bad_request"

    @staticmethod
    def _provider_test_message(kind: str) -> str:
        return {
            "auth_error": "Authentication failed for provider.",
            "rate_limit": "Provider rate limit reached.",
            "server_error": "Provider server error.",
            "bad_request": "Provider rejected request.",
            "payment_required": "Provider test hit a credits/limit issue. Add credits, lower max_tokens, or choose a cheaper model.",
            "misconfigured_path": "Provider chat_path/base_url path is misconfigured.",
            "missing_auth": "Provider requires authorization credentials but no Authorization header was set.",
            "bad_base_url": "Provider base_url is invalid.",
        }.get(kind, "Provider connectivity test failed.")

    def _resolve_provider_reference_value(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, str):
            return value
        if not isinstance(value, dict):
            return str(value)

        if "value" in value:
            static_value = value.get("value")
            return str(static_value) if static_value is not None else None
        if "from_env" in value:
            env_name = str(value.get("from_env") or "").strip()
            if not env_name:
                return None
            return (os.environ.get(env_name, "") or "").strip() or None
        if "from_secret" in value:
            secret_key = str(value.get("from_secret") or "").strip()
            if not secret_key:
                return None
            return (self.secret_store.get_secret(secret_key) or "").strip() or None
        if "from_secret_store" in value:
            secret_key = str(value.get("from_secret_store") or "").strip()
            if not secret_key:
                return None
            return (self.secret_store.get_secret(secret_key) or "").strip() or None
        return None

    def _provider_api_key(
        self,
        provider_payload: dict[str, Any],
        *,
        key_override: str | None = None,
    ) -> str | None:
        if key_override:
            return key_override.strip() or None
        source = provider_payload.get("api_key_source") if isinstance(provider_payload.get("api_key_source"), dict) else None
        if not isinstance(source, dict):
            return None
        source_type = str(source.get("type") or "").strip().lower()
        source_name = str(source.get("name") or "").strip()
        if not source_name:
            return None
        if source_type == "env":
            return (os.environ.get(source_name, "") or "").strip() or None
        if source_type == "secret":
            return (self.secret_store.get_secret(source_name) or "").strip() or None
        return None

    def _provider_request_headers(
        self,
        provider_payload: dict[str, Any],
        *,
        key_override: str | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        api_key = self._provider_api_key(provider_payload, key_override=key_override)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        default_headers = provider_payload.get("default_headers")
        if isinstance(default_headers, dict):
            for header_name, raw_value in default_headers.items():
                value = self._resolve_provider_reference_value(raw_value)
                if value is None or value == "":
                    continue
                headers[str(header_name)] = value
        return headers

    def _validate_provider_for_probe(
        self,
        provider_id: str,
        provider_payload: dict[str, Any],
        *,
        headers: dict[str, str],
        trigger: str,
    ) -> dict[str, Any]:
        payload = dict(provider_payload if isinstance(provider_payload, dict) else {})
        payload["_resolved_api_key_present"] = bool(self._provider_api_key(provider_payload))
        validation = validate_provider_call_format(provider_id, payload, headers=headers)
        self.audit_log.append(
            actor="system" if trigger == "scheduler" else "user",
            action="llm.provider.validate",
            params={
                "provider_id": str(provider_id or "").strip().lower(),
                "details": validation.get("details") if isinstance(validation.get("details"), dict) else {},
            },
            decision="allow" if bool(validation.get("ok")) else "deny",
            reason=str(validation.get("error_kind") or "ok"),
            dry_run=True,
            outcome="validated" if bool(validation.get("ok")) else "failed",
            error_kind=None if bool(validation.get("ok")) else str(validation.get("error_kind") or "provider_invalid"),
            duration_ms=0,
        )
        return validation

    @staticmethod
    def _canonical_model_name(provider_id: str, model_value: str, models: dict[str, Any]) -> str:
        candidate = str(model_value or "").strip()
        if not candidate:
            return ""
        model_payload = models.get(candidate) if isinstance(models.get(candidate), dict) else None
        if isinstance(model_payload, dict):
            if str(model_payload.get("provider") or "").strip().lower() == provider_id:
                return str(model_payload.get("model") or candidate).strip()
        if ":" in candidate:
            prefix, remainder = candidate.split(":", 1)
            if prefix.strip().lower() == provider_id:
                return remainder.strip()
        return candidate

    def _probe_provider_models(
        self,
        provider_id: str,
        provider_payload: dict[str, Any],
        *,
        key_override: str | None = None,
        timeout_seconds: float = 6.0,
    ) -> dict[str, Any]:
        base_url = str(provider_payload.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            return {"ok": False, "error": "bad_request", "message": "provider base_url is missing", "models": []}

        headers = self._provider_request_headers(provider_payload, key_override=key_override)
        validation = self._validate_provider_for_probe(
            provider_id,
            provider_payload,
            headers=headers,
            trigger="manual",
        )
        if not bool(validation.get("ok")):
            error_kind = str(validation.get("error_kind") or "provider_invalid")
            return {
                "ok": False,
                "error": error_kind,
                "status_code": None,
                "message": str(validation.get("message") or self._provider_test_message(error_kind)),
                "models": [],
            }
        provider_key = str(provider_id or "").strip().lower()
        if provider_key == "ollama" or bool(provider_payload.get("local", False)):
            bases = normalize_ollama_base_urls(base_url)
            native_url = str(bases.get("native_base") or "").strip()
            try:
                parsed = self._http_get_json(
                    native_url + "/api/tags",
                    timeout_seconds=timeout_seconds,
                    headers=headers,
                )
                models_rows = parsed.get("models") if isinstance(parsed.get("models"), list) else []
                model_names = sorted(
                    {
                        str(item.get("name") or item.get("model") or "").strip()
                        for item in models_rows
                        if isinstance(item, dict) and str(item.get("name") or item.get("model") or "").strip()
                    }
                )
                return {"ok": True, "models": model_names, "count": len(model_names)}
            except urllib.error.HTTPError as exc:
                status_code = int(getattr(exc, "code", 0) or 0) or None
                kind = self._normalize_provider_test_error(None, status_code)
                return {
                    "ok": False,
                    "error": kind,
                    "status_code": status_code,
                    "message": self._provider_test_message(kind),
                    "models": [],
                }
            except (OSError, TimeoutError, ValueError, UnicodeError, json.JSONDecodeError, urllib.error.URLError):
                return {
                    "ok": False,
                    "error": "server_error",
                    "status_code": None,
                    "message": "Unable to query Ollama /api/tags.",
                    "models": [],
                }
        try:
            parsed = self._http_get_json(base_url + "/v1/models", timeout_seconds=timeout_seconds, headers=headers)
            data = parsed.get("data") if isinstance(parsed.get("data"), list) else []
            model_names = [
                str(item.get("id") or "").strip()
                for item in data
                if isinstance(item, dict) and str(item.get("id") or "").strip()
            ]
            return {"ok": True, "models": model_names, "count": len(model_names)}
        except urllib.error.HTTPError as exc:
            status_code = int(getattr(exc, "code", 0) or 0) or None
            kind = self._normalize_provider_test_error(None, status_code)
            return {
                "ok": False,
                "error": kind,
                "status_code": status_code,
                "message": self._provider_test_message(kind),
                "models": [],
            }
        except (OSError, TimeoutError, ValueError, UnicodeError, json.JSONDecodeError, urllib.error.URLError):
            return {
                "ok": False,
                "error": "server_error",
                "status_code": None,
                "message": "Unable to query /v1/models.",
                "models": [],
            }

    def test_provider(self, provider_id: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        provider_key = provider_id.strip().lower()
        provider_entry = (
            self.registry_document.get("providers", {}).get(provider_key)
            if isinstance(self.registry_document.get("providers"), dict)
            else None
        )
        if not isinstance(provider_entry, dict):
            return False, {"ok": False, "error": "provider not found"}

        timeout_seconds = float(payload.get("timeout_seconds") or 6.0)
        key_override = str(payload.get("api_key") or "").strip() or None
        models = self.registry_document.get("models") if isinstance(self.registry_document.get("models"), dict) else {}
        model_override = self._canonical_model_name(
            provider_key,
            str(payload.get("model") or "").strip() or self._provider_default_model(provider_key) or "",
            models=models,
        )
        models_probe = self._probe_provider_models(
            provider_key,
            provider_entry,
            key_override=key_override,
            timeout_seconds=timeout_seconds,
        )
        if not bool(models_probe.get("ok")) and str(models_probe.get("error") or "") in {
            "misconfigured_path",
            "missing_auth",
            "bad_base_url",
        }:
            probe_error = str(models_probe.get("error") or "bad_request")
            response = {
                "ok": False,
                "provider": provider_key,
                "model": model_override or None,
                "error": probe_error,
                "error_kind": probe_error,
                "message": str(models_probe.get("message") or self._provider_test_message(probe_error)),
                "models_probe": models_probe,
            }
            self._log_request(f"/providers/{provider_key}/test", False, response)
            return False, response
        if not model_override:
            discovered = models_probe.get("models") if isinstance(models_probe.get("models"), list) else []
            if discovered:
                model_override = str(discovered[0]).strip()

        if not model_override:
            response = {
                "ok": False,
                "provider": provider_key,
                "model": None,
                "error": "bad_request",
                "error_kind": "bad_request",
                "message": "No model available for provider test.",
                "models_probe": models_probe,
            }
            self._log_request(f"/providers/{provider_key}/test", False, response)
            return False, response

        impl = self._router._providers.get(provider_key)  # type: ignore[attr-defined]
        if impl is None:
            response = {
                "ok": False,
                "provider": provider_key,
                "model": model_override,
                "error": "server_error",
                "error_kind": "server_error",
                "message": "Provider is not available in router.",
                "models_probe": models_probe,
            }
            self._log_request(f"/providers/{provider_key}/test", False, response)
            return False, response

        if key_override and hasattr(impl, "set_api_key"):
            getattr(impl, "set_api_key")(key_override)

        start = time.monotonic()
        try:
            prompt = (
                "Run a lightweight provider connectivity health check. "
                "Reply with exactly: OK."
            )
            response_obj = impl.chat(
                Request(
                    messages=(Message(role="user", content=prompt),),
                    purpose="health",
                    task_type="test",
                    temperature=0.0,
                    max_tokens=256,
                ),
                model=model_override,
                timeout_seconds=timeout_seconds,
            )
        except LLMError as exc:
            error_kind = self._normalize_provider_test_error(exc.kind, exc.status_code)
            if error_kind == "payment_required":
                message = self._provider_test_message(error_kind)
            else:
                message = str(exc.message or self._provider_test_message(error_kind))
            response = {
                "ok": False,
                "provider": provider_key,
                "model": model_override,
                "error": error_kind,
                "error_kind": error_kind,
                "status_code": exc.status_code,
                "message": message,
                "models_probe": models_probe,
            }
            self._log_request(f"/providers/{provider_key}/test", False, response)
            return False, response
        except (OSError, TimeoutError, ValueError, TypeError, RuntimeError, AttributeError, KeyError):
            response = {
                "ok": False,
                "provider": provider_key,
                "model": model_override,
                "error": "server_error",
                "error_kind": "server_error",
                "status_code": None,
                "message": self._provider_test_message("server_error"),
                "models_probe": models_probe,
            }
            self._log_request(f"/providers/{provider_key}/test", False, response)
            return False, response
        finally:
            if key_override and hasattr(impl, "set_api_key"):
                getattr(impl, "set_api_key")(None)

        response = {
            "ok": True,
            "provider": provider_key,
            "model": response_obj.model or model_override,
            "duration_ms": int((time.monotonic() - start) * 1000),
            "models_probe": models_probe,
        }
        self._log_request(f"/providers/{provider_key}/test", True, response)
        return True, response

    @staticmethod
    def _normalize_ollama_pull_model(model_value: str | None) -> str:
        normalized = str(model_value or "").strip().lower()
        if normalized.startswith("ollama:"):
            normalized = normalized.split(":", 1)[1].strip()
        return normalized

    def _ollama_tags_models(self, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
        providers = self.registry_document.get("providers") if isinstance(self.registry_document.get("providers"), dict) else {}
        ollama_provider = providers.get("ollama") if isinstance(providers.get("ollama"), dict) else {}
        raw_base = str(
            ollama_provider.get("base_url")
            or self.config.ollama_base_url
            or self.config.ollama_host
            or ""
        ).strip()
        if not raw_base:
            return {
                "ok": False,
                "error_kind": "ollama_unavailable",
                "status_code": None,
                "message": "Ollama base URL is not configured.",
                "models": [],
            }
        native_base = str(normalize_ollama_base_urls(raw_base).get("native_base") or "").strip().rstrip("/")
        if not native_base:
            return {
                "ok": False,
                "error_kind": "ollama_unavailable",
                "status_code": None,
                "message": "Ollama base URL is invalid.",
                "models": [],
            }
        headers = self._provider_request_headers(ollama_provider) if isinstance(ollama_provider, dict) else {}
        try:
            parsed = self._http_get_json(
                f"{native_base}/api/tags",
                timeout_seconds=float(timeout_seconds),
                headers=headers,
            )
        except urllib.error.HTTPError as exc:
            status_code = int(getattr(exc, "code", 0) or 0) or None
            return {
                "ok": False,
                "error_kind": "bad_status_code",
                "status_code": status_code,
                "message": "Unable to query Ollama /api/tags.",
                "models": [],
            }
        except TimeoutError:
            return {
                "ok": False,
                "error_kind": "timeout",
                "status_code": None,
                "message": "Timed out while querying Ollama.",
                "models": [],
            }
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError, urllib.error.URLError):
            return {
                "ok": False,
                "error_kind": "ollama_unavailable",
                "status_code": None,
                "message": "Unable to reach Ollama.",
                "models": [],
            }
        rows = parsed.get("models") if isinstance(parsed.get("models"), list) else []
        models = sorted(
            {
                str(row.get("name") or row.get("model") or "").strip()
                for row in rows
                if isinstance(row, dict) and str(row.get("name") or row.get("model") or "").strip()
            }
        )
        return {
            "ok": True,
            "error_kind": None,
            "status_code": None,
            "message": "ok",
            "models": models,
        }

    def pull_ollama_model(self, payload: dict[str, Any] | None = None) -> tuple[bool, dict[str, Any]]:
        body = payload if isinstance(payload, dict) else {}
        normalized_model = self._normalize_ollama_pull_model(body.get("model"))
        if not normalized_model:
            return False, {
                "ok": False,
                "error": "bad_request",
                "error_kind": "bad_request",
                "status_code": 400,
                "message": "model is required",
            }
        if normalized_model not in _OLLAMA_PULL_ALLOWLIST_SET:
            return False, {
                "ok": False,
                "error": "model_not_allowed",
                "error_kind": "model_not_allowed",
                "status_code": 400,
                "model": normalized_model,
                "allowlist": list(_OLLAMA_PULL_ALLOWLIST),
                "message": "Requested model is not in the local allowlist.",
            }

        start = time.monotonic()
        probe_before = self._ollama_tags_models(timeout_seconds=2.0)
        before_models = (
            {str(item).strip() for item in probe_before.get("models", []) if str(item).strip()}
            if bool(probe_before.get("ok"))
            else set()
        )
        already_present = normalized_model in before_models

        if not already_present:
            pull_timeout_seconds = float(body.get("timeout_seconds") or 1800.0)
            try:
                pull_result = self.modelops_executor.safe_runner.run(
                    ["ollama", "pull", normalized_model],
                    timeout_seconds=pull_timeout_seconds,
                )
            except FileNotFoundError:
                return False, {
                    "ok": False,
                    "error": "ollama_unavailable",
                    "error_kind": "ollama_unavailable",
                    "status_code": None,
                    "model": normalized_model,
                    "message": "Ollama CLI is not available on this host.",
                }
            except Exception as exc:
                return False, {
                    "ok": False,
                    "error": "ollama_unavailable",
                    "error_kind": "ollama_unavailable",
                    "status_code": None,
                    "model": normalized_model,
                    "message": f"Unable to pull model with Ollama ({exc.__class__.__name__}).",
                }
            if not pull_result.ok:
                error_kind = "timeout" if bool(pull_result.timed_out) else "ollama_unavailable"
                message = (
                    "Timed out while pulling model from Ollama."
                    if error_kind == "timeout"
                    else "Ollama failed to pull the requested model."
                )
                return False, {
                    "ok": False,
                    "error": error_kind,
                    "error_kind": error_kind,
                    "status_code": None,
                    "model": normalized_model,
                    "message": message,
                }

        refresh_ok, _refresh_body = self.refresh_models({"provider": "ollama"})
        duration_ms = int((time.monotonic() - start) * 1000)
        canonical_model = f"ollama:{normalized_model}"
        message = (
            f"Model {normalized_model} is already installed."
            if already_present
            else f"Pulled {normalized_model} and refreshed local model catalog."
        )
        if not refresh_ok:
            message = (
                f"{message} Catalog refresh did not complete; provider test will validate availability."
            )
        response = {
            "ok": True,
            "model": normalized_model,
            "canonical_model": canonical_model,
            "already_present": bool(already_present),
            "duration_ms": duration_ms,
            "message": message,
        }
        self._log_request("/providers/ollama/pull", True, response)
        return True, response

    def get_defaults(self) -> dict[str, Any]:
        document = self.registry_document if isinstance(self.registry_document, dict) else {}
        defaults = self._ensure_defaults(document)
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        provider_ids = {str(provider_id).strip().lower() for provider_id in providers.keys()}
        raw_chat_model = str(defaults.get("chat_model") or defaults.get("default_model") or "").strip() or None
        resolved_default_model = self._resolved_chat_default_model(
            defaults=defaults,
            models=models,
            provider_ids=provider_ids,
        )
        return {
            "routing_mode": defaults.get("routing_mode") or self._router.policy.mode,
            "default_provider": defaults.get("default_provider"),
            "chat_model": raw_chat_model,
            "embed_model": str(defaults.get("embed_model") or "").strip() or None,
            "last_chat_model": str(defaults.get("last_chat_model") or "").strip() or None,
            "default_model": raw_chat_model,
            "resolved_default_model": resolved_default_model,
            "allow_remote_fallback": bool(defaults.get("allow_remote_fallback", True)),
        }

    def _resolved_chat_default_model(
        self,
        *,
        defaults: dict[str, Any],
        models: dict[str, Any],
        provider_ids: set[str],
    ) -> str | None:
        chat_model = str(defaults.get("chat_model") or defaults.get("default_model") or "").strip() or None
        if not chat_model:
            return None
        provider_for_model = str(defaults.get("default_provider") or "").strip().lower() or None
        return self._normalize_default_model_id(
            chat_model,
            provider_for_model=provider_for_model,
            models=models,
            provider_ids=provider_ids,
        )

    def update_defaults(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        valid_modes = {
            "auto",
            "prefer_cheap",
            "prefer_best",
            "prefer_local_lowest_cost_capable",
        }

        document = self.registry_document
        defaults = self._ensure_defaults(document)
        previous_chat_model = str(defaults.get("chat_model") or defaults.get("default_model") or "").strip() or None
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        provider_ids = {str(provider_id).strip().lower() for provider_id in providers.keys()}
        provider_override = str(payload.get("default_provider") or "").strip().lower() if "default_provider" in payload else None

        if "routing_mode" in payload:
            mode = str(payload.get("routing_mode") or "").strip().lower()
            if mode not in valid_modes:
                return False, {"ok": False, "error": "invalid routing_mode"}
            defaults["routing_mode"] = mode

        if "default_provider" in payload:
            provider = provider_override or None
            if provider and provider not in self._sorted_provider_ids():
                return False, {"ok": False, "error": "default_provider not found"}
            defaults["default_provider"] = provider

        chat_model_key = "chat_model" if "chat_model" in payload else ("default_model" if "default_model" in payload else None)
        if chat_model_key is not None:
            model = str(payload.get(chat_model_key) or "").strip() or None
            if model is None:
                defaults["chat_model"] = None
                defaults["default_model"] = None
            else:
                provider_for_model = defaults.get("default_provider")
                provider_for_model = str(provider_for_model).strip().lower() if provider_for_model else None
                canonical_model = self._normalize_default_model_id(
                    model,
                    provider_for_model=provider_for_model,
                    models=models,
                    provider_ids=provider_ids,
                )
                if canonical_model is None:
                    return False, {"ok": False, "error": "default_model not found"}
                valid_default, validated_model, validation_error = validate_default_model(
                    canonical_model,
                    models,
                    purpose="chat",
                )
                if not valid_default:
                    legacy_error = (
                        {"ok": False, "error": "default_model_not_chat_capable", "error_kind": "chat_model_not_chat_capable"}
                        if chat_model_key == "default_model"
                        else {"ok": False, "error": "chat_model_not_chat_capable", "error_kind": "chat_model_not_chat_capable"}
                    )
                    details = (
                        dict(validation_error.get("details"))
                        if isinstance(validation_error, dict) and isinstance(validation_error.get("details"), dict)
                        else None
                    )
                    if details is not None:
                        legacy_error["details"] = details
                    return False, legacy_error
                defaults["chat_model"] = validated_model
                defaults["default_model"] = validated_model
        if "embed_model" in payload:
            model = str(payload.get("embed_model") or "").strip() or None
            if model is None:
                defaults["embed_model"] = None
            else:
                provider_for_model = defaults.get("default_provider")
                provider_for_model = str(provider_for_model).strip().lower() if provider_for_model else None
                canonical_model = self._normalize_default_model_id(
                    model,
                    provider_for_model=provider_for_model,
                    models=models,
                    provider_ids=provider_ids,
                )
                if canonical_model is None:
                    return False, {"ok": False, "error": "embed_model not found"}
                valid_default, validated_model, validation_error = validate_default_model(
                    canonical_model,
                    models,
                    purpose="embedding",
                )
                if not valid_default:
                    error_payload = {
                        "ok": False,
                        "error": "embed_model_not_embedding_capable",
                        "error_kind": "embed_model_not_embedding_capable",
                    }
                    details = (
                        dict(validation_error.get("details"))
                        if isinstance(validation_error, dict) and isinstance(validation_error.get("details"), dict)
                        else None
                    )
                    if details is not None:
                        error_payload["details"] = details
                    return False, error_payload
                defaults["embed_model"] = validated_model
        elif (
            "default_provider" in payload
            and defaults.get("chat_model")
            and str(defaults.get("default_provider") or "").strip()
        ):
            model_id = str(defaults.get("chat_model") or "").strip()
            if model_id and model_id in models:
                existing_provider = str((models.get(model_id) or {}).get("provider") or "").strip().lower()
                selected_provider = str(defaults.get("default_provider") or "").strip().lower()
                if existing_provider and selected_provider and existing_provider != selected_provider:
                    defaults["chat_model"] = None
                    defaults["default_model"] = None

        if "allow_remote_fallback" in payload:
            defaults["allow_remote_fallback"] = bool(payload.get("allow_remote_fallback"))

        defaults["default_model"] = str(defaults.get("chat_model") or defaults.get("default_model") or "").strip() or None
        current_chat_model = str(defaults.get("chat_model") or defaults.get("default_model") or "").strip() or None
        if current_chat_model != previous_chat_model:
            defaults["last_chat_model"] = previous_chat_model
        elif defaults.get("last_chat_model") in {"", "none"}:
            defaults["last_chat_model"] = None
        if defaults.get("embed_model") in {"", "none"}:
            defaults["embed_model"] = None
        document["defaults"] = defaults
        saved, error = self._persist_registry_document(document)
        if not saved:
            assert error is not None
            return False, error
        return True, {"ok": True, **self.get_defaults()}

    def rollback_defaults(self) -> tuple[bool, dict[str, Any]]:
        document = self.registry_document
        defaults = self._ensure_defaults(document)
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        provider_ids = {str(provider_id).strip().lower() for provider_id in providers.keys()}

        rollback_target = str(defaults.get("last_chat_model") or "").strip() or None
        if not rollback_target:
            return False, {
                "ok": False,
                "error": "no_rollback_available",
                "error_kind": "no_rollback_available",
                "message": "No previous chat model is available to roll back to.",
            }

        provider_for_model = str(defaults.get("default_provider") or "").strip().lower() or None
        canonical_target = self._normalize_default_model_id(
            rollback_target,
            provider_for_model=provider_for_model,
            models=models,
            provider_ids=provider_ids,
        )
        valid_target, validated_target, validation_error = validate_default_model(
            canonical_target,
            models,
            purpose="chat",
        )
        if not valid_target or not validated_target:
            response = {
                "ok": False,
                "error": "rollback_target_invalid",
                "error_kind": "rollback_target_invalid",
                "message": "Rollback target is not available as a chat-capable model.",
            }
            details = (
                dict(validation_error.get("details"))
                if isinstance(validation_error, dict) and isinstance(validation_error.get("details"), dict)
                else None
            )
            if details is not None:
                response["details"] = details
            return False, response

        previous_chat_model = str(defaults.get("chat_model") or defaults.get("default_model") or "").strip() or None
        ok, body = self.update_defaults({"chat_model": validated_target})
        if not ok:
            response = body if isinstance(body, dict) else {"ok": False, "error": "rollback_failed"}
            response.setdefault("error_kind", str(response.get("error") or "rollback_failed"))
            return False, response
        body = body if isinstance(body, dict) else {}
        body["rolled_back_from"] = previous_chat_model
        body["rolled_back_to"] = str(body.get("chat_model") or validated_target)
        return True, body

    @staticmethod
    def _normalize_default_model_id(
        model_value: str,
        *,
        provider_for_model: str | None,
        models: dict[str, Any],
        provider_ids: set[str],
    ) -> str | None:
        candidate = (model_value or "").strip()
        if not candidate:
            return None

        # Accept canonical full ids as-is.
        if candidate in models:
            return candidate

        # Only treat "<provider>:<name>" as canonical if the prefix is a known provider id.
        prefix = candidate.split(":", 1)[0].strip().lower() if ":" in candidate else ""
        if prefix and prefix in provider_ids:
            return None

        if not provider_for_model:
            return None

        scoped_model_id = f"{provider_for_model}:{candidate}"
        if scoped_model_id in models:
            return scoped_model_id
        return None

    @staticmethod
    def _default_refreshed_capabilities(model_name: str) -> list[str]:
        normalized_name = (model_name or "").strip().lower()
        if "embed" in normalized_name:
            return ["embedding"]
        return ["chat"]

    @staticmethod
    def _perception_event_payloads(events: list[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in events:
            rows.append(
                {
                    "kind": str(getattr(event, "kind", "") or ""),
                    "severity": str(getattr(event, "severity", "") or ""),
                    "summary": str(getattr(event, "summary", "") or ""),
                    "evidence_json": dict(getattr(event, "evidence_json", {}) or {}),
                }
            )
        return rows

    def _open_perception_db(self) -> MemoryDB | None:
        try:
            db = MemoryDB(self.config.db_path)
            schema_path = str(self._repo_root / "memory" / "schema.sql")
            db.init_schema(schema_path)
            return db
        except Exception:
            return None

    @staticmethod
    def _metrics_row_payload(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        return {
            "id": int(row.get("id") or 0),
            "ts": int(row.get("ts") or 0),
            "cpu_usage": float(row.get("cpu_usage") or 0.0),
            "cpu_freq": float(row.get("cpu_freq") or 0.0),
            "mem_used": int(row.get("mem_used") or 0),
            "mem_available": int(row.get("mem_available") or 0),
            "root_disk_used_pct": float(row.get("root_disk_used_pct") or 0.0),
            "gpu_usage": float(row.get("gpu_usage")) if row.get("gpu_usage") is not None else None,
            "gpu_mem_used": int(row.get("gpu_mem_used")) if row.get("gpu_mem_used") is not None else None,
            "gpu_temp": float(row.get("gpu_temp")) if row.get("gpu_temp") is not None else None,
        }

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _chat_autopilot_meta(self, request_started_epoch: int) -> dict[str, Any]:
        return {
            "last_notification": self._notification_store.latest_notification_summary(),
            "since_last_user_message": int(self._notification_store.count_newer_than(int(request_started_epoch))),
        }

    def _collect_authoritative_observations(self, domains: set[str]) -> dict[str, Any]:
        if not self.config.perception_enabled:
            raise RuntimeError("perception disabled")

        roots = list(self.config.perception_roots or ("/home", "/data/projects"))
        observations: dict[str, Any] = {}
        refs: dict[str, dict[str, Any]] = {}
        db = self._open_perception_db()
        fresh_snapshot: dict[str, Any] | None = None

        def _get_fresh_snapshot() -> dict[str, Any]:
            nonlocal fresh_snapshot
            if fresh_snapshot is None:
                fresh_snapshot = collect_snapshot(roots)
            return fresh_snapshot

        def _persist_snapshot(snapshot: dict[str, Any]) -> tuple[int | None, list[int], list[dict[str, Any]]]:
            events = analyze_snapshot(snapshot)
            payload_events = self._perception_event_payloads(events)
            if db is None:
                return None, [], payload_events
            snapshot_id = db.insert_metrics_snapshot(snapshot)
            event_ids: list[int] = []
            for event in events:
                event_ids.append(
                    db.insert_event(
                        int(snapshot.get("ts") or datetime.now(timezone.utc).timestamp()),
                        event.kind,
                        event.severity,
                        event.summary,
                        event.evidence_json,
                    )
                )
            return snapshot_id, event_ids, payload_events

        try:
            for domain in ("system.performance", "system.health", "system.storage"):
                if domain not in domains:
                    continue
                tool_name = {
                    "system.performance": "sys_metrics_snapshot",
                    "system.health": "sys_health_report",
                    "system.storage": "sys_inventory_summary",
                }[domain]

                if domain == "system.performance":
                    snapshot = _get_fresh_snapshot()
                    snapshot_id, event_ids, event_payloads = _persist_snapshot(snapshot)
                    payload = {
                        "ok": True,
                        "source": "fresh",
                        "snapshot": snapshot,
                        "events": event_payloads,
                        "stored": {"snapshot_id": snapshot_id, "event_ids": event_ids},
                    }
                    observations[domain] = payload
                    refs[domain] = {
                        "tool": tool_name,
                        "snapshot_id": snapshot_id,
                        "ts": self._coerce_int(snapshot.get("ts")),
                    }
                    continue

                if domain == "system.health":
                    latest = db.get_latest_metrics_snapshot() if db else None
                    source = "sqlite" if latest else "fresh"
                    snapshot: dict[str, Any] | None = None
                    if latest is None:
                        snapshot = _get_fresh_snapshot()
                        _persist_snapshot(snapshot)
                        latest = db.get_latest_metrics_snapshot() if db else None
                    payload = {
                        "ok": True,
                        "source": source,
                        "latest_metrics": self._metrics_row_payload(latest),
                        "recent_events": db.list_recent_events(limit=10) if db else [],
                        "system_health": ((snapshot or {}).get("system_health") if snapshot else None),
                    }
                    observations[domain] = payload
                    latest_metrics = payload.get("latest_metrics") if isinstance(payload.get("latest_metrics"), dict) else {}
                    refs[domain] = {
                        "tool": tool_name,
                        "snapshot_id": self._coerce_int(latest_metrics.get("id")),
                        "ts": self._coerce_int(latest_metrics.get("ts")),
                    }
                    continue

                if domain == "system.storage":
                    latest = db.get_latest_metrics_snapshot() if db else None
                    source = "sqlite" if latest else "fresh"
                    if latest:
                        snapshot = {
                            "cpu": {"freq_mhz": latest.get("cpu_freq") or 0.0, "load_avg": {"1m": 0.0}},
                            "memory": {
                                "total": 0,
                                "used": int(latest.get("mem_used") or 0),
                                "available": int(latest.get("mem_available") or 0),
                                "swap_total": 0,
                            },
                            "disk": {
                                "root": {"total": 0, "used_pct": float(latest.get("root_disk_used_pct") or 0.0)},
                                "top_dirs": [],
                            },
                            "gpu": {"available": latest.get("gpu_usage") is not None},
                        }
                    else:
                        snapshot = _get_fresh_snapshot()
                        _persist_snapshot(snapshot)
                        latest = db.get_latest_metrics_snapshot() if db else None
                    payload = {
                        "ok": True,
                        "source": source,
                        "inventory": summarize_inventory(snapshot, roots),
                    }
                    observations[domain] = payload
                    refs[domain] = {
                        "tool": tool_name,
                        "snapshot_id": self._coerce_int((latest or {}).get("id")),
                        "ts": self._coerce_int((latest or {}).get("ts")) or self._coerce_int(snapshot.get("ts")),
                    }

            return {
                "domains": sorted(domains),
                "grounding": {
                    "collected_at_ts": int(datetime.now(timezone.utc).timestamp()),
                    "observation_refs": refs,
                },
                "observations": observations,
            }
        finally:
            if db is not None:
                db.close()

    @staticmethod
    def _authoritative_tool_failure_text(domains: set[str], error: Exception) -> str:
        domain = sorted(domains)[0] if domains else "system.performance"
        tool_name = {
            "system.performance": "sys_metrics_snapshot",
            "system.health": "sys_health_report",
            "system.storage": "sys_inventory_summary",
        }.get(domain, "sys_metrics_snapshot")
        reason = " ".join(str(error).replace("?", "").split()) or "unknown error"
        return (
            "I’m not sure.\n\n"
            f"I couldn’t read local system data via {tool_name} ({reason}). "
            f"Do you want me to retry {tool_name} now?"
        )

    def _memory_v2_bootstrap_completed(self) -> bool:
        if self._memory_v2_store is None:
            return False
        return self._memory_v2_store.get_bool_state(_MEMORY_V2_BOOTSTRAP_COMPLETED_KEY)

    def _set_memory_v2_bootstrap_completed(self, *, now_ts: int) -> None:
        if self._memory_v2_store is None:
            return
        self._memory_v2_store.set_state(_MEMORY_V2_BOOTSTRAP_COMPLETED_KEY, "1", updated_at=int(now_ts))
        self._memory_v2_store.set_state(
            _MEMORY_V2_BOOTSTRAP_COMPLETED_AT_KEY,
            str(int(now_ts)),
            updated_at=int(now_ts),
        )
        self._memory_v2_store.set_state(
            _MEMORY_V2_LAST_BOOTSTRAP_TS_KEY,
            str(int(now_ts)),
            updated_at=int(now_ts),
        )

    def _memory_v2_greeted_once(self) -> bool:
        if self._memory_v2_store is None:
            return False
        return self._memory_v2_store.get_bool_state(_MEMORY_V2_GREETED_ONCE_KEY)

    def _set_memory_v2_greeted_once(self, *, now_ts: int) -> None:
        if self._memory_v2_store is None:
            return
        self._memory_v2_store.set_state(_MEMORY_V2_GREETED_ONCE_KEY, "1", updated_at=int(now_ts))

    def consume_bootstrap_greeting_if_needed(self) -> str | None:
        if not bool(self.config.memory_v2_enabled):
            return None
        if self._memory_v2_store is None:
            return None
        if not self._memory_v2_bootstrap_completed():
            return None
        if self._memory_v2_greeted_once():
            return None
        now_ts = int(time.time())
        self._set_memory_v2_greeted_once(now_ts=now_ts)
        return _BOOTSTRAP_GREETING_TEXT

    def run_memory_v2_bootstrap(
        self,
        *,
        source_ref: str,
        promote_semantic: bool,
        reason: str | None = None,
        now_ts: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        if not bool(self.config.memory_v2_enabled):
            return False, {
                "ok": False,
                "error": "memory_v2_disabled",
                "message": "memory_v2 is disabled.",
            }
        if self._memory_v2_store is None:
            return False, {
                "ok": False,
                "error": "memory_v2_store_unavailable",
                "message": "memory_v2 store is unavailable.",
            }
        run_ts = int(now_ts if now_ts is not None else int(time.time()))
        try:
            snapshot = collect_bootstrap_snapshot(self, now_ts=run_ts)
            ingest_result = ingest_bootstrap_snapshot(
                store=self._memory_v2_store,
                snapshot=snapshot,
                source_ref=str(source_ref or "bootstrap_run"),
                promote_semantic=bool(promote_semantic),
                now_ts=run_ts,
                trigger_reason=reason,
            )
            self._set_memory_v2_bootstrap_completed(now_ts=run_ts)
            if self._memory_v2_store.get_state(_MEMORY_V2_GREETED_ONCE_KEY) is None:
                self._memory_v2_store.set_state(_MEMORY_V2_GREETED_ONCE_KEY, "0", updated_at=run_ts)
            self._memory_v2_bootstrap_status = {
                "ran": True,
                "ok": True,
                "created_at_ts": int(snapshot.created_at_ts),
                "episodic_count": len(ingest_result.get("episodic_ids") or []),
                "semantic_count": len((ingest_result.get("semantic_updates") or {}).get("inserted") or []),
            }
            log_event(
                self.config.log_path,
                "memory_v2_bootstrap",
                {
                    "ok": True,
                    "created_at_ts": int(snapshot.created_at_ts),
                    "episodic_count": len(ingest_result.get("episodic_ids") or []),
                    "semantic_count": len((ingest_result.get("semantic_updates") or {}).get("inserted") or []),
                    "source_ref": str(source_ref or "bootstrap_run"),
                    "promote_semantic": bool(promote_semantic),
                },
            )
            return True, {
                "ok": True,
                "created_at_ts": int(snapshot.created_at_ts),
                "notes": list(snapshot.notes),
                "ingest": ingest_result,
            }
        except Exception as exc:
            self._memory_v2_bootstrap_status = {"ran": True, "ok": False, "reason": exc.__class__.__name__}
            log_event(
                self.config.log_path,
                "memory_v2_bootstrap",
                {
                    "ok": False,
                    "error": exc.__class__.__name__,
                    "source_ref": str(source_ref or "bootstrap_run"),
                },
            )
            return False, {
                "ok": False,
                "error": "bootstrap_failed",
                "message": f"Bootstrap failed: {exc.__class__.__name__}",
            }

    def _initialize_memory_v2_bootstrap(self) -> None:
        if not bool(self.config.memory_v2_enabled):
            self._memory_v2_bootstrap_status = {"ran": False, "ok": False, "reason": "memory_v2_disabled"}
            return
        if self._memory_v2_store is None:
            self._memory_v2_bootstrap_status = {"ran": False, "ok": False, "reason": "memory_v2_store_unavailable"}
            return
        if self._memory_v2_bootstrap_completed():
            self._memory_v2_bootstrap_status = {"ran": False, "ok": True, "reason": "already_completed"}
            return
        now_ts = int(time.time())
        try:
            snapshot = collect_bootstrap_snapshot(self, now_ts=now_ts)
            ingest_result = ingest_bootstrap_snapshot(
                store=self._memory_v2_store,
                snapshot=snapshot,
                source_ref="agent_runtime_startup",
            )
            self._set_memory_v2_bootstrap_completed(now_ts=now_ts)
            if self._memory_v2_store.get_state(_MEMORY_V2_GREETED_ONCE_KEY) is None:
                self._memory_v2_store.set_state(_MEMORY_V2_GREETED_ONCE_KEY, "0", updated_at=now_ts)
            self._memory_v2_bootstrap_status = {
                "ran": True,
                "ok": True,
                "created_at_ts": int(snapshot.created_at_ts),
                "episodic_count": len(ingest_result.get("episodic_ids") or []),
                "semantic_count": len(ingest_result.get("semantic_ids") or []),
            }
            log_event(
                self.config.log_path,
                "memory_v2_bootstrap",
                {
                    "ok": True,
                    "created_at_ts": int(snapshot.created_at_ts),
                    "episodic_count": len(ingest_result.get("episodic_ids") or []),
                    "semantic_count": len(ingest_result.get("semantic_ids") or []),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive bootstrap path
            self._memory_v2_bootstrap_status = {"ran": True, "ok": False, "reason": exc.__class__.__name__}
            log_event(
                self.config.log_path,
                "memory_v2_bootstrap",
                {
                    "ok": False,
                    "error": exc.__class__.__name__,
                },
            )

    @staticmethod
    def _normalize_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
        raw = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(raw, list):
            return []
        messages: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip() or "user"
            content = str(item.get("content") or "")
            messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _memory_message_content_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        values: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str) and text_value.strip():
                values.append(text_value.strip())
        return " ".join(values)

    @staticmethod
    def _extract_memory_query_text(payload: dict[str, Any], *, intent: str) -> str:
        if not isinstance(payload, dict):
            return ""
        for key in ("text", "message", "content", "input", "query"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            return ""
        for row in reversed(raw_messages):
            if not isinstance(row, dict):
                continue
            role = str(row.get("role") or "").strip().lower() or "user"
            if role != "user":
                continue
            content = AgentRuntime._memory_message_content_text(row.get("content"))
            if content:
                text = content.strip()
                lowered = text.lower()
                for prefix in ("/ask ", "/done ", "/chat "):
                    if lowered.startswith(prefix):
                        text = text[len(prefix):].strip()
                        lowered = text.lower()
                        break
                if lowered in {"/ask", "/done", "/chat"}:
                    return ""
                return text
        return ""

    @staticmethod
    def _memory_query_tags(payload: dict[str, Any], *, intent: str) -> dict[str, str]:
        tags: dict[str, str] = {}
        if isinstance(payload, dict):
            raw_tags = payload.get("memory_tags")
            if isinstance(raw_tags, dict):
                for key, value in sorted(raw_tags.items(), key=lambda item: str(item[0])):
                    key_norm = str(key).strip().lower()
                    value_norm = str(value).strip()
                    if key_norm and value_norm:
                        tags[key_norm] = value_norm
            project = str(payload.get("project") or payload.get("project_tag") or "").strip()
            if project:
                tags["project"] = project
        return tags

    def build_memory_context_for_payload(
        self,
        payload: dict[str, Any],
        *,
        intent: str,
        trace_id: str | None = None,
        now_ts: int | None = None,
    ) -> dict[str, Any] | None:
        if not bool(self.config.memory_v2_enabled):
            return None
        if self._memory_v2_store is None:
            return None
        query_text = self._extract_memory_query_text(payload, intent=intent)
        if not query_text:
            return {
                "selected_ids": [],
                "levels": {
                    MemoryLevel.EPISODIC.value: [],
                    MemoryLevel.SEMANTIC.value: [],
                    MemoryLevel.PROCEDURAL.value: [],
                },
                "debug": {
                    "query": {"norm": "", "tokens": [], "entities": [], "tags": {}},
                    "selected_ids": [],
                    "selected": [],
                    "skipped": [],
                },
                "merged_context_text": "",
            }
        try:
            selection = with_built_context(
                select_memory(
                    MemoryQuery(
                        text=query_text,
                        tags=self._memory_query_tags(payload, intent=intent),
                        now_ts=now_ts if now_ts is not None else int(time.time()),
                    ),
                    self._memory_v2_store,
                )
            )
            levels = {
                level.value: [item.id for item in selection.items_by_level.get(level, [])]
                for level in (
                    MemoryLevel.EPISODIC,
                    MemoryLevel.SEMANTIC,
                    MemoryLevel.PROCEDURAL,
                )
            }
            selected_ids = [item_id for level in levels.values() for item_id in level]
            return {
                "selected_ids": selected_ids,
                "levels": levels,
                "debug": selection.debug,
                "merged_context_text": selection.merged_context_text,
            }
        except Exception as exc:  # pragma: no cover - defensive path
            log_event(
                self.config.log_path,
                "memory_v2_select_failed",
                {
                    "error": exc.__class__.__name__,
                    "intent": str(intent or "").strip().lower() or "chat",
                    "trace_id": str(trace_id or "").strip() or None,
                },
            )
            return None

    def _record_memory_event(
        self,
        *,
        text: str,
        tags: dict[str, str] | None = None,
        source_kind: str,
        source_ref: str,
    ) -> None:
        if not bool(self.config.memory_v2_enabled):
            return
        if self._memory_v2_store is None:
            return
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return
        try:
            self._memory_v2_store.append_episodic_event(
                text=normalized_text,
                tags=tags or {},
                source_kind=source_kind,
                source_ref=source_ref,
                pinned=False,
            )
        except Exception as exc:  # pragma: no cover - defensive logging only
            log_event(
                self.config.log_path,
                "memory_v2_event_write_failed",
                {
                    "error": exc.__class__.__name__,
                    "source_kind": str(source_kind or "").strip() or "unknown",
                    "source_ref": str(source_ref or "").strip() or None,
                },
            )

    def _value_policy(self, name: str) -> ValuePolicy:
        policy_name = str(name or "default").strip().lower() or "default"
        raw = self.config.premium_policy if policy_name == "premium" else self.config.default_policy
        return normalize_policy(raw if isinstance(raw, dict) else {}, name=policy_name)

    def _model_policy_candidates(self) -> list[dict[str, Any]]:
        document = self.registry_document if isinstance(self.registry_document, dict) else {}
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        snapshot = self._router.doctor_snapshot()
        snapshot_rows = snapshot.get("models") if isinstance(snapshot.get("models"), list) else []
        snapshot_by_id = {
            str(row.get("id") or "").strip(): row
            for row in snapshot_rows
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        candidates: list[dict[str, Any]] = []
        for model_id, model_row in sorted(models.items()):
            if not isinstance(model_row, dict):
                continue
            provider_id = str(model_row.get("provider") or "").strip().lower()
            provider_row = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
            if not provider_id or not isinstance(provider_row, dict):
                continue
            pricing = model_row.get("pricing") if isinstance(model_row.get("pricing"), dict) else {}
            snapshot_row = snapshot_by_id.get(str(model_id))
            health = snapshot_row.get("health") if isinstance(snapshot_row, dict) and isinstance(snapshot_row.get("health"), dict) else {}
            candidates.append(
                {
                    "model_id": str(model_id).strip(),
                    "provider": provider_id,
                    "local": bool(provider_row.get("local", False)),
                    "enabled": bool(model_row.get("enabled", False)),
                    "available": bool(model_row.get("available", False)),
                    "routable": bool(snapshot_row.get("routable", False)) if isinstance(snapshot_row, dict) else False,
                    "quality_rank": int(model_row.get("quality_rank", 0) or 0),
                    "price_in": pricing.get("input_per_million_tokens"),
                    "price_out": pricing.get("output_per_million_tokens"),
                    "context_tokens": (
                        int(model_row.get("max_context_tokens"))
                        if model_row.get("max_context_tokens") is not None
                        else None
                    ),
                    "health_status": str(health.get("status") or "unknown").strip().lower(),
                }
            )
        return candidates

    def _rank_models_for_policy(
        self,
        *,
        policy: ValuePolicy,
        allow_remote_fallback: bool,
    ) -> tuple[list[Any], list[Any]]:
        return rank_candidates_by_utility(
            self._model_policy_candidates(),
            policy=policy,
            allow_remote_fallback=allow_remote_fallback,
        )

    def _premium_override_active(self, *, now_epoch: int) -> bool:
        if self._premium_override_once:
            return True
        until = int(self._premium_override_until_ts or 0)
        if until and now_epoch <= until:
            return True
        if until and now_epoch > until:
            self._premium_override_until_ts = None
        return False

    def _persist_premium_over_cap_prompt(
        self,
        *,
        baseline_model: str,
        premium_model: str,
        premium_cost: float,
        premium_cap: float,
    ) -> str:
        now_epoch = int(time.time())
        issue_hash_payload = {
            "issue_code": "premium_over_cap",
            "baseline_model": baseline_model,
            "premium_model": premium_model,
            "premium_cost": round(float(premium_cost), 6),
            "premium_cap": round(float(premium_cap), 6),
        }
        issue_hash = hashlib.sha256(
            json.dumps(issue_hash_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        choices = [
            {"id": "continue_baseline", "label": "Continue baseline", "recommended": True},
            {"id": "upgrade_once", "label": "Upgrade once", "recommended": False},
            {"id": "set_premium_1h", "label": "Set premium for 1h", "recommended": False},
        ]
        self._llm_fixit_store.save(
            {
                "active": True,
                "issue_hash": issue_hash,
                "issue_code": "premium_over_cap",
                "step": "awaiting_choice",
                "question": "Reply 1, 2, or 3.",
                "choices": choices,
                "pending_plan": [],
                "pending_confirm_token": None,
                "pending_created_ts": None,
                "pending_expires_ts": None,
                "pending_issue_code": None,
                "last_prompt_ts": now_epoch,
            }
        )
        return "\n".join(
            [
                f"Premium escalation is over the cost cap ({premium_cost:.2f} > {premium_cap:.2f} per 1M tokens).",
                "1) Continue baseline.",
                "2) Upgrade once.",
                "3) Set premium for 1h.",
                "Reply 1, 2, or 3.",
            ]
        )

    @staticmethod
    def _last_user_message_text(messages: list[dict[str, str]]) -> str:
        for item in reversed(messages):
            if str(item.get("role") or "").strip().lower() == "user":
                return str(item.get("content") or "")
        if messages:
            return str((messages[-1] or {}).get("content") or "")
        return ""

    def chat(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        request_started_epoch = int(time.time())
        trace_id = str(payload.get("trace_id") or "").strip() or f"chat-{request_started_epoch}"
        messages = self._normalize_messages(payload)
        if not messages:
            return False, {"ok": False, "error": "messages must be a non-empty list"}

        defaults = self.get_defaults()
        explicit_model_override = bool(str(payload.get("model") or "").strip())
        explicit_provider_override = bool(str(payload.get("provider") or "").strip())
        model_override = (
            str(payload.get("model") or "").strip()
            or str(defaults.get("chat_model") or "").strip()
            or str(defaults.get("default_model") or "").strip()
            or None
        )
        provider_override = str(payload.get("provider") or "").strip().lower() or defaults.get("default_provider")
        allow_remote_fallback = bool(defaults.get("allow_remote_fallback", True))
        explicit_require_tools = "require_tools" in payload
        require_tools = bool(payload.get("require_tools"))
        memory_context_text = str(payload.get("memory_context_text") or "").strip()
        memory_prefix_messages: list[dict[str, str]] = []
        if memory_context_text:
            memory_prefix_messages = [{"role": "system", "content": memory_context_text}]
        routed_messages = [*memory_prefix_messages, *messages]

        last_user_text = self._last_user_message_text(messages)
        selection_logged = False
        selection_reason = (
            "explicit_override"
            if explicit_model_override or explicit_provider_override
            else "default_policy"
        )

        def _log_selection_once(
            *,
            provider: str | None,
            model: str | None,
            reason: str,
            fallback_used: bool = False,
        ) -> None:
            nonlocal selection_logged
            if selection_logged:
                return
            self._safe_log_event(
                "llm.selection",
                {
                    "trace_id": trace_id,
                    "selected_provider": str(provider or "").strip().lower() or None,
                    "selected_model": str(model or "").strip() or None,
                    "selection_reason": str(reason or "").strip().lower() or "default_policy",
                    "fallback_used": bool(fallback_used),
                },
            )
            selection_logged = True

        # Default policy: keep current default unless policy allowlist/cap disallows it.
        baseline_policy = self._value_policy("default")
        default_allowed, default_rejected = self._rank_models_for_policy(
            policy=baseline_policy,
            allow_remote_fallback=allow_remote_fallback,
        )
        default_by_id = {str(row.model_id): row for row in [*default_allowed, *default_rejected]}
        default_model_id = str(model_override or "").strip()
        default_score = default_by_id.get(default_model_id)
        if (
            (not explicit_model_override)
            and default_model_id
            and default_score is not None
            and not bool(default_score.allowed)
            and default_allowed
        ):
            selected_default = default_allowed[0]
            model_override = str(selected_default.model_id)
            provider_override = str(selected_default.provider)
            default_score = selected_default
            selection_reason = "policy_replaced_default"
        elif default_score is None and default_allowed and (not explicit_model_override):
            selected_default = default_allowed[0]
            model_override = str(selected_default.model_id)
            provider_override = str(selected_default.provider)
            default_score = selected_default
            selection_reason = "policy_selected_default"

        escalation_reasons = detect_premium_escalation_triggers(user_text=last_user_text, payload=payload)
        premium_selected = None
        if escalation_reasons and not explicit_model_override and not explicit_provider_override:
            premium_policy = self._value_policy("premium")
            premium_allowed, premium_rejected = self._rank_models_for_policy(
                policy=premium_policy,
                allow_remote_fallback=allow_remote_fallback,
            )
            premium_unbounded = ValuePolicy(
                name=premium_policy.name,
                cost_cap_per_1m=1_000_000.0,
                allowlist=premium_policy.allowlist,
                quality_weight=premium_policy.quality_weight,
                price_weight=premium_policy.price_weight,
                latency_weight=premium_policy.latency_weight,
                instability_weight=premium_policy.instability_weight,
            )
            premium_no_cap_allowed, _premium_no_cap_rejected = self._rank_models_for_policy(
                policy=premium_unbounded,
                allow_remote_fallback=allow_remote_fallback,
            )
            baseline_utility = float(default_score.utility) if default_score is not None else -10_000.0
            top_premium = next(
                (
                    row
                    for row in premium_allowed
                    if str(row.model_id) != str(model_override or "")
                ),
                None,
            )
            top_no_cap = next(
                (
                    row
                    for row in premium_no_cap_allowed
                    if str(row.model_id) != str(model_override or "")
                ),
                None,
            )
            override_active = self._premium_override_active(now_epoch=request_started_epoch)
            if top_no_cap is not None and (
                float(top_no_cap.expected_cost_per_1m) > float(premium_policy.cost_cap_per_1m)
            ):
                if override_active and (float(top_no_cap.utility) > baseline_utility):
                    premium_selected = top_no_cap
                elif not override_active:
                    prompt = self._persist_premium_over_cap_prompt(
                        baseline_model=str(model_override or ""),
                        premium_model=str(top_no_cap.model_id),
                        premium_cost=float(top_no_cap.expected_cost_per_1m),
                        premium_cap=float(premium_policy.cost_cap_per_1m),
                    )
                    _log_selection_once(
                        provider=str(provider_override or ""),
                        model=str(model_override or ""),
                        reason="premium_over_cap_confirmation_required",
                        fallback_used=False,
                    )
                    response = {
                        "ok": True,
                        "assistant": {"role": "assistant", "content": prompt},
                        "meta": {
                            "provider": provider_override,
                            "model": model_override,
                            "fallback_used": False,
                            "attempts": [],
                            "duration_ms": 0,
                            "error": None,
                            "autopilot": self._chat_autopilot_meta(request_started_epoch),
                            "selection_policy": {
                                "mode": "premium_over_cap",
                                "baseline_model": str(model_override or ""),
                                "premium_candidate": str(top_no_cap.model_id),
                                "premium_cost_per_1m": float(top_no_cap.expected_cost_per_1m),
                                "premium_cap_per_1m": float(premium_policy.cost_cap_per_1m),
                                "escalation_reasons": list(escalation_reasons),
                            },
                        },
                    }
                    self._log_request("/chat", True, response["meta"])
                    return True, response
            if premium_selected is None and top_premium is not None and float(top_premium.utility) > baseline_utility:
                premium_selected = top_premium
            if premium_selected is not None:
                model_override = str(premium_selected.model_id)
                provider_override = str(premium_selected.provider)
                selection_reason = "premium_escalation"
                if self._premium_override_once:
                    self._premium_override_once = False

        if not explicit_require_tools:
            domains = classify_authoritative_domain(last_user_text)
            if domains and not has_local_observations_block(last_user_text):
                try:
                    local_observations = self._collect_authoritative_observations(domains)
                except Exception as exc:
                    text = self._authoritative_tool_failure_text(domains, exc)
                    _log_selection_once(
                        provider="tool_gate",
                        model=None,
                        reason="authoritative_tool_failure",
                        fallback_used=True,
                    )
                    response = {
                        "ok": True,
                        "assistant": {"role": "assistant", "content": text},
                        "meta": {
                            "provider": "tool_gate",
                            "model": None,
                            "fallback_used": False,
                            "attempts": [],
                            "duration_ms": 0,
                            "error": "authoritative_tool_failure",
                            "autopilot": self._chat_autopilot_meta(request_started_epoch),
                        },
                    }
                    self._log_request("/chat", True, response["meta"])
                    return True, response

                observations_json = json.dumps(local_observations, ensure_ascii=True, sort_keys=True)
                routed_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Use LOCAL_OBSERVATIONS as authoritative local evidence. "
                            "Do not invent system facts.\n\n"
                            f"LOCAL_OBSERVATIONS\n{observations_json}"
                        ),
                    },
                    *memory_prefix_messages,
                    *messages,
                ]
                require_tools = True

        result = self._router.chat(
            routed_messages,
            purpose=str(payload.get("purpose") or "chat"),
            task_type=str(payload.get("task_type") or payload.get("purpose") or "chat"),
            provider_override=provider_override,
            model_override=model_override,
            require_tools=require_tools,
            require_json=bool(payload.get("require_json")),
            require_vision=bool(payload.get("require_vision")),
            min_context_tokens=int(payload.get("min_context_tokens") or 0) or None,
            timeout_seconds=float(payload.get("timeout_seconds") or 0) or None,
            metadata={
                "trace_id": trace_id,
                "selection_reason": selection_reason,
            },
        )
        _log_selection_once(
            provider=str(result.get("provider") or provider_override or ""),
            model=str(result.get("model") or model_override or ""),
            reason=selection_reason if bool(result.get("ok")) else "routing_failure",
            fallback_used=bool(result.get("fallback_used")),
        )

        response = {
            "ok": bool(result.get("ok")),
            "assistant": {
                "role": "assistant",
                "content": result.get("text") or "",
            },
            "meta": {
                "provider": result.get("provider"),
                "model": result.get("model"),
                "route": "chat",
                "source_surface": str(payload.get("source_surface") or "api"),
                "fallback_used": bool(result.get("fallback_used")),
                "attempts": result.get("attempts") or [],
                "duration_ms": int(result.get("duration_ms") or 0),
                "error": result.get("error_class"),
                "autopilot": self._chat_autopilot_meta(request_started_epoch),
                "selection_policy": {
                    "default_model": str(defaults.get("chat_model") or defaults.get("default_model") or ""),
                    "selected_model": str(model_override or ""),
                    "selected_provider": str(provider_override or ""),
                    "escalation_reasons": list(escalation_reasons),
                    "premium_selected": str(getattr(premium_selected, "model_id", "") or ""),
                },
            },
        }
        self._log_request("/chat", bool(result.get("ok")), response["meta"])
        return bool(result.get("ok")), response

    @staticmethod
    def _http_get_json(
        url: str,
        timeout_seconds: float = 4.0,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        req = urllib.request.Request(url, method="GET", headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw or "{}")
        if isinstance(parsed, dict):
            return parsed
        return {}

    @staticmethod
    def _http_post_json(
        url: str,
        *,
        payload: dict[str, Any],
        timeout_seconds: float = 4.0,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        req_headers = {"Content-Type": "application/json"}
        if isinstance(headers, dict):
            req_headers.update({str(key): str(value) for key, value in headers.items() if str(key).strip()})
        req = urllib.request.Request(url, method="POST", headers=req_headers, data=body)
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw or "{}")
        if isinstance(parsed, dict):
            return parsed
        return {}

    def refresh_models(self, payload: dict[str, Any] | None = None) -> tuple[bool, dict[str, Any]]:
        payload = payload or {}
        document = copy.deepcopy(self.registry_document)
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models = document.get("models") if isinstance(document.get("models"), dict) else {}

        provider_filter = str(payload.get("provider") or "").strip().lower()
        if provider_filter and provider_filter not in providers:
            return False, {"ok": False, "error": "provider not found"}

        refreshed: dict[str, list[str]] = {}
        inventory: dict[str, dict[str, Any]] = {}
        changed = False
        modified_ids: set[str] = set()

        for provider_id, provider_payload in sorted(providers.items()):
            if not isinstance(provider_payload, dict):
                continue
            if provider_filter and provider_id != provider_filter:
                continue
            if not bool(provider_payload.get("enabled", True)):
                continue

            base_url = str(provider_payload.get("base_url") or "").rstrip("/")
            if not base_url:
                continue

            request_headers = self._provider_request_headers(provider_payload)
            is_local = bool(provider_payload.get("local", False))

            names_from_v1: set[str] = set()
            v1_ok = False

            names_from_tags: set[str] = set()
            tags_ok = False

            if is_local:
                resolved_base_url = base_url
                if str(provider_id).strip().lower() == "ollama":
                    resolved_base_url = str(
                        normalize_ollama_base_urls(base_url).get("native_base") or base_url
                    ).strip().rstrip("/")
                try:
                    parsed = self._http_get_json(resolved_base_url + "/api/tags", headers=request_headers)
                    tags = parsed.get("models") if isinstance(parsed.get("models"), list) else []
                    for item in tags:
                        if not isinstance(item, dict):
                            continue
                        model_name = str(item.get("name") or "").strip()
                        if model_name:
                            names_from_tags.add(model_name)
                    tags_ok = True
                except (OSError, TimeoutError, ValueError, UnicodeError, json.JSONDecodeError, urllib.error.URLError):
                    names_from_tags = set()
                    tags_ok = False
            else:
                try:
                    parsed = self._http_get_json(base_url + "/v1/models", headers=request_headers)
                    data = parsed.get("data") if isinstance(parsed.get("data"), list) else []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        model_name = str(item.get("id") or "").strip()
                        if model_name:
                            names_from_v1.add(model_name)
                    v1_ok = True
                except (OSError, TimeoutError, ValueError, UnicodeError, json.JSONDecodeError, urllib.error.URLError):
                    names_from_v1 = set()
                    v1_ok = False

            refreshed[provider_id] = []
            discovered_names = names_from_tags if is_local else names_from_v1
            inventory[provider_id] = {
                "authoritative": bool(tags_ok if is_local else v1_ok),
                "models": sorted(discovered_names),
            }
            for model_name in sorted(discovered_names):
                model_id = f"{provider_id}:{model_name}"
                refreshed[provider_id].append(model_id)
                existing = models.get(model_id) if isinstance(models.get(model_id), dict) else {}
                if is_local and tags_ok:
                    available = bool(model_name in names_from_tags)
                elif (not is_local) and v1_ok:
                    available = bool(model_name in names_from_v1)
                else:
                    available = bool(existing.get("available", True))

                existing_capabilities = [
                    str(item).strip().lower()
                    for item in (existing.get("capabilities") or [])
                    if str(item).strip()
                ]
                inferred_capabilities = capability_list_from_inference(
                    infer_capabilities_from_catalog(
                        provider_id,
                        {
                            "id": model_id,
                            "provider_id": provider_id,
                            "model": model_name,
                            "capabilities": existing_capabilities,
                        },
                    )
                )
                if existing_capabilities:
                    capabilities = inferred_capabilities
                else:
                    capabilities = inferred_capabilities or self._default_refreshed_capabilities(model_name)

                next_payload = {
                    **existing,
                    "provider": provider_id,
                    "model": model_name,
                    "capabilities": capabilities,
                    "quality_rank": int(existing.get("quality_rank", 2) or 2),
                    "cost_rank": int(existing.get("cost_rank", 0) or 0),
                    "default_for": list(existing.get("default_for") or ["chat"]),
                    "enabled": bool(existing.get("enabled", True)),
                    "available": available,
                    "pricing": existing.get("pricing")
                    or {
                        "input_per_million_tokens": None,
                        "output_per_million_tokens": None,
                    },
                    "max_context_tokens": existing.get("max_context_tokens"),
                }
                if next_payload != existing:
                    changed = True
                    modified_ids.add(f"model:{model_id}")
                models[model_id] = next_payload

            # Quarantine stale entries when listing endpoint is authoritative.
            for model_id, model_payload in sorted(list(models.items())):
                if not isinstance(model_payload, dict):
                    continue
                if str(model_payload.get("provider") or "").strip().lower() != provider_id:
                    continue
                model_name = str(model_payload.get("model") or "").strip()
                if not model_name:
                    continue
                if is_local and tags_ok and model_name in names_from_tags:
                    continue
                if not is_local and v1_ok and model_name in names_from_v1:
                    continue
                if is_local and not tags_ok:
                    continue
                if not is_local and not v1_ok:
                    continue
                if not bool(model_payload.get("available", True)):
                    continue
                models[model_id] = {
                    **model_payload,
                    "available": False,
                }
                changed = True
                modified_ids.add(f"model:{model_id}")

        document["models"] = models
        if not changed:
            return True, {
                "ok": True,
                "changed": False,
                "refreshed": refreshed,
                "inventory": inventory,
                "modified_ids": [],
                "models": self.models().get("models"),
            }
        saved, error = self._persist_registry_document(document)
        if not saved:
            assert error is not None
            return False, error
        return True, {
            "ok": True,
            "changed": True,
            "refreshed": refreshed,
            "inventory": inventory,
            "modified_ids": sorted(modified_ids),
            "models": self.models().get("models"),
        }

    @staticmethod
    def _catalog_model_map(catalog_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        providers = catalog_state.get("providers") if isinstance(catalog_state.get("providers"), dict) else {}
        output: dict[str, dict[str, Any]] = {}
        for provider_id in sorted(providers.keys()):
            row = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
            models = row.get("models") if isinstance(row.get("models"), list) else []
            for model_row in models:
                if not isinstance(model_row, dict):
                    continue
                model_id = str(model_row.get("id") or "").strip()
                if not model_id:
                    continue
                output[model_id] = model_row
        return output

    @staticmethod
    def _catalog_diff(before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, Any]:
        before_models = AgentRuntime._catalog_model_map(before_state)
        after_models = AgentRuntime._catalog_model_map(after_state)
        lines: list[str] = []
        added = 0
        removed = 0
        changed = 0
        for model_id in sorted(set(before_models.keys()) | set(after_models.keys())):
            before_row = before_models.get(model_id)
            after_row = after_models.get(model_id)
            if before_row is None and after_row is not None:
                added += 1
                lines.append(f"Catalog: added {model_id}")
                continue
            if before_row is not None and after_row is None:
                removed += 1
                lines.append(f"Catalog: removed {model_id}")
                continue
            assert before_row is not None and after_row is not None
            changed_fields: list[str] = []
            for field in (
                "capabilities",
                "max_context_tokens",
                "input_cost_per_million_tokens",
                "output_cost_per_million_tokens",
                "source",
            ):
                if before_row.get(field) != after_row.get(field):
                    changed_fields.append(field)
            if changed_fields:
                changed += 1
                lines.append(f"Catalog: updated {model_id} ({','.join(changed_fields)})")
        lines.sort()
        return {
            "added": added,
            "removed": removed,
            "changed": changed,
            "lines": lines,
            "notable_changes": lines[:3],
        }

    def _sync_catalog_into_registry(self) -> tuple[bool, list[str]]:
        document = copy.deepcopy(self.registry_document)
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        changed = False
        modified_ids: set[str] = set()
        catalog_rows = self._catalog_store.all_models(limit=10000)
        for row in catalog_rows:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id") or "").strip()
            provider_id = str(row.get("provider_id") or "").strip().lower()
            model_name = str(row.get("model") or "").strip()
            if not model_id or not provider_id or not model_name:
                continue
            provider_payload = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else None
            if not isinstance(provider_payload, dict):
                continue
            existing = models.get(model_id) if isinstance(models.get(model_id), dict) else {}
            capabilities = [
                str(item).strip().lower()
                for item in (row.get("capabilities") or [])
                if str(item).strip()
            ] or ["chat"]
            default_for = list(existing.get("default_for") or (["chat"] if "chat" in capabilities else ["embedding"]))
            pricing_payload = {
                "input_per_million_tokens": row.get("input_cost_per_million_tokens"),
                "output_per_million_tokens": row.get("output_cost_per_million_tokens"),
            }
            next_payload = {
                **existing,
                "provider": provider_id,
                "model": model_name,
                "capabilities": capabilities,
                "quality_rank": int(existing.get("quality_rank", 2) or 2),
                "cost_rank": int(existing.get("cost_rank", 0) or 0),
                "default_for": default_for,
                "enabled": bool(existing.get("enabled", True)),
                "available": bool(existing.get("available", True)),
                "pricing": pricing_payload,
                "max_context_tokens": row.get("max_context_tokens"),
            }
            if next_payload != existing:
                changed = True
                modified_ids.add(f"model:{model_id}")
                models[model_id] = next_payload

        if not changed:
            return False, []
        document["models"] = models
        saved, error = self._persist_registry_document(document)
        if not saved:
            _ = error
            return False, []
        return True, sorted(modified_ids)

    def run_llm_catalog_refresh(
        self,
        *,
        trigger: str = "manual",
        provider_filter: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        now_epoch = int(time.time())
        before_state = self._catalog_store.snapshot()
        document = self.registry_document if isinstance(self.registry_document, dict) else {}
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        target_provider = str(provider_filter or "").strip().lower() or None
        provider_results: dict[str, Any] = {}
        for provider_id in sorted(providers.keys()):
            if target_provider and provider_id != target_provider:
                continue
            provider_payload = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
            if not isinstance(provider_payload, dict):
                continue
            if not bool(provider_payload.get("enabled", True)):
                continue
            resolved_headers = self._provider_request_headers(provider_payload)
            catalog_cfg = {
                "base_url": provider_payload.get("base_url"),
                "local": bool(provider_payload.get("local", False)),
                "api_key_source": provider_payload.get("api_key_source"),
                "resolved_headers": resolved_headers,
                "timeout_seconds": 6.0,
            }
            result = fetch_provider_catalog(provider_id, catalog_cfg, catalog_http_get_json_with_policy)
            provider_results[provider_id] = {
                "ok": bool(result.get("ok")),
                "source": result.get("source"),
                "error_kind": result.get("error_kind"),
                "models_count": len(result.get("models") if isinstance(result.get("models"), list) else []),
            }
            self._catalog_store.update_provider_result(provider_id, result, now_epoch=now_epoch)

        after_state = self._catalog_store.snapshot()
        catalog_diff = self._catalog_diff(before_state, after_state)
        registry_changed, modified_ids = self._sync_catalog_into_registry()
        changed_any = bool(catalog_diff.get("lines")) or bool(registry_changed)
        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor="system" if trigger == "scheduler" else "user",
            action="llm.catalog.refresh",
            params={
                "trigger": trigger,
                "provider_filter": target_provider,
                "counts": {
                    "added": int(catalog_diff.get("added") or 0),
                    "removed": int(catalog_diff.get("removed") or 0),
                    "changed": int(catalog_diff.get("changed") or 0),
                },
                "modified_ids": modified_ids,
                "provider_results": provider_results,
            },
            decision="allow",
            reason="refresh_completed" if changed_any else "no_changes",
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        return True, {
            "ok": True,
            "trigger": trigger,
            "changed": changed_any,
            "catalog_changed": bool(catalog_diff.get("lines")),
            "registry_changed": bool(registry_changed),
            "counts": {
                "added": int(catalog_diff.get("added") or 0),
                "removed": int(catalog_diff.get("removed") or 0),
                "changed": int(catalog_diff.get("changed") or 0),
            },
            "notable_changes": list(catalog_diff.get("notable_changes") or []),
            "modified_ids": modified_ids,
            "provider_results": provider_results,
        }

    def llm_catalog(self, *, provider_id: str | None = None, limit: int = 200) -> dict[str, Any]:
        rows = self._catalog_store.all_models(provider_id=provider_id, limit=max(1, int(limit)))
        return {
            "ok": True,
            "provider_id": str(provider_id or "").strip().lower() or None,
            "count": len(rows),
            "models": rows,
        }

    def llm_catalog_status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": self._catalog_store.status(),
        }

    def llm_capabilities_reconcile_plan(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or "user")
        plan = build_capabilities_reconcile_plan(
            self.registry_document,
            self._catalog_store.snapshot(),
        )
        modified_ids = self._plan_modified_ids(plan)
        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor=actor,
            action="llm.capabilities.reconcile.plan",
            params={
                "impact": plan.get("impact"),
                "modified_ids": modified_ids,
            },
            decision="allow",
            reason="planned",
            dry_run=True,
            outcome="planned",
            error_kind=None,
            duration_ms=duration_ms,
        )
        return True, {
            "ok": True,
            "plan": plan,
            "modified_ids": modified_ids,
        }

    def llm_capabilities_reconcile_apply(
        self,
        payload: dict[str, Any],
        *,
        trigger: str = "manual",
    ) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or ("system" if trigger == "scheduler" else "user"))
        confirm = bool(payload.get("confirm", False))
        plan = build_capabilities_reconcile_plan(
            self.registry_document,
            self._catalog_store.snapshot(),
        )
        modified_ids = self._plan_modified_ids(plan)
        decision = self._modelops_permission_decision(
            "llm.capabilities.reconcile.apply",
            params={},
            estimated_download_bytes=0,
            estimated_cost=None,
            risk_level="low",
            dry_run=False,
        )
        scheduler_auto_policy = compute_capabilities_reconcile_apply_policy(self)
        scheduler_auto_allow = trigger == "scheduler" and bool(scheduler_auto_policy.get("allow_apply_effective"))
        effective_allow = bool(decision.get("allow")) or bool(scheduler_auto_allow)

        if not effective_allow:
            duration_ms = int((time.monotonic() - start) * 1000)
            reason = str(decision.get("reason") or "policy_deny")
            self.audit_log.append(
                actor=actor,
                action="llm.capabilities.reconcile.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="deny",
                reason=reason,
                dry_run=False,
                outcome="blocked",
                error_kind=reason,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.capabilities.reconcile.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason=reason,
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {
                "ok": False,
                "error": reason,
                "plan": plan,
                "modified_ids": modified_ids,
            }

        if bool(decision.get("requires_confirmation")) and not scheduler_auto_allow and not confirm:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.capabilities.reconcile.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="deny",
                reason="confirmation_required",
                dry_run=False,
                outcome="blocked",
                error_kind="confirmation_required",
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.capabilities.reconcile.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason="confirmation_required",
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {
                "ok": False,
                "error": "confirmation_required",
                "plan": plan,
                "modified_ids": modified_ids,
            }

        if not any(isinstance(item, dict) for item in (plan.get("changes") or [])):
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.capabilities.reconcile.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="allow",
                reason="no_changes",
                dry_run=False,
                outcome="noop",
                error_kind=None,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.capabilities.reconcile.apply",
                actor=actor,
                decision="allow",
                outcome="noop",
                reason="no_changes",
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return True, {
                "ok": True,
                "applied": False,
                "plan": plan,
                "modified_ids": modified_ids,
            }

        saved, txn_meta = self._persist_registry_document_transactional(
            lambda current: apply_capabilities_reconcile_plan(current, plan)
        )
        if not saved:
            error = txn_meta
            duration_ms = int((time.monotonic() - start) * 1000)
            reason = str(error.get("error") or "registry_write_failed")
            self.audit_log.append(
                actor=actor,
                action="llm.capabilities.reconcile.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": error.get("snapshot_id"),
                    "resulting_registry_hash": None,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="allow",
                reason=reason,
                dry_run=False,
                outcome="failed",
                error_kind=reason,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.capabilities.reconcile.apply",
                actor=actor,
                decision="allow",
                outcome="failed",
                reason=reason,
                trigger=trigger,
                snapshot_id=str(error.get("snapshot_id") or "") or None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {**error, "plan": plan, "modified_ids": modified_ids}

        duration_ms = int((time.monotonic() - start) * 1000)
        snapshot_id = str(txn_meta.get("snapshot_id") or "") or None
        snapshot_id_after = str(txn_meta.get("snapshot_id_after") or "") or None
        resulting_registry_hash = str(txn_meta.get("resulting_registry_hash") or "") or None
        success_reason = (
            self._plan_reasons(plan)[0]
            if self._plan_reasons(plan)
            else "allowed"
        )
        self.audit_log.append(
            actor=actor,
            action="llm.capabilities.reconcile.apply",
            params={
                "impact": plan.get("impact"),
                "modified_ids": modified_ids,
                "changed_ids": modified_ids,
                "snapshot_id": snapshot_id,
                "snapshot_id_after": snapshot_id_after,
                "resulting_registry_hash": resulting_registry_hash,
                "trigger": trigger,
                "scheduler_auto_policy": scheduler_auto_policy,
            },
            decision="allow",
            reason=success_reason,
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        self._record_action_ledger(
            action="llm.capabilities.reconcile.apply",
            actor=actor,
            decision="allow",
            outcome="success",
            reason=success_reason,
            trigger=trigger,
            snapshot_id=snapshot_id,
            snapshot_id_after=snapshot_id_after,
            resulting_registry_hash=resulting_registry_hash,
            changed_ids=modified_ids,
        )
        return True, {
            "ok": True,
            "applied": True,
            "plan": plan,
            "modified_ids": modified_ids,
            "snapshot_id": snapshot_id,
            "snapshot_id_after": snapshot_id_after,
            "resulting_registry_hash": resulting_registry_hash,
        }

    def get_config(self) -> dict[str, Any]:
        defaults = self.get_defaults()
        return {
            "routing_mode": defaults.get("routing_mode"),
            "retry_attempts": self._router.policy.retry_attempts,
            "timeout_seconds": self._router.policy.default_timeout_seconds,
            "secret_storage": self.secret_store.backend_name,
        }

    def update_config(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if "routing_mode" not in payload:
            return False, {"ok": False, "error": "routing_mode is required"}
        ok, updated = self.update_defaults({"routing_mode": payload.get("routing_mode")})
        if not ok:
            return False, updated
        return True, {"ok": True, "routing_mode": updated.get("routing_mode")}

    @staticmethod
    def _plan_modified_ids(plan: dict[str, Any]) -> list[str]:
        changes = plan.get("changes") if isinstance(plan.get("changes"), list) else []
        prune_candidates = plan.get("prune_candidates") if isinstance(plan.get("prune_candidates"), list) else []
        ids: set[str] = set()
        for row in list(changes) + list(prune_candidates):
            if not isinstance(row, dict):
                continue
            kind = str(row.get("kind") or "").strip().lower()
            field = str(row.get("field") or "").strip()
            item_id = str(row.get("id") or "").strip()
            if kind == "defaults" and field:
                ids.add(f"defaults:{field}")
            elif kind and item_id:
                ids.add(f"{kind}:{item_id}")
        return sorted(ids)

    @staticmethod
    def _plan_reasons(plan: dict[str, Any]) -> list[str]:
        reasons = plan.get("reasons") if isinstance(plan.get("reasons"), list) else []
        if reasons:
            return sorted({str(item).strip() for item in reasons if str(item).strip()})
        change_rows = plan.get("changes") if isinstance(plan.get("changes"), list) else []
        inferred = {
            str(row.get("reason") or "").strip()
            for row in change_rows
            if isinstance(row, dict) and str(row.get("reason") or "").strip()
        }
        return sorted(inferred)

    def _latest_autopilot_apply_entry(self) -> dict[str, Any] | None:
        for row in self._action_ledger.recent(limit=max(50, int(self.config.llm_autopilot_churn_recent_limit))):
            if not isinstance(row, dict):
                continue
            if str(row.get("action") or "").strip() not in _AUTOPILOT_APPLY_ACTIONS:
                continue
            if str(row.get("outcome") or "").strip() != "success":
                continue
            if not str(row.get("snapshot_id") or "").strip():
                continue
            return row
        return None

    def _autopilot_apply_policy(self, action: str) -> dict[str, Any]:
        target = str(action or "").strip()
        if target == "llm.self_heal.apply":
            policy = compute_self_heal_apply_policy(self)
            return {
                "allow_apply_effective": bool(policy.get("allow_apply_effective")),
                "allow_reason": str(policy.get("allow_reason") or "permission_required"),
                "loopback": bool(policy.get("is_loopback")),
            }
        if target == "llm.cleanup.apply":
            policy = compute_registry_prune_apply_policy(self)
            return {
                "allow_apply_effective": bool(policy.get("allow_apply_effective")),
                "allow_reason": str(policy.get("allow_reason") or "permission_required"),
                "loopback": bool(policy.get("is_loopback")),
            }
        if target == "llm.capabilities.reconcile.apply":
            policy = compute_capabilities_reconcile_apply_policy(self)
            return {
                "allow_apply_effective": bool(policy.get("allow_apply_effective")),
                "allow_reason": str(policy.get("allow_reason") or "permission_required"),
                "loopback": bool(policy.get("is_loopback")),
            }
        if target == "llm.autopilot.bootstrap.apply":
            policy = compute_autopilot_bootstrap_apply_policy(self)
            return {
                "allow_apply_effective": bool(policy.get("allow_apply_effective")),
                "allow_reason": str(policy.get("allow_reason") or "permission_required"),
                "loopback": bool(policy.get("is_loopback")),
            }
        parsed = urllib.parse.urlparse(str(self.listening_url or ""))
        bind_host = str(parsed.hostname or "").strip()
        return {
            "allow_apply_effective": False,
            "allow_reason": "permission_required",
            "loopback": self._host_is_loopback(bind_host),
        }

    def _evaluate_autopilot_churn(self, *, now_epoch: int, trigger: str) -> dict[str, Any]:
        entries = self._action_ledger.recent(limit=max(50, int(self.config.llm_autopilot_churn_recent_limit)))
        churn = detect_autopilot_churn(
            entries,
            now_epoch=int(now_epoch),
            window_seconds=max(60, int(self.config.llm_autopilot_churn_window_seconds)),
            min_applies=max(2, int(self.config.llm_autopilot_churn_min_applies)),
        )
        self._autopilot_safety_state.update_recent_apply_ids(list(churn.get("recent_apply_ids") or []))
        if not bool(churn.get("triggered")):
            return {"entered_safe_mode": False, "reason": "stable"}
        if self._autopilot_apply_pause_enabled():
            return {"entered_safe_mode": False, "reason": "already_paused"}

        reason = str(churn.get("reason") or "churn_detected")
        state = self._autopilot_safety_state.enter_safe_mode(reason=reason, now_epoch=int(now_epoch))
        self._safe_mode_last_escalation_reason = str(state.get("last_churn_reason") or reason)
        self._safe_mode_last_blocked_reason = f"churn_detected: {self._safe_mode_last_escalation_reason}"

        self.audit_log.append(
            actor="system" if trigger == "scheduler" else "user",
            action="llm.autopilot.safe_mode.enter",
            params={
                "trigger": trigger,
                "reason": reason,
                "apply_count_window": int(churn.get("apply_count_window") or 0),
                "window_seconds": int(churn.get("window_seconds") or 0),
                "flip_flop_models": list(churn.get("flip_flop_models") or []),
                "modified_ids": [],
            },
            decision="allow",
            reason="churn_detected",
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=0,
        )

        notification_line = build_safe_mode_paused_message(
            reason=reason,
            blocked_detail="churn-detected autopilot apply pause",
        )
        return {
            "entered_safe_mode": True,
            "reason": reason,
            "notification_line": notification_line,
        }

    def _current_drift_report(
        self,
        *,
        health_summary: dict[str, Any] | None = None,
        router_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        summary = health_summary if isinstance(health_summary, dict) else self._health_monitor.summary(self.registry_document)
        snapshot = router_snapshot if isinstance(router_snapshot, dict) else self._router.doctor_snapshot()
        drift = build_drift_report(
            self.registry_document,
            summary,
            router_snapshot=snapshot,
        )
        details = drift.get("details") if isinstance(drift.get("details"), dict) else {}
        default_provider = str(details.get("default_provider") or "").strip().lower()
        if default_provider == "ollama":
            provider_rows = snapshot.get("providers") if isinstance(snapshot.get("providers"), list) else []
            ollama_row = next(
                (
                    row
                    for row in provider_rows
                    if isinstance(row, dict) and str(row.get("id") or "").strip().lower() == "ollama"
                ),
                None,
            )
            snapshot_health = (
                ollama_row.get("health")
                if isinstance(ollama_row, dict) and isinstance(ollama_row.get("health"), dict)
                else {}
            )
            effective = self._effective_provider_health("ollama", snapshot_health)
            details["provider_health_status"] = str(effective.get("status") or "unknown").strip().lower() or "unknown"
            drift["details"] = details
        return drift

    def _autopilot_notify_state_snapshot(self) -> dict[str, Any]:
        document = self.registry_document if isinstance(self.registry_document, dict) else {}
        defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
        providers_doc = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models_doc = document.get("models") if isinstance(document.get("models"), dict) else {}
        health_state = self._health_monitor.state if isinstance(self._health_monitor.state, dict) else {}
        provider_health = health_state.get("providers") if isinstance(health_state.get("providers"), dict) else {}
        model_health = health_state.get("models") if isinstance(health_state.get("models"), dict) else {}
        doctor_snapshot = self._router.doctor_snapshot()

        provider_available: dict[str, bool] = {}
        for row in doctor_snapshot.get("providers") or []:
            if not isinstance(row, dict):
                continue
            provider_id = str(row.get("id") or "").strip().lower()
            if not provider_id:
                continue
            provider_available[provider_id] = bool(row.get("available", False))

        model_routable: dict[str, bool] = {}
        for row in doctor_snapshot.get("models") or []:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id") or "").strip()
            if not model_id:
                continue
            model_routable[model_id] = bool(row.get("routable", False))

        provider_ids = sorted(
            {
                str(item).strip().lower()
                for item in list(providers_doc.keys()) + list(provider_health.keys())
                if str(item).strip()
            }
        )
        model_ids = sorted(
            {
                str(item).strip()
                for item in list(models_doc.keys()) + list(model_health.keys())
                if str(item).strip()
            }
        )

        providers: dict[str, Any] = {}
        for provider_id in provider_ids:
            provider_payload = providers_doc.get(provider_id) if isinstance(providers_doc.get(provider_id), dict) else {}
            health_payload = provider_health.get(provider_id) if isinstance(provider_health.get(provider_id), dict) else {}
            try:
                provider_failure_streak = int(health_payload.get("failure_streak") or 0)
            except (TypeError, ValueError):
                provider_failure_streak = 0
            providers[provider_id] = {
                "enabled": bool(provider_payload.get("enabled", False)),
                "available": bool(provider_available.get(provider_id, False)),
                "health": {
                    "status": str(health_payload.get("status") or "unknown").strip().lower() or "unknown",
                    "cooldown_until": health_payload.get("cooldown_until"),
                    "down_since": health_payload.get("down_since"),
                    "failure_streak": max(0, provider_failure_streak),
                },
            }

        models: dict[str, Any] = {}
        for model_id in model_ids:
            model_payload = models_doc.get(model_id) if isinstance(models_doc.get(model_id), dict) else {}
            health_payload = model_health.get(model_id) if isinstance(model_health.get(model_id), dict) else {}
            try:
                model_failure_streak = int(health_payload.get("failure_streak") or 0)
            except (TypeError, ValueError):
                model_failure_streak = 0
            models[model_id] = {
                "enabled": bool(model_payload.get("enabled", False)),
                "available": bool(model_payload.get("available", False)),
                "routable": bool(model_routable.get(model_id, False)),
                "health": {
                    "status": str(health_payload.get("status") or "unknown").strip().lower() or "unknown",
                    "cooldown_until": health_payload.get("cooldown_until"),
                    "down_since": health_payload.get("down_since"),
                    "failure_streak": max(0, model_failure_streak),
                },
            }

        return {
            "defaults": {
                "routing_mode": defaults.get("routing_mode"),
                "default_provider": defaults.get("default_provider"),
                "default_model": defaults.get("default_model"),
                "allow_remote_fallback": defaults.get("allow_remote_fallback"),
            },
            "providers": providers,
            "models": models,
        }

    @staticmethod
    def _notification_audit_reason(reason: str) -> str:
        mapping = {
            "sent": "sent",
            "sent_local": "sent",
            "sent_local_fallback": "sent",
            "rate_limited": "skipped_rate_limit",
            "dedupe_hash_match": "skipped_dedupe",
            "safe_mode_blocked_deduped": "skipped_dedupe",
            "safe_mode_paused_health_suppressed": "skipped_dedupe",
            "quiet_hours": "skipped_quiet_hours",
            "quiet_hours_deferred": "skipped_quiet_hours",
            "permission_required": "permission_required",
            "telegram_not_configured_or_no_chat": "not_configured",
            "disabled": "not_configured",
            "no_changes": "no_changes",
        }
        return mapping.get(str(reason or "").strip(), str(reason or "").strip() or "not_configured")

    @staticmethod
    def _sanitize_public_payload(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): AgentRuntime._sanitize_public_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [AgentRuntime._sanitize_public_payload(item) for item in value]
        if isinstance(value, str):
            return sanitize_notification_text(value)
        return value

    def _set_last_notify_status(
        self,
        *,
        outcome: str,
        reason: str,
        dedupe_hash: str | None,
        changed_defaults: int,
        changed_providers: int,
        changed_models: int,
        ts: int,
    ) -> None:
        self._last_notify_status = {
            "outcome": str(outcome or "unknown"),
            "reason": str(reason or "unknown"),
            "dedupe_hash": str(dedupe_hash or "").strip() or None,
            "ts": int(ts) if int(ts) > 0 else None,
            "changed_defaults": int(changed_defaults),
            "changed_providers": int(changed_providers),
            "changed_models": int(changed_models),
        }

    @staticmethod
    def _safe_mode_block_subject(
        *,
        modified_ids: list[str],
        extra_changes: list[str],
    ) -> str:
        for token in modified_ids:
            model_id = str(token or "").strip()
            if model_id.startswith("model:"):
                subject = model_id[len("model:") :].strip().lower()
                if subject:
                    return f"model:{subject}"
        for token in modified_ids:
            provider_id = str(token or "").strip()
            if provider_id.startswith("provider:"):
                subject = provider_id[len("provider:") :].strip().lower()
                if subject:
                    return f"provider:{subject}"
        for line in sorted({str(item).strip().lower() for item in extra_changes if str(item).strip()}):
            model_match = re.search(r"\bmodel[:\s]+([a-z0-9._:/-]+)", line)
            if model_match:
                return f"model:{model_match.group(1).strip().lower()}"
            provider_match = re.search(r"\bprovider[:\s]+([a-z0-9._:/-]+)", line)
            if provider_match:
                return f"provider:{provider_match.group(1).strip().lower()}"
        return "global"

    def _safe_mode_block_key(
        self,
        *,
        reasons: list[str],
        modified_ids: list[str],
        extra_changes: list[str],
    ) -> str | None:
        normalized_reasons = {str(item or "").strip().lower() for item in reasons if str(item).strip()}
        if "safe_mode_blocked" not in normalized_reasons:
            return None
        subject = self._safe_mode_block_subject(modified_ids=modified_ids, extra_changes=extra_changes)
        return f"safe_mode_blocked:{subject}"

    @staticmethod
    def _health_status_value(row: dict[str, Any] | None) -> str:
        payload = row if isinstance(row, dict) else {}
        health_payload = payload.get("health") if isinstance(payload.get("health"), dict) else {}
        return str(health_payload.get("status") or "unknown").strip().lower() or "unknown"

    @staticmethod
    def _health_numeric_value(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _ok_down_transition_in_state_diff(
        cls,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
    ) -> bool:
        for key in ("providers", "models"):
            before_rows = before_state.get(key) if isinstance(before_state.get(key), dict) else {}
            after_rows = after_state.get(key) if isinstance(after_state.get(key), dict) else {}
            for row_id in sorted(set(before_rows.keys()) | set(after_rows.keys())):
                before_row = before_rows.get(row_id) if isinstance(before_rows.get(row_id), dict) else {}
                after_row = after_rows.get(row_id) if isinstance(after_rows.get(row_id), dict) else {}
                before_status = cls._health_status_value(before_row)
                after_status = cls._health_status_value(after_row)
                if before_status == after_status:
                    continue
                if (before_status == "ok" and after_status == "down") or (
                    before_status == "down" and after_status == "ok"
                ):
                    return True
        return False

    @classmethod
    def _state_diff_health_flags(
        cls,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
    ) -> tuple[bool, bool]:
        health_changed = False
        non_health_changed = False

        before_providers = before_state.get("providers") if isinstance(before_state.get("providers"), dict) else {}
        after_providers = after_state.get("providers") if isinstance(after_state.get("providers"), dict) else {}
        for provider_id in sorted(set(before_providers.keys()) | set(after_providers.keys())):
            before_row = before_providers.get(provider_id) if isinstance(before_providers.get(provider_id), dict) else {}
            after_row = after_providers.get(provider_id) if isinstance(after_providers.get(provider_id), dict) else {}
            if bool(before_row.get("enabled", False)) != bool(after_row.get("enabled", False)):
                non_health_changed = True
            if bool(before_row.get("available", False)) != bool(after_row.get("available", False)):
                non_health_changed = True
            before_health = before_row.get("health") if isinstance(before_row.get("health"), dict) else {}
            after_health = after_row.get("health") if isinstance(after_row.get("health"), dict) else {}
            if cls._health_status_value(before_row) != cls._health_status_value(after_row):
                health_changed = True
            if cls._health_numeric_value(before_health.get("cooldown_until")) != cls._health_numeric_value(
                after_health.get("cooldown_until")
            ):
                health_changed = True
            if cls._health_numeric_value(before_health.get("down_since")) != cls._health_numeric_value(
                after_health.get("down_since")
            ):
                health_changed = True
            if cls._health_numeric_value(before_health.get("failure_streak")) != cls._health_numeric_value(
                after_health.get("failure_streak")
            ):
                health_changed = True

        before_models = before_state.get("models") if isinstance(before_state.get("models"), dict) else {}
        after_models = after_state.get("models") if isinstance(after_state.get("models"), dict) else {}
        for model_id in sorted(set(before_models.keys()) | set(after_models.keys())):
            before_row = before_models.get(model_id) if isinstance(before_models.get(model_id), dict) else {}
            after_row = after_models.get(model_id) if isinstance(after_models.get(model_id), dict) else {}
            if bool(before_row.get("enabled", False)) != bool(after_row.get("enabled", False)):
                non_health_changed = True
            if bool(before_row.get("available", False)) != bool(after_row.get("available", False)):
                non_health_changed = True
            if bool(before_row.get("routable", False)) != bool(after_row.get("routable", False)):
                non_health_changed = True
            before_health = before_row.get("health") if isinstance(before_row.get("health"), dict) else {}
            after_health = after_row.get("health") if isinstance(after_row.get("health"), dict) else {}
            if cls._health_status_value(before_row) != cls._health_status_value(after_row):
                health_changed = True
            if cls._health_numeric_value(before_health.get("cooldown_until")) != cls._health_numeric_value(
                after_health.get("cooldown_until")
            ):
                health_changed = True
            if cls._health_numeric_value(before_health.get("down_since")) != cls._health_numeric_value(
                after_health.get("down_since")
            ):
                health_changed = True
            if cls._health_numeric_value(before_health.get("failure_streak")) != cls._health_numeric_value(
                after_health.get("failure_streak")
            ):
                health_changed = True

        return health_changed, non_health_changed

    def _deliver_autopilot_notification(
        self,
        *,
        message: str,
        allow_remote: bool,
    ) -> DeliveryResult:
        outbound_message, fixit_prompt_state = self._autopilot_notification_message_and_fixit(message)
        token, chat_id = self._resolve_telegram_target()
        telegram_target = TelegramTarget(
            token=token,
            chat_id=chat_id,
            send_fn=self._send_telegram_message,
            enabled=bool(allow_remote),
        )
        local_target = LocalTarget(enabled=True)
        payload = {"message": str(outbound_message or "").strip()}
        telegram_descriptor = telegram_target.target
        local_descriptor = local_target.target

        if telegram_descriptor.enabled and telegram_descriptor.configured:
            result = telegram_target.deliver(payload)
            if result.ok:
                if isinstance(fixit_prompt_state, dict):
                    self._audit_telegram_fixit_prompt_shown(
                        chat_id=chat_id,
                        status=str(fixit_prompt_state.get("status") or ""),
                        issue_code=str(fixit_prompt_state.get("issue_code") or ""),
                        step=str(fixit_prompt_state.get("step") or ""),
                    )
                return result
            fallback = local_target.deliver(payload)
            if fallback.ok:
                return DeliveryResult(
                    ok=True,
                    delivered_to=fallback.delivered_to,
                    error_kind=result.error_kind,
                    reason="sent_local_fallback",
                )
            return result

        if local_descriptor.enabled and local_descriptor.configured:
            return local_target.deliver(payload)

        return DeliveryResult(
            ok=False,
            delivered_to="none",
            error_kind="no_delivery_target",
            reason="not_configured",
        )

    @staticmethod
    def _redact_telegram_chat_id(chat_id: str | None) -> str:
        value = str(chat_id or "").strip()
        if not value:
            return "unknown"
        if len(value) <= 4:
            return "***"
        return f"***{value[-4:]}"

    def _persist_fixit_prompt_state_for_notification(
        self,
        *,
        status_payload: dict[str, Any],
        decision: WizardDecision,
    ) -> dict[str, Any]:
        now_epoch = int(time.time())
        existing_state = (
            self._llm_fixit_store.state
            if isinstance(self._llm_fixit_store.state, dict)
            else self._llm_fixit_store.empty_state()
        )
        openrouter_last_test = (
            existing_state.get("openrouter_last_test")
            if isinstance(existing_state.get("openrouter_last_test"), dict)
            else None
        )
        choices_json = wizard_decision_to_json(decision).get("choices", [])
        state_to_save = {
            "active": True,
            "issue_hash": wizard_decision_issue_hash(decision, status_payload),
            "issue_code": decision.issue_code,
            "step": "awaiting_choice",
            "question": str(decision.question or ""),
            "choices": choices_json,
            "pending_plan": [],
            "pending_confirm_token": None,
            "pending_created_ts": None,
            "pending_expires_ts": None,
            "pending_issue_code": None,
            "last_prompt_ts": now_epoch,
            "openrouter_last_test": openrouter_last_test,
        }
        try:
            return self._llm_fixit_store.save(state_to_save)
        except Exception:
            return (
                existing_state if isinstance(existing_state, dict) else self._llm_fixit_store.empty_state()
            )

    def _autopilot_notification_message_and_fixit(
        self,
        raw_message: str,
    ) -> tuple[str, dict[str, Any] | None]:
        status_payload = self.llm_status()
        decision = evaluate_wizard_decision(status_payload)
        if decision.status != "ok":
            state = self._persist_fixit_prompt_state_for_notification(
                status_payload=status_payload,
                decision=decision,
            )
            return render_wizard_prompt(decision), {
                "status": str(decision.status or "needs_user_choice"),
                "issue_code": str(decision.issue_code or ""),
                "step": str(state.get("step") or "awaiting_choice"),
            }
        return summarize_notification_message(raw_message), None

    def _audit_telegram_fixit_prompt_shown(
        self,
        *,
        chat_id: str | None,
        status: str,
        issue_code: str,
        step: str,
    ) -> None:
        try:
            self.audit_log.append(
                actor="system",
                action="telegram.fixit.prompt_shown",
                params={
                    "chat_id_redacted": self._redact_telegram_chat_id(chat_id),
                    "status": str(status or ""),
                    "issue_code": str(issue_code or ""),
                    "step": str(step or ""),
                },
                decision="allow",
                reason="prompt_shown",
                dry_run=False,
                outcome="sent",
                error_kind=None,
                duration_ms=0,
            )
        except Exception:
            pass

    def _autopilot_notification_user_message(self, raw_message: str) -> str:
        outbound_message, _fixit_prompt_state = self._autopilot_notification_message_and_fixit(raw_message)
        return outbound_message

    def _process_scheduler_notification_cycle(
        self,
        *,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        reasons: list[str],
        extra_changes: list[str] | None = None,
        trigger: str,
    ) -> dict[str, Any]:
        start = time.monotonic()
        now_epoch = int(time.time())
        extra_changes_list = sorted({str(item).strip() for item in (extra_changes or []) if str(item).strip()})
        diff = build_notification_from_state_diff(
            before_state,
            after_state,
            reasons,
            extra_changes=extra_changes_list,
        )
        message = str(diff.get("message") or "").strip()
        dedupe_hash = str(diff.get("dedupe_hash") or "").strip()
        counts = diff.get("counts") if isinstance(diff.get("counts"), dict) else {}
        modified_ids = [
            str(item).strip()
            for item in (diff.get("modified_ids") or [])
            if str(item).strip()
        ]
        changed_defaults = int(counts.get("defaults") or 0)
        changed_providers = int(counts.get("providers") or 0)
        changed_models = int(counts.get("models") or 0)
        changed_any = bool(message and dedupe_hash)
        actor = "system" if trigger == "scheduler" else "user"
        safe_mode_key = self._safe_mode_block_key(
            reasons=[str(item or "") for item in (reasons or [])],
            modified_ids=modified_ids,
            extra_changes=extra_changes_list,
        )
        health_changed, non_health_changed = self._state_diff_health_flags(before_state, after_state)
        ok_down_transition = self._ok_down_transition_in_state_diff(before_state, after_state)
        threshold_crossed = failure_streak_threshold_crossed(before_state, after_state)
        if (
            changed_any
            and changed_defaults == 0
            and health_changed
            and not non_health_changed
            and not ok_down_transition
            and not threshold_crossed
        ):
            changed_any = False
            message = ""
            dedupe_hash = ""
        health_pause_key = "safe_mode_paused_health"
        should_track_paused_health_send = bool(
            self._autopilot_apply_pause_enabled()
            and changed_defaults == 0
            and health_changed
            and not non_health_changed
            and not ok_down_transition
        )

        if not changed_any:
            duration_ms = int((time.monotonic() - start) * 1000)
            self._set_last_notify_status(
                outcome="skipped",
                reason="no_changes",
                dedupe_hash=None,
                changed_defaults=0,
                changed_providers=0,
                changed_models=0,
                ts=now_epoch,
            )
            self.audit_log.append(
                actor=actor,
                action="llm.autopilot.notify",
                params={
                    "trigger": trigger,
                    "dedupe_hash": None,
                    "changed_defaults": 0,
                    "changed_providers": 0,
                    "changed_models": 0,
                    "modified_ids": [],
                    "send_policy": compute_notification_send_policy(self),
                },
                decision="allow",
                reason="no_changes",
                dry_run=False,
                outcome="skipped",
                error_kind=None,
                duration_ms=duration_ms,
            )
            return {"ok": True, "outcome": "skipped", "reason": "no_changes"}

        if safe_mode_key:
            last_subject_sent_ts = self._notification_store.reason_subject_last_sent_ts(safe_mode_key)
            if (
                isinstance(last_subject_sent_ts, int)
                and last_subject_sent_ts > 0
                and (now_epoch - last_subject_sent_ts) < _SAFE_MODE_BLOCKED_DEDUPE_SECONDS
            ):
                self._notification_store.append(
                    ts=now_epoch,
                    message=message,
                    dedupe_hash=dedupe_hash,
                    delivered_to="none",
                    deferred=False,
                    outcome="skipped",
                    reason="safe_mode_blocked_deduped",
                    modified_ids=modified_ids,
                    mark_sent=False,
                )
                duration_ms = int((time.monotonic() - start) * 1000)
                audit_reason = self._notification_audit_reason("safe_mode_blocked_deduped")
                self._set_last_notify_status(
                    outcome="skipped",
                    reason=audit_reason,
                    dedupe_hash=dedupe_hash,
                    changed_defaults=changed_defaults,
                    changed_providers=changed_providers,
                    changed_models=changed_models,
                    ts=now_epoch,
                )
                self.audit_log.append(
                    actor=actor,
                    action="llm.autopilot.notify",
                    params={
                        "trigger": trigger,
                        "dedupe_hash": dedupe_hash,
                        "changed_defaults": changed_defaults,
                        "changed_providers": changed_providers,
                        "changed_models": changed_models,
                        "modified_ids": modified_ids,
                        "safe_mode_key": safe_mode_key,
                    },
                    decision="allow",
                    reason=audit_reason,
                    dry_run=False,
                    outcome="skipped",
                    error_kind=None,
                    duration_ms=duration_ms,
                )
                return {
                    "ok": True,
                    "outcome": "skipped",
                    "reason": audit_reason,
                    "dedupe_hash": dedupe_hash,
                    "counts": {
                        "defaults": changed_defaults,
                        "providers": changed_providers,
                        "models": changed_models,
                    },
                    "delivered_to": "none",
                }

        if should_track_paused_health_send:
            last_paused_health_sent_ts = self._notification_store.reason_subject_last_sent_ts(health_pause_key)
            if (
                isinstance(last_paused_health_sent_ts, int)
                and last_paused_health_sent_ts > 0
                and (now_epoch - last_paused_health_sent_ts) < _SAFE_MODE_PAUSED_HEALTH_NOTIFY_SECONDS
            ):
                self._notification_store.append(
                    ts=now_epoch,
                    message=message,
                    dedupe_hash=dedupe_hash,
                    delivered_to="none",
                    deferred=False,
                    outcome="skipped",
                    reason="safe_mode_paused_health_suppressed",
                    modified_ids=modified_ids,
                    mark_sent=False,
                )
                duration_ms = int((time.monotonic() - start) * 1000)
                audit_reason = self._notification_audit_reason("safe_mode_paused_health_suppressed")
                self._set_last_notify_status(
                    outcome="skipped",
                    reason=audit_reason,
                    dedupe_hash=dedupe_hash,
                    changed_defaults=changed_defaults,
                    changed_providers=changed_providers,
                    changed_models=changed_models,
                    ts=now_epoch,
                )
                self.audit_log.append(
                    actor=actor,
                    action="llm.autopilot.notify",
                    params={
                        "trigger": trigger,
                        "dedupe_hash": dedupe_hash,
                        "changed_defaults": changed_defaults,
                        "changed_providers": changed_providers,
                        "changed_models": changed_models,
                        "modified_ids": modified_ids,
                        "safe_mode_health_key": health_pause_key,
                    },
                    decision="allow",
                    reason=audit_reason,
                    dry_run=False,
                    outcome="skipped",
                    error_kind=None,
                    duration_ms=duration_ms,
                )
                return {
                    "ok": True,
                    "outcome": "skipped",
                    "reason": audit_reason,
                    "dedupe_hash": dedupe_hash,
                    "counts": {
                        "defaults": changed_defaults,
                        "providers": changed_providers,
                        "models": changed_models,
                    },
                    "delivered_to": "none",
                }

        send_gate = should_send(
            now_epoch=now_epoch,
            last_sent_ts=self._notification_store.state.get("last_sent_ts"),
            last_sent_hash=self._notification_store.state.get("last_sent_hash"),
            message_hash=dedupe_hash,
            enabled=bool(self.config.autopilot_notify_enabled),
            rate_limit_seconds=max(0, int(self.config.autopilot_notify_rate_limit_seconds)),
            dedupe_window_seconds=max(0, int(self.config.autopilot_notify_dedupe_window_seconds)),
            quiet_start_hour=self.config.autopilot_notify_quiet_start_hour,
            quiet_end_hour=self.config.autopilot_notify_quiet_end_hour,
            timezone_name=self.config.agent_timezone,
        )
        send_policy = compute_notification_send_policy(self)
        permission = self._modelops_permission_decision(
            "llm.notifications.send",
            params={"trigger": trigger, "counts": counts},
            estimated_download_bytes=0,
            estimated_cost=None,
            risk_level="low",
            dry_run=False,
        )
        permission_allow = bool(permission.get("allow"))
        local_default_allow = bool(send_policy.get("allow_send_effective"))
        remote_send_allowed = permission_allow or local_default_allow

        delivered_to = "none"
        deferred = False
        outcome = "skipped"
        reason = str(send_gate.get("reason") or "disabled")
        decision = "allow"
        mark_sent = False
        error_kind: str | None = None

        if not bool(send_gate.get("send")):
            deferred = bool(send_gate.get("deferred"))
            if deferred:
                reason = "quiet_hours"
                mark_sent = True
            elif reason == "rate_limited":
                reason = "rate_limited"
            elif reason == "dedupe_hash_match":
                reason = "dedupe_hash_match"
            else:
                reason = "disabled"
        else:
            delivery_result = self._deliver_autopilot_notification(
                message=message,
                allow_remote=remote_send_allowed,
            )
            delivered_to = str(delivery_result.delivered_to or "none")
            error_kind = delivery_result.error_kind
            if delivery_result.ok:
                outcome = "sent"
                reason = str(delivery_result.reason or "sent")
                mark_sent = True
            else:
                outcome = "skipped"
                reason = str(delivery_result.reason or "not_configured")
                mark_sent = True
                if not remote_send_allowed and reason in {
                    "telegram_not_configured_or_no_chat",
                    "not_configured",
                    "delivery_disabled",
                }:
                    decision = "deny"
                    reason = "permission_required"
                    error_kind = "permission_required"

        self._notification_store.append(
            ts=now_epoch,
            message=message,
            dedupe_hash=dedupe_hash,
            delivered_to=delivered_to,
            deferred=deferred,
            outcome=outcome,
            reason=reason,
            modified_ids=modified_ids,
            mark_sent=mark_sent,
        )
        if safe_mode_key and outcome == "sent":
            self._notification_store.mark_reason_subject_sent(safe_mode_key, now_epoch)
        if should_track_paused_health_send and outcome == "sent":
            self._notification_store.mark_reason_subject_sent(health_pause_key, now_epoch)

        audit_reason = self._notification_audit_reason(reason)
        duration_ms = int((time.monotonic() - start) * 1000)
        self._set_last_notify_status(
            outcome=outcome,
            reason=audit_reason,
            dedupe_hash=dedupe_hash,
            changed_defaults=changed_defaults,
            changed_providers=changed_providers,
            changed_models=changed_models,
            ts=now_epoch,
        )
        self.audit_log.append(
            actor=actor,
            action="llm.autopilot.notify",
            params={
                "trigger": trigger,
                "dedupe_hash": dedupe_hash,
                "changed_defaults": changed_defaults,
                "changed_providers": changed_providers,
                "changed_models": changed_models,
                "modified_ids": modified_ids,
                "delivered_to": delivered_to,
                "deferred": deferred,
                "send_policy": send_policy,
                "remote_permission_allow": permission_allow,
                "remote_policy_allow": local_default_allow,
                "remote_send_allowed": remote_send_allowed,
                "remote_policy_reason": str(send_policy.get("allow_reason") or ""),
            },
            decision=decision,
            reason=audit_reason,
            dry_run=False,
            outcome=outcome,
            error_kind=error_kind,
            duration_ms=duration_ms,
        )
        return {
            "ok": True,
            "outcome": outcome,
            "reason": audit_reason,
            "dedupe_hash": dedupe_hash,
            "counts": {
                "defaults": changed_defaults,
                "providers": changed_providers,
                "models": changed_models,
            },
            "delivered_to": delivered_to,
        }

    def _record_autopilot_notification(
        self,
        *,
        message: str,
        dedupe_hash: str,
        modified_ids: list[str],
        trigger: str,
        forced: bool = False,
    ) -> dict[str, Any]:
        now_epoch = int(time.time())
        decision = should_send(
            now_epoch=now_epoch,
            last_sent_ts=self._notification_store.state.get("last_sent_ts"),
            last_sent_hash=self._notification_store.state.get("last_sent_hash"),
            message_hash=dedupe_hash,
            enabled=bool(self.config.autopilot_notify_enabled),
            rate_limit_seconds=max(0, int(self.config.autopilot_notify_rate_limit_seconds)),
            dedupe_window_seconds=max(0, int(self.config.autopilot_notify_dedupe_window_seconds)),
            quiet_start_hour=self.config.autopilot_notify_quiet_start_hour,
            quiet_end_hour=self.config.autopilot_notify_quiet_end_hour,
            timezone_name=self.config.agent_timezone,
        )
        if forced:
            decision = {"send": True, "deferred": False, "reason": "forced_test"}

        delivered_to = "none"
        deferred = bool(decision.get("deferred"))
        outcome = "skipped"
        reason = str(decision.get("reason") or "skipped")
        mark_sent = False
        error_kind: str | None = None

        if bool(decision.get("send")):
            delivery_result = self._deliver_autopilot_notification(message=message, allow_remote=True)
            delivered_to = str(delivery_result.delivered_to or "none")
            error_kind = delivery_result.error_kind
            if delivery_result.ok:
                outcome = "sent"
                reason = str(delivery_result.reason or "sent")
                mark_sent = True
            else:
                outcome = "skipped"
                reason = str(delivery_result.reason or "not_configured")
                mark_sent = True
        elif deferred:
            outcome = "skipped"
            reason = "quiet_hours_deferred"
            mark_sent = True

        self._notification_store.append(
            ts=now_epoch,
            message=message,
            dedupe_hash=dedupe_hash,
            delivered_to=delivered_to,
            deferred=deferred,
            outcome=outcome,
            reason=reason,
            modified_ids=modified_ids,
            mark_sent=mark_sent,
        )

        self.audit_log.append(
            actor="system" if trigger == "scheduler" else "user",
            action="llm.autopilot.notify",
            params={
                "trigger": trigger,
                "modified_ids": sorted({str(item).strip() for item in modified_ids if str(item).strip()}),
                "dedupe_hash": dedupe_hash,
                "delivered_to": delivered_to,
                "deferred": deferred,
            },
            decision="allow",
            reason=reason,
            dry_run=False,
            outcome=outcome,
            error_kind=error_kind if error_kind is not None else (None if outcome == "sent" else reason),
            duration_ms=0,
        )
        return {
            "ok": True,
            "outcome": outcome,
            "reason": reason,
            "delivered_to": delivered_to,
            "deferred": deferred,
            "dedupe_hash": dedupe_hash,
        }

    def _notify_autopilot_changes(
        self,
        *,
        before_document: dict[str, Any],
        after_document: dict[str, Any],
        reasons: list[str],
        modified_ids: list[str],
        trigger: str,
    ) -> dict[str, Any] | None:
        message, dedupe_hash, _change_lines = build_notification_from_diff(
            before_document,
            after_document,
            reasons,
            modified_ids,
        )
        if not message or not dedupe_hash:
            return None
        return self._record_autopilot_notification(
            message=message,
            dedupe_hash=dedupe_hash,
            modified_ids=modified_ids,
            trigger=trigger,
        )

    def llm_notifications(self, limit: int = 20) -> dict[str, Any]:
        return {
            "ok": True,
            "notifications": self._notification_store.recent(limit=max(1, int(limit))),
        }

    def llm_notifications_status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": self._notification_store.status(),
        }

    def llm_notifications_last_change(self) -> dict[str, Any]:
        last_change = self._notification_store.last_change()
        if not isinstance(last_change, dict):
            return {"ok": True, "found": False, "last_change": None}
        return {"ok": True, "found": True, "last_change": last_change}

    def llm_notifications_policy(self) -> dict[str, Any]:
        return {
            "ok": True,
            "policy": compute_notification_test_policy(self),
        }

    def llm_support_bundle(self) -> dict[str, Any]:
        document = self.registry_document if isinstance(self.registry_document, dict) else {}
        defaults = self._ensure_defaults(copy.deepcopy(document))
        providers_doc = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models_doc = document.get("models") if isinstance(document.get("models"), dict) else {}
        health_payload = self.llm_health_summary()
        health = health_payload.get("health") if isinstance(health_payload.get("health"), dict) else {}
        snapshot = self._router.doctor_snapshot()
        catalog_rows = self._catalog_store.all_models(limit=20_000)
        catalog_lookup = {
            str(row.get("id") or "").strip(): row
            for row in catalog_rows
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }

        provider_health_lookup = {
            str(row.get("id") or "").strip().lower(): row
            for row in (health.get("providers") if isinstance(health.get("providers"), list) else [])
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        model_health_lookup = {
            str(row.get("id") or "").strip(): row
            for row in (health.get("models") if isinstance(health.get("models"), list) else [])
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        provider_snapshot_lookup = {
            str(row.get("id") or "").strip().lower(): row
            for row in (snapshot.get("providers") if isinstance(snapshot.get("providers"), list) else [])
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        model_snapshot_lookup = {
            str(row.get("id") or "").strip(): row
            for row in (snapshot.get("models") if isinstance(snapshot.get("models"), list) else [])
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }

        providers: list[dict[str, Any]] = []
        for provider_id, payload in sorted(providers_doc.items()):
            if not isinstance(payload, dict):
                continue
            health_row = provider_health_lookup.get(str(provider_id).strip().lower()) or {}
            snapshot_row = provider_snapshot_lookup.get(str(provider_id).strip().lower()) or {}
            source = payload.get("api_key_source") if isinstance(payload.get("api_key_source"), dict) else {}
            providers.append(
                {
                    "id": str(provider_id).strip().lower(),
                    "provider_type": str(payload.get("provider_type") or "").strip() or None,
                    "enabled": bool(payload.get("enabled", True)),
                    "local": bool(payload.get("local", False)),
                    "base_url": payload.get("base_url"),
                    "chat_path": payload.get("chat_path"),
                    "available": bool(snapshot_row.get("available", False)),
                    "health_status": str(health_row.get("status") or "unknown").strip().lower() or "unknown",
                    "last_error_kind": str(health_row.get("last_error_kind") or "").strip().lower() or None,
                    "status_code": health_row.get("status_code"),
                    "failure_streak": int(health_row.get("failure_streak") or 0),
                    "api_key_source": {
                        "type": str(source.get("type") or "").strip() or None,
                        "configured": bool(self._provider_api_key(payload)),
                    },
                    "default_headers": payload.get("default_headers") if isinstance(payload.get("default_headers"), dict) else {},
                    "default_query_params": (
                        payload.get("default_query_params")
                        if isinstance(payload.get("default_query_params"), dict)
                        else {}
                    ),
                }
            )

        models: list[dict[str, Any]] = []
        for model_id, payload in sorted(models_doc.items()):
            if not isinstance(payload, dict):
                continue
            model_key = str(model_id).strip()
            health_row = model_health_lookup.get(model_key) or {}
            snapshot_row = model_snapshot_lookup.get(model_key) or {}
            catalog_row = catalog_lookup.get(model_key) if isinstance(catalog_lookup.get(model_key), dict) else {}
            models.append(
                {
                    "id": model_key,
                    "provider": str(payload.get("provider") or "").strip().lower() or None,
                    "model": str(payload.get("model") or "").strip() or None,
                    "enabled": bool(payload.get("enabled", False)),
                    "available": bool(payload.get("available", False)),
                    "routable": bool(snapshot_row.get("routable", False)),
                    "capabilities": sorted(
                        {
                            str(item).strip().lower()
                            for item in (payload.get("capabilities") or [])
                            if str(item).strip()
                        }
                    ),
                    "health_status": str(health_row.get("status") or "unknown").strip().lower() or "unknown",
                    "last_error_kind": str(health_row.get("last_error_kind") or "").strip().lower() or None,
                    "status_code": health_row.get("status_code"),
                    "failure_streak": int(health_row.get("failure_streak") or 0),
                    "max_context_tokens": catalog_row.get("max_context_tokens"),
                    "input_cost_per_million_tokens": catalog_row.get("input_cost_per_million_tokens"),
                    "output_cost_per_million_tokens": catalog_row.get("output_cost_per_million_tokens"),
                    "catalog_source": str(catalog_row.get("source") or "").strip() or None,
                }
            )

        audit_rows: list[dict[str, Any]] = []
        for row in self.audit_log.recent(limit=50):
            if not isinstance(row, dict):
                continue
            audit_rows.append(
                {
                    "ts": str(row.get("ts") or "").strip() or None,
                    "actor": str(row.get("actor") or "").strip() or None,
                    "action": str(row.get("action") or "").strip() or None,
                    "decision": str(row.get("decision") or "").strip() or None,
                    "reason": str(row.get("reason") or "").strip() or None,
                    "dry_run": bool(row.get("dry_run", False)),
                    "outcome": str(row.get("outcome") or "").strip() or None,
                    "error_kind": str(row.get("error_kind") or "").strip() or None,
                    "duration_ms": int(row.get("duration_ms") or 0),
                    "params_redacted": row.get("params_redacted") if isinstance(row.get("params_redacted"), dict) else {},
                }
            )

        ledger_rows: list[dict[str, Any]] = []
        for row in self._action_ledger.recent(limit=50):
            if not isinstance(row, dict):
                continue
            ledger_rows.append(
                {
                    "id": str(row.get("id") or "").strip() or None,
                    "ts": int(row.get("ts") or 0) or None,
                    "ts_iso": str(row.get("ts_iso") or "").strip() or None,
                    "action": str(row.get("action") or "").strip() or None,
                    "actor": str(row.get("actor") or "").strip() or None,
                    "decision": str(row.get("decision") or "").strip() or None,
                    "outcome": str(row.get("outcome") or "").strip() or None,
                    "reason": str(row.get("reason") or "").strip() or None,
                    "trigger": str(row.get("trigger") or "").strip() or None,
                    "snapshot_id_before": str(row.get("snapshot_id_before") or row.get("snapshot_id") or "").strip()
                    or None,
                    "snapshot_id_after": str(row.get("snapshot_id_after") or "").strip() or None,
                    "resulting_registry_hash": str(row.get("resulting_registry_hash") or "").strip() or None,
                    "changed_ids": sorted(
                        {
                            str(item).strip()
                            for item in (row.get("changed_ids") or [])
                            if str(item).strip()
                        }
                    ),
                }
            )

        notification_rows: list[dict[str, Any]] = []
        for row in self._notification_store.recent(limit=20):
            if not isinstance(row, dict):
                continue
            notification_rows.append(
                {
                    "ts": int(row.get("ts") or 0) or None,
                    "ts_iso": str(row.get("ts_iso") or "").strip() or None,
                    "title": str(row.get("message") or "").splitlines()[0:1][0].strip()
                    if str(row.get("message") or "").splitlines()[0:1]
                    else "LLM Autopilot updated configuration",
                    "message": str(row.get("message") or "").strip() or None,
                    "dedupe_hash": str(row.get("dedupe_hash") or "").strip() or None,
                    "outcome": str(row.get("outcome") or "").strip() or None,
                    "reason": str(row.get("reason") or "").strip() or None,
                    "delivered_to": str(row.get("delivered_to") or "").strip() or None,
                    "deferred": bool(row.get("deferred", False)),
                    "modified_ids": sorted(
                        {
                            str(item).strip()
                            for item in (row.get("modified_ids") or [])
                            if str(item).strip()
                        }
                    ),
                }
            )

        safety_state = self._safe_mode_status()
        safe_mode_reason = (
            str(safety_state.get("safe_mode_reason") or "").strip()
            or str(self._safe_mode_last_blocked_reason or "").strip()
            or None
        )

        health_summary = {
            "last_run_at": health.get("last_run_at"),
            "last_run_at_iso": health.get("last_run_at_iso"),
            "counts": health.get("counts") if isinstance(health.get("counts"), dict) else {},
            "drift": health.get("drift") if isinstance(health.get("drift"), dict) else {},
            "providers": sorted(
                [
                    {
                        "id": str(row.get("id") or "").strip().lower(),
                        "status": str(row.get("status") or "unknown").strip().lower(),
                        "last_error_kind": str(row.get("last_error_kind") or "").strip().lower() or None,
                        "status_code": row.get("status_code"),
                        "failure_streak": int(row.get("failure_streak") or 0),
                        "cooldown_until_iso": row.get("cooldown_until_iso"),
                    }
                    for row in (health.get("providers") if isinstance(health.get("providers"), list) else [])
                    if isinstance(row, dict) and str(row.get("id") or "").strip()
                ],
                key=lambda item: str(item.get("id") or ""),
            ),
            "models": sorted(
                [
                    {
                        "id": str(row.get("id") or "").strip(),
                        "provider_id": str(row.get("provider_id") or "").strip().lower() or None,
                        "status": str(row.get("status") or "unknown").strip().lower(),
                        "last_error_kind": str(row.get("last_error_kind") or "").strip().lower() or None,
                        "status_code": row.get("status_code"),
                        "failure_streak": int(row.get("failure_streak") or 0),
                        "cooldown_until_iso": row.get("cooldown_until_iso"),
                    }
                    for row in (health.get("models") if isinstance(health.get("models"), list) else [])
                    if isinstance(row, dict) and str(row.get("id") or "").strip()
                ],
                key=lambda item: str(item.get("id") or ""),
            ),
        }

        bundle = {
            "created_at_iso": self.started_at_iso,
            "api_version": "v1",
            "git_commit": self.git_commit or "unknown",
            "registry_hash": self._registry_hash(document),
            "safe_mode": {
                "enabled": bool(self._effective_safe_mode()),
                "reason": safe_mode_reason,
                "since_iso": safety_state.get("safe_mode_entered_ts_iso"),
            },
            "defaults": {
                "routing_mode": defaults.get("routing_mode"),
                "default_provider": defaults.get("default_provider"),
                "default_model": defaults.get("default_model"),
                "allow_remote_fallback": bool(defaults.get("allow_remote_fallback", True)),
            },
            "providers": providers,
            "models": models,
            "health_summary": health_summary,
            "last_actions": audit_rows[:50],
            "ledger_tail": ledger_rows[:50],
            "notifications_tail": notification_rows[:20],
            "policies": {
                "notify_test": compute_notification_test_policy(self),
                "notify_send": compute_notification_send_policy(self),
                "rollback": compute_registry_rollback_policy(self),
                "self_heal_apply": compute_self_heal_apply_policy(self),
                "cleanup_apply": compute_registry_prune_apply_policy(self),
                "capabilities_reconcile_apply": compute_capabilities_reconcile_apply_policy(self),
                "bootstrap_apply": compute_autopilot_bootstrap_apply_policy(self),
            },
        }
        safe_bundle = sanitize_support_payload(redact_audit_value(bundle))
        return {"ok": True, "bundle": safe_bundle}

    def llm_support_diagnose(self, target_id: str) -> tuple[bool, dict[str, Any]]:
        target = str(target_id or "").strip()
        if not target:
            return False, {"ok": False, "error": "id is required"}

        document = self.registry_document if isinstance(self.registry_document, dict) else {}
        providers_doc = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models_doc = document.get("models") if isinstance(document.get("models"), dict) else {}
        health = self._health_monitor.summary(self.registry_document)
        snapshot = self._router.doctor_snapshot()
        catalog_models = self._catalog_store.all_models(limit=20_000)
        catalog_lookup = {
            str(row.get("id") or "").strip(): row
            for row in catalog_models
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        provider_catalog_ids: dict[str, list[str]] = {}
        for row in catalog_models:
            if not isinstance(row, dict):
                continue
            provider_id = str(row.get("provider_id") or "").strip().lower()
            model_id = str(row.get("id") or "").strip()
            if not provider_id or not model_id:
                continue
            provider_catalog_ids.setdefault(provider_id, []).append(model_id)
        for provider_id in list(provider_catalog_ids.keys()):
            provider_catalog_ids[provider_id] = sorted({item for item in provider_catalog_ids[provider_id] if item})

        provider_health_lookup = {
            str(row.get("id") or "").strip().lower(): row
            for row in (health.get("providers") if isinstance(health.get("providers"), list) else [])
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        model_health_lookup = {
            str(row.get("id") or "").strip(): row
            for row in (health.get("models") if isinstance(health.get("models"), list) else [])
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        model_snapshot_lookup = {
            str(row.get("id") or "").strip(): row
            for row in (snapshot.get("models") if isinstance(snapshot.get("models"), list) else [])
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }

        provider_key = target.lower()
        if provider_key in providers_doc and isinstance(providers_doc.get(provider_key), dict):
            provider_payload = providers_doc.get(provider_key) if isinstance(providers_doc.get(provider_key), dict) else {}
            headers = self._provider_request_headers(provider_payload)
            validation_payload = dict(provider_payload)
            validation_payload["_resolved_api_key_present"] = bool(self._provider_api_key(provider_payload))
            validation = validate_provider_call_format(provider_key, validation_payload, headers=headers)
            related_models = [
                {
                    "id": str(row.get("id") or "").strip(),
                    "status": str((row.get("health") or {}).get("status") or "unknown").strip().lower(),
                    "last_error_kind": str((row.get("health") or {}).get("last_error_kind") or "").strip().lower() or None,
                    "routable": bool(row.get("routable", False)),
                }
                for row in (snapshot.get("models") if isinstance(snapshot.get("models"), list) else [])
                if isinstance(row, dict) and str(row.get("provider") or "").strip().lower() == provider_key
            ]
            diagnosis = build_provider_diagnosis(
                provider_id=provider_key,
                provider_payload=provider_payload,
                provider_health=provider_health_lookup.get(provider_key),
                validation=validation,
                related_models=related_models,
            )
            return True, {
                "ok": True,
                "kind": "provider",
                "id": provider_key,
                "diagnosis": diagnosis,
            }

        if target in models_doc and isinstance(models_doc.get(target), dict):
            model_payload = models_doc.get(target) if isinstance(models_doc.get(target), dict) else {}
            provider_id = str(model_payload.get("provider") or "").strip().lower()
            provider_payload = providers_doc.get(provider_id) if isinstance(providers_doc.get(provider_id), dict) else {}
            diagnosis = build_model_diagnosis(
                model_id=target,
                model_payload=model_payload,
                model_health=model_health_lookup.get(target),
                model_snapshot=model_snapshot_lookup.get(target),
                provider_payload=provider_payload,
                provider_health=provider_health_lookup.get(provider_id),
                catalog_entry=catalog_lookup.get(target) if isinstance(catalog_lookup.get(target), dict) else None,
                provider_catalog_ids=provider_catalog_ids.get(provider_id) or [],
            )
            return True, {
                "ok": True,
                "kind": "model",
                "id": target,
                "diagnosis": diagnosis,
            }

        return False, {"ok": False, "error": "not_found"}

    def llm_support_remediate_plan(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        target = str(payload.get("target") or "").strip() or None
        intent = str(payload.get("intent") or "fix_routing").strip().lower() or "fix_routing"
        diagnosis: dict[str, Any] | None = None
        if target and target != "defaults":
            ok, body = self.llm_support_diagnose(target)
            if not ok:
                return False, {"ok": False, "error": "target_not_found"}
            diagnosis = body.get("diagnosis") if isinstance(body.get("diagnosis"), dict) else None
        defaults = self._ensure_defaults(self.registry_document)
        health_summary = self._health_monitor.summary(self.registry_document)
        model_watch_state = normalize_model_watch_state(self._model_watch_store.load())
        latest_batch = summarize_model_watch_batch(latest_model_watch_batch(model_watch_state))
        last_error_kind = (
            str(payload.get("last_error_kind") or "").strip().lower()
            or str((diagnosis or {}).get("last_error_kind") or "").strip().lower()
            or None
        )
        last_status_code: int | None
        try:
            last_status_code = int(payload.get("status_code")) if payload.get("status_code") is not None else None
        except (TypeError, ValueError):
            last_status_code = None
        if last_status_code is None:
            try:
                last_status_code = int((diagnosis or {}).get("status_code")) if (diagnosis or {}).get("status_code") is not None else None
            except (TypeError, ValueError):
                last_status_code = None
        last_error = (
            str(payload.get("error") or "").strip()
            or str(payload.get("detail") or "").strip()
            or None
        )
        plan = build_llm_remediation_plan(
            registry_snapshot=self.registry_document if isinstance(self.registry_document, dict) else {},
            defaults=defaults,
            health_summary=health_summary if isinstance(health_summary, dict) else {},
            last_error_kind=last_error_kind,
            last_status_code=last_status_code,
            last_error=last_error,
            safe_mode=self._safe_mode_health_payload(),
            routing_mode=str(defaults.get("routing_mode") or ""),
            latest_model_watch_batch=latest_batch,
            ollama_model_fallback=self.config.ollama_model,
            target=target,
            intent=intent,
        )
        return True, {"ok": True, "plan": plan}

    def llm_support_remediate_execute(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        confirm = bool(payload.get("confirm", False))
        actor = str(payload.get("actor") or "user").strip() or "user"
        if not confirm:
            return False, {"ok": False, "error": "confirmation_required", "message": "confirm=true is required"}

        plan_ok, plan_body = self.llm_support_remediate_plan(payload)
        if not plan_ok:
            return False, plan_body
        plan = plan_body.get("plan") if isinstance(plan_body.get("plan"), dict) else {}
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []

        executed_steps: list[dict[str, Any]] = []
        blocked_steps: list[dict[str, Any]] = []
        failed_steps: list[dict[str, Any]] = []
        user_actions: list[dict[str, Any]] = []

        for index, raw_step in enumerate(steps, start=1):
            step = raw_step if isinstance(raw_step, dict) else {}
            step_id = str(step.get("id") or f"step_{index}")
            kind = str(step.get("kind") or "").strip().lower()
            action = str(step.get("action") or "").strip()
            reason = str(step.get("reason") or "").strip()
            params = step.get("params") if isinstance(step.get("params"), dict) else {}
            instructions = str(step.get("instructions") or "").strip()
            safe_to_execute = bool(step.get("safe_to_execute", False))

            if kind == "user_action":
                user_actions.append(
                    {
                        "id": step_id,
                        "action": action,
                        "reason": reason,
                        "instructions": instructions,
                    }
                )
                continue

            if not safe_to_execute:
                blocked_steps.append(
                    {
                        "id": step_id,
                        "action": action,
                        "reason": reason or "step_not_marked_safe",
                        "error": "unsafe_step",
                    }
                )
                continue

            if not action.startswith("modelops."):
                blocked_steps.append(
                    {
                        "id": step_id,
                        "action": action,
                        "reason": reason or "unsupported_safe_step",
                        "error": "unsupported_action",
                    }
                )
                continue

            plan_preview_ok, plan_preview = self.modelops_plan(
                {
                    "action": action,
                    "params": params,
                    "dry_run": False,
                    "actor": actor,
                }
            )
            if not plan_preview_ok or not bool(plan_preview.get("ok")):
                blocked_steps.append(
                    {
                        "id": step_id,
                        "action": action,
                        "reason": reason or "planning_failed",
                        "error": str(plan_preview.get("error") or "plan_failed"),
                    }
                )
                continue

            decision = plan_preview.get("decision") if isinstance(plan_preview.get("decision"), dict) else {}
            if not bool(decision.get("allow")):
                blocked_steps.append(
                    {
                        "id": step_id,
                        "action": action,
                        "reason": str(decision.get("reason") or reason or "policy_deny"),
                        "error": "policy_deny",
                    }
                )
                continue

            execute_ok, execute_body = self.modelops_execute(
                {
                    "action": action,
                    "params": params,
                    "confirm": True,
                    "dry_run": False,
                    "actor": actor,
                }
            )
            if not execute_ok:
                failed_steps.append(
                    {
                        "id": step_id,
                        "action": action,
                        "reason": reason or "execution_failed",
                        "error": str(execute_body.get("error") or "execution_failed"),
                    }
                )
                continue
            executed_steps.append(
                {
                    "id": step_id,
                    "action": action,
                    "reason": reason,
                    "result": execute_body.get("result"),
                }
            )

        if failed_steps:
            message = "Some remediation steps failed during execution."
        elif blocked_steps:
            message = "Some remediation steps were blocked by policy or require user action."
        elif executed_steps:
            message = "Safe remediation steps executed."
        else:
            message = "No executable remediation steps were found."

        if executed_steps:
            self._record_memory_event(
                text=f"Remediation executed with {len(executed_steps)} safe step(s).",
                tags={
                    "project": "personal-agent",
                    "topic": "llm_remediation",
                },
                source_kind="api",
                source_ref="/llm/support/remediate/execute",
            )

        return (False if failed_steps else True), {
            "ok": False if failed_steps else True,
            "applied": bool(executed_steps) and not blocked_steps and not failed_steps,
            "message": message,
            "plan": plan,
            "executed_steps": executed_steps,
            "blocked_steps": blocked_steps,
            "failed_steps": failed_steps,
            "user_actions": user_actions,
        }

    def llm_notifications_mark_read(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        read_hash = str(payload.get("hash") or "").strip()
        if not read_hash:
            return False, {"ok": False, "error": "hash is required"}
        result = self._notification_store.mark_read(read_hash)
        if not bool(result.get("ok")):
            error = str(result.get("error") or "hash_not_found")
            status_code_error = "hash_not_found" if error == "hash_not_found" else "bad_request"
            return False, {"ok": False, "error": status_code_error}
        return True, {"ok": True, "status": self._notification_store.status()}

    def llm_notifications_prune(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        actor = str(payload.get("actor") or "user")
        start = time.monotonic()
        decision = self._modelops_permission_decision(
            "llm.notifications.prune",
            params={},
            estimated_download_bytes=0,
            estimated_cost=None,
            risk_level="low",
            dry_run=False,
        )
        if not bool(decision.get("allow")):
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.notifications.prune",
                params={},
                decision="deny",
                reason=str(decision.get("reason") or "action_not_permitted"),
                dry_run=False,
                outcome="blocked",
                error_kind=str(decision.get("reason") or "action_not_permitted"),
                duration_ms=duration_ms,
            )
            return False, {"ok": False, "error": str(decision.get("reason") or "action_not_permitted")}
        if bool(decision.get("requires_confirmation")) and not bool(payload.get("confirm", False)):
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.notifications.prune",
                params={},
                decision="deny",
                reason="confirmation_required",
                dry_run=False,
                outcome="blocked",
                error_kind="confirmation_required",
                duration_ms=duration_ms,
            )
            return False, {"ok": False, "error": "confirmation_required"}

        result = self._notification_store.prune_now()
        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor=actor,
            action="llm.notifications.prune",
            params={"result": result},
            decision="allow",
            reason="pruned",
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        return True, {"ok": True, "result": result}

    def llm_notifications_test(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        actor = str(payload.get("actor") or "user")
        start = time.monotonic()
        policy = compute_notification_test_policy(self)
        local_default_allow = bool(policy.get("allow_test_effective"))
        decision = self._modelops_permission_decision(
            "llm.notifications.test",
            params={},
            estimated_download_bytes=0,
            estimated_cost=None,
            risk_level="low",
            dry_run=False,
        )
        permission_allow = bool(decision["allow"])
        effective_allow = permission_allow or local_default_allow
        policy_reason = (
            "local_loopback_default"
            if local_default_allow and not permission_allow
            else str(decision.get("reason") or "policy_deny")
        )
        base_params = {
            "local_default_allow": local_default_allow,
            "allow_reason": str(policy.get("allow_reason") or ""),
            "permission_allow": permission_allow,
            "modified_ids": ["test:notification"],
        }

        if not effective_allow:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.notifications.test",
                params=base_params,
                decision="deny",
                reason=policy_reason,
                dry_run=False,
                outcome="blocked",
                error_kind=policy_reason,
                duration_ms=duration_ms,
            )
            return False, {"ok": False, "error": str(decision.get("reason") or "policy_deny")}
        if bool(decision["requires_confirmation"]) and not local_default_allow and not bool(payload.get("confirm", False)):
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.notifications.test",
                params=base_params,
                decision="deny",
                reason="confirmation_required",
                dry_run=False,
                outcome="blocked",
                error_kind="confirmation_required",
                duration_ms=duration_ms,
            )
            return False, {"ok": False, "error": "confirmation_required"}

        test_message = "LLM Autopilot updated configuration\n- Test notification\nReason: manual_test"
        try:
            record = self._record_autopilot_notification(
                message=test_message,
                dedupe_hash=f"test-{int(time.time())}",
                modified_ids=["test:notification"],
                trigger=actor,
                forced=True,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.notifications.test",
                params=base_params,
                decision="allow",
                reason=policy_reason,
                dry_run=False,
                outcome="failed",
                error_kind=exc.__class__.__name__,
                duration_ms=duration_ms,
            )
            return False, {"ok": False, "error": "notification_test_failed"}

        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor=actor,
            action="llm.notifications.test",
            params={
                **base_params,
                "delivered_to": str(record.get("delivered_to") or "none"),
                "deferred": bool(record.get("deferred", False)),
            },
            decision="allow",
            reason=policy_reason,
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        return True, {"ok": True, "result": record}

    def llm_health_summary(self) -> dict[str, Any]:
        summary = self._health_monitor.summary(self.registry_document)
        drift_report = self._current_drift_report(health_summary=summary)
        recent_actions = [
            entry
            for entry in self.audit_log.recent(limit=30)
            if isinstance(entry, dict)
            and str(entry.get("action") or "").strip()
            in {
                "llm.health.run",
                "llm.catalog.refresh",
                "llm.capabilities.reconcile.apply",
                "llm.cleanup.apply",
                "llm.autoconfig.apply",
                "llm.hygiene.apply",
                "llm.self_heal.apply",
            }
        ][:5]
        notifications = self._notification_store.recent(limit=1)
        latest_notification = notifications[0] if notifications else {}
        notifications_store_status = self._notification_store.status()
        last_notify_ts = self._last_notify_status.get("ts")
        last_notify_ts_iso = None
        if last_notify_ts:
            try:
                last_notify_ts_iso = datetime.fromtimestamp(int(last_notify_ts), tz=timezone.utc).isoformat()
            except (OSError, OverflowError, ValueError):
                last_notify_ts_iso = None
        summary["scheduler"] = {
            "enabled": bool(self.config.llm_automation_enabled),
            "next_refresh_run_at": int(self._scheduler_next_run.get("refresh") or 0) or None,
            "next_bootstrap_run_at": int(self._scheduler_next_run.get("bootstrap") or 0) or None,
            "next_catalog_run_at": int(self._scheduler_next_run.get("catalog") or 0) or None,
            "next_capabilities_reconcile_run_at": int(self._scheduler_next_run.get("capabilities_reconcile") or 0)
            or None,
            "next_health_run_at": int(self._scheduler_next_run.get("health") or 0) or None,
            "next_hygiene_run_at": int(self._scheduler_next_run.get("hygiene") or 0) or None,
            "next_cleanup_run_at": int(self._scheduler_next_run.get("cleanup") or 0) or None,
            "next_self_heal_run_at": int(self._scheduler_next_run.get("self_heal") or 0) or None,
            "next_model_scout_run_at": int(self._scheduler_next_run.get("model_scout") or 0) or None,
            "next_model_watch_run_at": int(self._scheduler_next_run.get("model_watch") or 0) or None,
            "next_autoconfig_run_at": int(self._scheduler_next_run.get("autoconfig") or 0) or None,
        }
        summary["last_actions"] = recent_actions
        summary["drift"] = drift_report
        summary["catalog"] = self._catalog_store.status()
        reconcile_plan = build_capabilities_reconcile_plan(self.registry_document, self._catalog_store.snapshot())
        summary["capabilities_reconcile"] = {
            "mismatch_count": int((reconcile_plan.get("impact") or {}).get("models_with_mismatch") or 0),
            "changes_count": int((reconcile_plan.get("impact") or {}).get("changes_count") or 0),
            "reasons": list(reconcile_plan.get("reasons") or []),
        }
        summary["notifications"] = {
            "last_outcome": self._last_notify_status.get("outcome"),
            "last_reason": self._last_notify_status.get("reason"),
            "last_hash": self._last_notify_status.get("dedupe_hash"),
            "last_ts": self._last_notify_status.get("ts"),
            "last_ts_iso": last_notify_ts_iso,
            "changed_defaults": int(self._last_notify_status.get("changed_defaults") or 0),
            "changed_providers": int(self._last_notify_status.get("changed_providers") or 0),
            "changed_models": int(self._last_notify_status.get("changed_models") or 0),
            "last_notification": latest_notification if isinstance(latest_notification, dict) else {},
            "store_status": notifications_store_status,
        }
        safety_state = self._safe_mode_status()
        summary["autopilot"] = {
            "safe_mode": bool(self._effective_safe_mode()),
            "safe_mode_config_default": bool(self.config.llm_autopilot_safe_mode),
            "safe_mode_override": bool(safety_state.get("safe_mode_override") is True),
            "safe_mode_reason": safety_state.get("safe_mode_reason"),
            "safe_mode_entered_ts": safety_state.get("safe_mode_entered_ts"),
            "safe_mode_entered_ts_iso": safety_state.get("safe_mode_entered_ts_iso"),
            "last_churn_event_ts": safety_state.get("last_churn_event_ts"),
            "last_churn_event_ts_iso": safety_state.get("last_churn_event_ts_iso"),
            "last_churn_reason": safety_state.get("last_churn_reason"),
            "last_blocked_reason": self._safe_mode_last_blocked_reason,
            "last_escalation_reason": self._safe_mode_last_escalation_reason,
            "rollback_policy": compute_registry_rollback_policy(self),
            "bootstrap_policy": compute_autopilot_bootstrap_apply_policy(self),
        }
        return {"ok": True, "health": summary}

    def _llm_status_payload(self, *, health_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        health_payload = health_summary if isinstance(health_summary, dict) else self.llm_health_summary()
        health = health_payload.get("health") if isinstance(health_payload.get("health"), dict) else {}
        snapshot = self._router.doctor_snapshot()
        defaults = self.get_defaults()
        document = self.registry_document if isinstance(self.registry_document, dict) else {}
        defaults_raw = self._ensure_defaults(document)
        models_document = document.get("models") if isinstance(document.get("models"), dict) else {}
        providers_document = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        provider_ids = {str(provider_id).strip().lower() for provider_id in providers_document.keys()}
        drift = health.get("drift") if isinstance(health.get("drift"), dict) else {}
        drift_details = drift.get("details") if isinstance(drift.get("details"), dict) else {}
        resolved_default_model = (
            str(drift_details.get("resolved_default_model") or "").strip()
            or self._resolved_chat_default_model(
                defaults=defaults_raw,
                models=models_document,
                provider_ids=provider_ids,
            )
            or None
        )
        default_provider = (
            str(defaults.get("default_provider") or "").strip().lower()
            or (
                str(resolved_default_model or "").split(":", 1)[0].strip().lower()
                if resolved_default_model and ":" in str(resolved_default_model)
                else None
            )
        )
        providers_raw = snapshot.get("providers") if isinstance(snapshot.get("providers"), list) else []
        models_raw = snapshot.get("models") if isinstance(snapshot.get("models"), list) else []
        providers, models = self._normalize_llm_status_rows(
            provider_rows=[dict(row) for row in providers_raw if isinstance(row, dict)],
            model_rows=[dict(row) for row in models_raw if isinstance(row, dict)],
        )
        allow_remote_fallback = bool(defaults.get("allow_remote_fallback", True))
        local_provider_lookup = {
            str(row.get("id") or "").strip().lower(): bool(row.get("local", False))
            for row in providers
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        hidden_models_by_provider: dict[str, int] = {}
        total_models_count = len(models)
        visible_models = models
        if not allow_remote_fallback:
            visible_models = []
            for row in models:
                provider_id = str(row.get("provider") or "").strip().lower()
                is_local_provider = bool(local_provider_lookup.get(provider_id, provider_id == "ollama"))
                if is_local_provider:
                    visible_models.append(row)
                else:
                    hidden_models_by_provider[provider_id] = int(hidden_models_by_provider.get(provider_id, 0)) + 1
        visible_models_count = len(visible_models)
        provider_lookup = {
            str(row.get("id") or "").strip().lower(): row
            for row in providers
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        model_lookup = {
            str(row.get("id") or "").strip(): row
            for row in visible_models
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        active_provider_row = provider_lookup.get(default_provider or "") if default_provider else None
        active_model_row = model_lookup.get(str(resolved_default_model or ""))
        active_provider_health = (
            active_provider_row.get("health")
            if isinstance(active_provider_row, dict) and isinstance(active_provider_row.get("health"), dict)
            else {}
        )
        active_provider_health = self._normalize_health_record(
            active_provider_health if isinstance(active_provider_health, dict) else {},
            now_epoch=int(time.time()),
        )
        active_model_health = (
            active_model_row.get("health")
            if isinstance(active_model_row, dict) and isinstance(active_model_row.get("health"), dict)
            else {}
        )
        active_model_health = self._normalize_health_record(
            active_model_health if isinstance(active_model_health, dict) else {},
            now_epoch=int(time.time()),
        )
        safe_mode = self._safe_mode_health_payload()
        visible_counts = {
            "total": visible_models_count,
            "ok": 0,
            "degraded": 0,
            "down": 0,
            "unknown": 0,
            "not_applicable": 0,
        }
        for row in visible_models:
            health_row = row.get("health") if isinstance(row.get("health"), dict) else {}
            status_key = str((health_row or {}).get("status") or "unknown").strip().lower() or "unknown"
            if status_key not in visible_counts:
                status_key = "unknown"
            visible_counts[status_key] = int(visible_counts.get(status_key, 0)) + 1
        return {
            "ok": True,
            "default_provider": default_provider,
            "chat_model": str(defaults.get("chat_model") or "").strip() or None,
            "embed_model": str(defaults.get("embed_model") or "").strip() or None,
            "last_chat_model": str(defaults.get("last_chat_model") or "").strip() or None,
            "default_model": defaults.get("default_model"),
            "resolved_default_model": resolved_default_model,
            "routing_mode": defaults.get("routing_mode"),
            "allow_remote_fallback": allow_remote_fallback,
            "safe_mode": safe_mode,
            "active_provider_health": active_provider_health,
            "active_model_health": active_model_health,
            "providers": providers,
            "models": visible_models,
            "hidden_models_by_provider": dict(sorted(hidden_models_by_provider.items())),
            "total_models_count": total_models_count,
            "visible_models_count": visible_models_count,
            "visible_counts": visible_counts,
        }

    def llm_status(self) -> dict[str, Any]:
        health_summary = self.llm_health_summary()
        status = self._llm_status_payload(health_summary=health_summary)
        status["health"] = health_summary.get("health") if isinstance(health_summary.get("health"), dict) else {}
        return status

    def public_llm_identity_string(self) -> str:
        try:
            status = self.llm_status()
        except Exception:
            identity = get_public_identity(provider=None, model=None, local_providers={"ollama"})
            return str(identity.get("summary") or "I’m running inside your Personal Agent. The active model is currently unknown.")
        if not isinstance(status, dict):
            identity = get_public_identity(provider=None, model=None, local_providers={"ollama"})
            return str(identity.get("summary") or "I’m running inside your Personal Agent. The active model is currently unknown.")
        provider = str(status.get("default_provider") or "").strip().lower()
        model = (
            str(status.get("resolved_default_model") or "").strip()
            or str(status.get("default_model") or "").strip()
        )
        local_providers = {
            str((row or {}).get("id") or "").strip().lower()
            for row in (status.get("providers") if isinstance(status.get("providers"), list) else [])
            if isinstance(row, dict) and bool(row.get("local", False))
        }
        if not local_providers:
            local_providers = {"ollama"}
        identity = get_public_identity(
            provider=provider if provider and provider != "unknown" else None,
            model=model if model and model != "unknown" else None,
            local_providers=local_providers,
        )
        return str(identity.get("summary") or "I’m running inside your Personal Agent. The active model is currently unknown.")

    @staticmethod
    def _value_policy_summary(policy: ValuePolicy) -> dict[str, Any]:
        return {
            "name": str(policy.name),
            "cost_cap_per_1m": round(float(policy.cost_cap_per_1m), 4),
            "allowlist": [str(item) for item in sorted(set(policy.allowlist))],
            "weights": {
                "quality": round(float(policy.quality_weight), 4),
                "price": round(float(policy.price_weight), 4),
                "latency": round(float(policy.latency_weight), 4),
                "instability": round(float(policy.instability_weight), 4),
            },
        }

    @staticmethod
    def _compact_model_watch_candidate(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row.get("id") or "").strip(),
            "provider": str(row.get("provider") or "").strip().lower() or None,
            "model": str(row.get("model") or "").strip() or None,
            "score": round(float(row.get("score") or 0.0), 2),
            "reason": str(row.get("reason") or "").strip() or None,
            "tradeoffs": [
                str(item).strip()
                for item in (row.get("tradeoffs") if isinstance(row.get("tradeoffs"), list) else [])
                if str(item).strip()
            ][:3],
            "utility_delta": (
                round(float(row.get("utility_delta") or 0.0), 4)
                if row.get("utility_delta") is not None
                else None
            ),
            "quality_delta": (
                round(float(row.get("quality_delta") or 0.0), 4)
                if row.get("quality_delta") is not None
                else None
            ),
            "expected_cost_delta": (
                round(float(row.get("expected_cost_delta") or 0.0), 4)
                if row.get("expected_cost_delta") is not None
                else None
            ),
        }

    def _latest_model_watch_summary(self) -> dict[str, Any]:
        latest_payload = self.model_watch_latest()
        latest_found = bool(latest_payload.get("found", False))
        batch = latest_payload.get("batch") if isinstance(latest_payload.get("batch"), dict) else {}
        top_pick = batch.get("top_pick") if isinstance(batch.get("top_pick"), dict) else {}
        other = batch.get("other_candidates") if isinstance(batch.get("other_candidates"), list) else []
        top_candidates: list[dict[str, Any]] = []
        if top_pick:
            top_candidates.append(self._compact_model_watch_candidate(top_pick))
        for row in other:
            if not isinstance(row, dict):
                continue
            if len(top_candidates) >= 4:
                break
            top_candidates.append(self._compact_model_watch_candidate(row))

        latest_event: dict[str, Any] | None = None
        for row in self.audit_log.recent(limit=200):
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").strip()
            if action in {
                "llm.model_watch.proposal_created",
                "llm.model_watch.hf_proposal_created",
                "llm.model_watch.no_change",
            }:
                latest_event = row
                break

        status = "unknown"
        event_action = str((latest_event or {}).get("action") or "").strip()
        if event_action in {"llm.model_watch.proposal_created", "llm.model_watch.hf_proposal_created"}:
            status = "proposal"
        elif event_action == "llm.model_watch.no_change":
            status = "no_change"
        elif latest_found:
            status = "latest_batch_available"

        params = (
            latest_event.get("params_redacted")
            if isinstance(latest_event, dict) and isinstance(latest_event.get("params_redacted"), dict)
            else {}
        )
        proposal = None
        if status == "proposal":
            proposal = {
                "from_model": str(params.get("from_model") or "").strip() or None,
                "to_model": str(params.get("to_model") or "").strip() or None,
                "provider": str(params.get("provider") or "").strip().lower() or None,
                "score_delta": (
                    round(float(params.get("score_delta") or 0.0), 4)
                    if params.get("score_delta") is not None
                    else None
                ),
                "utility_delta": (
                    round(float(params.get("utility_delta") or 0.0), 4)
                    if params.get("utility_delta") is not None
                    else None
                ),
                "quality_delta": (
                    round(float(params.get("quality_delta") or 0.0), 4)
                    if params.get("quality_delta") is not None
                    else None
                ),
                "expected_cost_delta": (
                    round(float(params.get("expected_cost_delta") or 0.0), 4)
                    if params.get("expected_cost_delta") is not None
                    else None
                ),
            }

        summary_line = "Model Watch: no summary available."
        if top_candidates:
            top = top_candidates[0]
            top_label = str(top.get("model") or top.get("id") or "unknown").strip()
            top_score = float(top.get("score") or 0.0)
            summary_line = f"Model Watch: top candidate {top_label} (score {top_score:.2f})."
        elif str(latest_payload.get("reason") or "").strip():
            summary_line = f"Model Watch: {str(latest_payload.get('reason') or '').strip()}."

        return {
            "status": status,
            "found": latest_found,
            "reason": str((latest_event or {}).get("reason") or latest_payload.get("reason") or "").strip() or None,
            "latest_event": {
                "ts": str((latest_event or {}).get("ts") or "").strip() or None,
                "action": event_action or None,
                "outcome": str((latest_event or {}).get("outcome") or "").strip() or None,
                "error_kind": str((latest_event or {}).get("error_kind") or "").strip() or None,
            },
            "proposal": proposal,
            "top_candidates": top_candidates,
            "summary_line": summary_line,
        }

    def model_status(self) -> dict[str, Any]:
        self._ensure_fresh_ollama_probe_state(ttl_seconds=30, timeout_seconds=2.0)
        status = self.llm_status()
        current_model = (
            str(status.get("resolved_default_model") or "").strip()
            or str(status.get("default_model") or "").strip()
            or None
        )
        current_provider = (
            str(status.get("default_provider") or "").strip().lower()
            or (
                str(current_model).split(":", 1)[0].strip().lower()
                if current_model and ":" in str(current_model)
                else None
            )
        )
        providers_rows = status.get("providers") if isinstance(status.get("providers"), list) else []
        providers_doc = (
            self.registry_document.get("providers")
            if isinstance(self.registry_document.get("providers"), dict)
            else {}
        )
        ollama_cfg = providers_doc.get("ollama") if isinstance(providers_doc.get("ollama"), dict) else {}
        configured_ollama_base = (
            str(ollama_cfg.get("base_url") or "").strip()
            or str(self.config.ollama_base_url or self.config.ollama_host or "").strip()
        )
        ollama_bases = normalize_ollama_base_urls(configured_ollama_base)
        cached_ollama_probe = self._ensure_fresh_ollama_probe_state(ttl_seconds=30, timeout_seconds=2.0)
        native_ok_value = bool(cached_ollama_probe.get("native_ok", False))
        local_up: list[str] = []
        local_down: list[str] = []
        local_unknown: list[str] = []
        ollama_provider_status = "unknown"
        ollama_provider_error_kind: str | None = None
        ollama_provider_status_code: int | None = None
        openrouter_payload = {
            "known": False,
            "status": "unknown",
            "last_error_kind": None,
            "status_code": None,
            "cooldown_until": None,
        }
        configured_provider_ids: list[str] = []
        for row in providers_rows:
            if not isinstance(row, dict):
                continue
            provider_id = str(row.get("id") or "").strip().lower()
            if not provider_id:
                continue
            configured_provider_ids.append(provider_id)
            snapshot_health = row.get("health") if isinstance(row.get("health"), dict) else {}
            health = self._effective_provider_health(provider_id, snapshot_health)
            health_status = str(health.get("status") or "unknown").strip().lower()
            if bool(row.get("local", False)):
                if provider_id == "ollama":
                    if health_status == "ok":
                        local_up.append(provider_id)
                    elif health_status == "down":
                        local_down.append(provider_id)
                    else:
                        local_unknown.append(provider_id)
                else:
                    if health_status == "ok":
                        local_up.append(provider_id)
                    elif health_status == "down":
                        local_down.append(provider_id)
                    else:
                        local_unknown.append(provider_id)
            if provider_id == "ollama":
                ollama_provider_status = health_status or "unknown"
                ollama_provider_error_kind = str(health.get("last_error_kind") or "").strip().lower() or None
                ollama_provider_status_code = (
                    int(health.get("status_code"))
                    if isinstance(health.get("status_code"), int)
                    else None
                )
            if provider_id == "openrouter":
                openrouter_payload = {
                    "known": True,
                    "status": health_status or "unknown",
                    "last_error_kind": str(health.get("last_error_kind") or "").strip() or None,
                    "status_code": (
                        int(health.get("status_code"))
                        if isinstance(health.get("status_code"), int)
                        else None
                    ),
                    "cooldown_until": (
                        int(health.get("cooldown_until"))
                        if isinstance(health.get("cooldown_until"), int)
                        else None
                    ),
                }

        llm_available = self.llm_availability_state()
        if ollama_provider_status == "ok":
            native_ok_value = True
        openai_compat_ok_value = bool(cached_ollama_probe.get("openai_compat_ok", False))
        ollama_last_error = (
            ollama_provider_error_kind
            or str(cached_ollama_probe.get("last_error_kind") or "").strip().lower()
            or None
        )
        ollama_last_status_code = (
            ollama_provider_status_code
            if ollama_provider_status_code is not None
            else (
                int(cached_ollama_probe.get("last_status_code"))
                if isinstance(cached_ollama_probe.get("last_status_code"), int)
                else None
            )
        )
        model_watch_summary = self._latest_model_watch_summary()
        return {
            "ok": True,
            "identity": self.public_llm_identity_string(),
            "current": {
                "provider": current_provider,
                "model_id": current_model,
                "default_provider": str(status.get("default_provider") or "").strip().lower() or None,
                "default_model": str(status.get("default_model") or "").strip() or None,
                "resolved_default_model": str(status.get("resolved_default_model") or "").strip() or None,
            },
            "selection_policy": {
                "default_policy": self._value_policy_summary(self._value_policy("default")),
                "premium_policy": self._value_policy_summary(self._value_policy("premium")),
                "premium_override": {
                    "once": bool(self._premium_override_once),
                    "until_ts": int(self._premium_override_until_ts) if self._premium_override_until_ts else None,
                },
            },
            "model_watch": model_watch_summary,
            "llm_availability": {
                "available": bool(llm_available.get("available", False)),
                "reason": str(llm_available.get("reason") or "unknown"),
                "providers": {
                    "configured": sorted({item for item in configured_provider_ids if item}),
                    "local_up": sorted({item for item in local_up if item}),
                    "local_down": sorted({item for item in local_down if item}),
                    "local_unknown": sorted({item for item in local_unknown if item}),
                },
                "ollama": {
                    "configured_base_url": str(ollama_bases.get("configured_base_url") or configured_ollama_base or "").strip(),
                    "native_base": str(ollama_bases.get("native_base") or "").strip(),
                    "openai_base": str(ollama_bases.get("openai_base") or "").strip(),
                    "native_ok": bool(native_ok_value),
                    "openai_compat_ok": bool(openai_compat_ok_value),
                    "last_error_kind": ollama_last_error,
                    "last_status_code": ollama_last_status_code,
                },
                "openrouter": openrouter_payload,
            },
        }

    def llm_availability_state(self) -> dict[str, Any]:
        try:
            status = self.llm_status()
        except Exception:
            return {"available": False, "reason": "llm_unavailable"}
        if not isinstance(status, dict):
            return {"available": False, "reason": "llm_unavailable"}
        safe_mode = status.get("safe_mode") if isinstance(status.get("safe_mode"), dict) else {}
        if bool(safe_mode.get("paused", False)):
            return {"available": False, "reason": "safe_mode_paused"}
        resolved_model = str(status.get("resolved_default_model") or "").strip()
        if not resolved_model:
            return {"available": False, "reason": "llm_unavailable"}
        provider_health = (
            status.get("active_provider_health")
            if isinstance(status.get("active_provider_health"), dict)
            else {}
        )
        model_health = (
            status.get("active_model_health")
            if isinstance(status.get("active_model_health"), dict)
            else {}
        )
        provider_state = str(provider_health.get("status") or "").strip().lower()
        model_state = str(model_health.get("status") or "").strip().lower()
        if provider_state != "ok":
            return {"available": False, "reason": "provider_unhealthy"}
        if model_state != "ok":
            return {"available": False, "reason": "model_unhealthy"}
        return {"available": True, "reason": "ok"}

    def llm_available(self) -> bool:
        state = self.llm_availability_state()
        return bool(state.get("available", False))

    def set_clarify_recovery_prompt(self, *, source: str, reason: str, ttl_seconds: int = 300) -> dict[str, Any]:
        now_epoch = int(time.time())
        ttl = max(30, int(ttl_seconds))
        state = {
            "active": True,
            "source": str(source or "").strip().lower() or "api",
            "reason": str(reason or "").strip().lower() or "llm_unavailable",
            "choices": recovery_options(),
            "created_ts": now_epoch,
            "expires_ts": now_epoch + ttl,
        }
        self._clarify_recovery_state = state
        return dict(state)

    def clear_clarify_recovery_prompt(self) -> None:
        self._clarify_recovery_state = {
            "active": False,
            "source": None,
            "reason": None,
            "choices": [],
            "created_ts": None,
            "expires_ts": None,
        }

    def _llm_status_summary_line(self) -> str:
        return self.public_llm_identity_string()

    def _ready_summary_line(self) -> str:
        ready = self.ready_payload()
        ready_bool = bool(ready.get("ready", False))
        phase = str(ready.get("phase") or "unknown").strip()
        telegram = ready.get("telegram") if isinstance(ready.get("telegram"), dict) else {}
        telegram_state = str(telegram.get("state") or "unknown").strip()
        return f"Ready: {'yes' if ready_bool else 'no'} (phase {phase}, telegram {telegram_state})"

    def consume_clarify_recovery_choice(self, *, source: str, text: str) -> tuple[bool, dict[str, Any]]:
        state = self._clarify_recovery_state if isinstance(self._clarify_recovery_state, dict) else {}
        if not bool(state.get("active", False)):
            return False, {}
        state_source = str(state.get("source") or "").strip().lower()
        if state_source and state_source != str(source or "").strip().lower():
            return False, {}
        now_epoch = int(time.time())
        expires_ts = int(state.get("expires_ts") or 0)
        if expires_ts and now_epoch > expires_ts:
            self.clear_clarify_recovery_prompt()
            return True, {
                "ok": True,
                "intent": "chat",
                "confidence": 0.0,
                "did_work": False,
                "error_kind": "needs_clarification",
                "message": "No active choice right now. Ask again and I will show options.",
                "next_question": "No active choice right now. Ask again and I will show options.",
                "actions": [],
                "errors": ["needs_clarification"],
            }
        choice = parse_recovery_choice(text, options=state.get("choices") if isinstance(state.get("choices"), list) else None)
        if not choice:
            return False, {}
        self.clear_clarify_recovery_prompt()
        if choice == "status":
            message = self._llm_status_summary_line()
            return True, {
                "ok": True,
                "intent": "chat",
                "confidence": 1.0,
                "did_work": True,
                "error_kind": None,
                "message": message,
                "next_question": None,
                "actions": [],
                "errors": [],
            }
        if choice == "fixit":
            ok, body = self.llm_fixit({"actor": "api"})
            message = str((body or {}).get("message") or (body or {}).get("next_question") or "").strip()
            if not message:
                message = "LLM fix-it is ready."
            return True, {
                "ok": bool(ok),
                "intent": "chat",
                "confidence": 1.0 if ok else 0.0,
                "did_work": bool(ok),
                "error_kind": str((body or {}).get("error_kind") or "") or None,
                "message": message,
                "next_question": str((body or {}).get("next_question") or "").strip() or None,
                "actions": [],
                "errors": [str((body or {}).get("error") or "fixit")] if not ok else [],
            }
        message = self._ready_summary_line()
        return True, {
            "ok": True,
            "intent": "chat",
            "confidence": 1.0,
            "did_work": True,
            "error_kind": None,
            "message": message,
            "next_question": None,
            "actions": [],
            "errors": [],
        }

    def _openrouter_secret_present(self) -> bool:
        providers = self.registry_document.get("providers") if isinstance(self.registry_document.get("providers"), dict) else {}
        openrouter = providers.get("openrouter") if isinstance(providers.get("openrouter"), dict) else None
        if not isinstance(openrouter, dict):
            return False
        return bool(self._provider_api_key(openrouter))

    def _summarize_provider_test_result(
        self,
        *,
        provider_id: str,
        ok: bool,
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = body if isinstance(body, dict) else {}
        error_kind = str(payload.get("error_kind") or payload.get("error") or "").strip().lower() or None
        try:
            status_code = int(payload.get("status_code") or 0) or None
        except (TypeError, ValueError):
            status_code = None
        if ok:
            human_reason = "Provider test succeeded."
        else:
            provider_message = str(payload.get("message") or "").strip()
            normalized_kind = self._normalize_provider_test_error(error_kind, status_code)
            human_reason = provider_message or self._provider_test_message(normalized_kind)
        return {
            "provider": str(provider_id or "").strip().lower(),
            "ts": int(time.time()),
            "ok": bool(ok),
            "status_code": status_code,
            "error_kind": error_kind,
            "human_reason": human_reason,
        }

    def _execute_llm_fixit_plan(
        self,
        *,
        plan_rows: list[dict[str, Any]],
        actor: str,
    ) -> tuple[bool, dict[str, Any]]:
        executed: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        provider_tests: dict[str, dict[str, Any]] = {}
        for row in sorted(
            [item for item in plan_rows if isinstance(item, dict)],
            key=lambda item: str(item.get("id") or ""),
        ):
            action = str(row.get("action") or "").strip()
            params = row.get("params") if isinstance(row.get("params"), dict) else {}
            safe_to_execute = bool(row.get("safe_to_execute", False))
            if not safe_to_execute:
                blocked.append(
                    {
                        "id": str(row.get("id") or ""),
                        "action": action,
                        "reason": str(row.get("reason") or "manual_step"),
                        "error": "user_action_required",
                    }
                )
                continue

            if action == "provider.set_enabled":
                provider_id = str(params.get("provider") or "").strip().lower()
                if not provider_id:
                    failed.append({"id": str(row.get("id") or ""), "action": action, "error": "provider_required"})
                    continue
                ok, body = self.update_provider(provider_id, {"enabled": bool(params.get("enabled", False))})
                if not ok:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": str((body or {}).get("error") or "update_failed"),
                        }
                    )
                    continue
                executed.append({"id": str(row.get("id") or ""), "action": action, "provider": provider_id})
                continue

            if action == "defaults.set":
                ok, body = self.update_defaults(params)
                if not ok:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": str((body or {}).get("error") or "update_failed"),
                        }
                    )
                    continue
                executed.append(
                    {
                        "id": str(row.get("id") or ""),
                        "action": action,
                        "defaults": {
                            "default_provider": body.get("default_provider"),
                            "default_model": body.get("default_model"),
                            "allow_remote_fallback": body.get("allow_remote_fallback"),
                        },
                    }
                )
                continue

            if action == "defaults.rollback":
                ok, body = self.rollback_defaults()
                if not ok:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": str((body or {}).get("error") or "rollback_failed"),
                            "error_kind": str((body or {}).get("error_kind") or (body or {}).get("error") or ""),
                            "message": str((body or {}).get("message") or ""),
                        }
                    )
                    continue
                executed.append(
                    {
                        "id": str(row.get("id") or ""),
                        "action": action,
                        "rolled_back_from": (body or {}).get("rolled_back_from"),
                        "rolled_back_to": (body or {}).get("rolled_back_to"),
                        "defaults": {
                            "chat_model": (body or {}).get("chat_model"),
                            "last_chat_model": (body or {}).get("last_chat_model"),
                            "resolved_default_model": (body or {}).get("resolved_default_model"),
                        },
                    }
                )
                continue

            if action == "autopilot.unpause":
                ok, body = self.llm_autopilot_unpause({"confirm": True, "actor": actor})
                if not ok:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": str((body or {}).get("error") or "unpause_failed"),
                        }
                    )
                    continue
                executed.append({"id": str(row.get("id") or ""), "action": action})
                continue

            if action == "provider.test":
                provider_id = str(params.get("provider") or "").strip().lower()
                if not provider_id:
                    failed.append({"id": str(row.get("id") or ""), "action": action, "error": "provider_required"})
                    continue
                provider_test_payload: dict[str, Any] = {}
                model_override = str(params.get("model") or "").strip()
                if model_override:
                    provider_test_payload["model"] = model_override
                timeout_override = params.get("timeout_seconds")
                if timeout_override is not None:
                    provider_test_payload["timeout_seconds"] = timeout_override
                ok, body = self.test_provider(provider_id, provider_test_payload)
                provider_tests[provider_id] = self._summarize_provider_test_result(
                    provider_id=provider_id,
                    ok=ok,
                    body=body,
                )
                if not ok:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": str((body or {}).get("error") or "provider_test_failed"),
                            "provider": provider_id,
                            "error_kind": str((body or {}).get("error_kind") or (body or {}).get("error") or ""),
                            "status_code": (body or {}).get("status_code"),
                            "message": str((body or {}).get("message") or ""),
                        }
                    )
                    continue
                executed.append({"id": str(row.get("id") or ""), "action": action, "provider": provider_id})
                continue

            if action == "ollama.pull_model":
                model_name = str(params.get("model") or "").strip()
                if not model_name:
                    failed.append({"id": str(row.get("id") or ""), "action": action, "error": "model_required"})
                    continue
                ok, body = self.pull_ollama_model({"model": model_name})
                if not ok:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": str((body or {}).get("error") or "ollama_pull_failed"),
                            "error_kind": str((body or {}).get("error_kind") or (body or {}).get("error") or ""),
                            "status_code": (body or {}).get("status_code"),
                            "message": str((body or {}).get("message") or ""),
                            "model": model_name,
                        }
                    )
                    continue
                executed.append(
                    {
                        "id": str(row.get("id") or ""),
                        "action": action,
                        "model": str((body or {}).get("model") or model_name),
                        "already_present": bool((body or {}).get("already_present", False)),
                    }
                )
                continue

            if action == "health.run":
                ok, body = self.run_llm_health(trigger="manual")
                if not ok:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": str((body or {}).get("error") or "health_run_failed"),
                        }
                    )
                    continue
                executed.append({"id": str(row.get("id") or ""), "action": action})
                continue

            if action == "hf.snapshot_download":
                repo_id = str(params.get("repo_id") or "").strip()
                revision = str(params.get("revision") or "").strip() or "main"
                target_dir = str(params.get("target_dir") or "").strip()
                allow_patterns = (
                    [str(item).strip() for item in params.get("allow_patterns", []) if str(item).strip()]
                    if isinstance(params.get("allow_patterns"), list)
                    else []
                )
                if not repo_id or not target_dir:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": "repo_id_and_target_dir_required",
                        }
                    )
                    continue
                try:
                    downloaded_path = hf_snapshot_download(
                        repo_id=repo_id,
                        revision=revision,
                        target_dir=target_dir,
                        allow_patterns=allow_patterns,
                    )
                except Exception as exc:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": f"snapshot_download_failed:{exc.__class__.__name__}",
                        }
                    )
                    continue
                executed.append(
                    {
                        "id": str(row.get("id") or ""),
                        "action": action,
                        "repo_id": repo_id,
                        "revision": revision,
                        "download_path": str(downloaded_path),
                    }
                )
                continue

            if action == "hf.generate_modelfile":
                selected_gguf = str(params.get("selected_gguf") or "").strip()
                target_dir = str(params.get("target_dir") or "").strip()
                modelfile_path = str(params.get("modelfile_path") or "").strip()
                if not selected_gguf or not target_dir or not modelfile_path:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": "selected_gguf_target_dir_modelfile_path_required",
                        }
                    )
                    continue
                gguf_path = (Path(target_dir).expanduser().resolve() / selected_gguf).resolve()
                if not gguf_path.is_file():
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": "selected_gguf_missing",
                            "gguf_path": str(gguf_path),
                        }
                    )
                    continue
                modelfile_target = Path(modelfile_path).expanduser().resolve()
                modelfile_target.parent.mkdir(parents=True, exist_ok=True)
                modelfile_content = f"FROM {str(gguf_path)}\n"
                fd = -1
                tmp_path = ""
                try:
                    fd, tmp_path = tempfile.mkstemp(
                        prefix=f".{modelfile_target.name}.",
                        suffix=".tmp",
                        dir=str(modelfile_target.parent),
                    )
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        handle.write(modelfile_content)
                        handle.flush()
                        os.fsync(handle.fileno())
                    fd = -1
                    os.replace(tmp_path, modelfile_target)
                    tmp_path = ""
                except Exception as exc:
                    if fd >= 0:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                    if tmp_path:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": f"modelfile_write_failed:{exc.__class__.__name__}",
                        }
                    )
                    continue
                executed.append(
                    {
                        "id": str(row.get("id") or ""),
                        "action": action,
                        "modelfile_path": str(modelfile_target),
                    }
                )
                continue

            if action == "hf.ollama_create":
                modelfile_path = str(params.get("modelfile_path") or "").strip()
                ollama_model_name = str(params.get("ollama_model_name") or "").strip()
                if not modelfile_path or not ollama_model_name:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": "modelfile_path_and_ollama_model_name_required",
                        }
                    )
                    continue
                try:
                    completed = subprocess.run(
                        ["ollama", "create", ollama_model_name, "-f", modelfile_path],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=900,
                    )
                except Exception as exc:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": f"ollama_create_failed:{exc.__class__.__name__}",
                            "model": ollama_model_name,
                        }
                    )
                    continue
                if int(completed.returncode) != 0:
                    stderr_text = str((completed.stderr or "").strip() or "ollama_create_failed")
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": "ollama_create_failed",
                            "model": ollama_model_name,
                            "detail": stderr_text[:240],
                        }
                    )
                    continue
                executed.append(
                    {
                        "id": str(row.get("id") or ""),
                        "action": action,
                        "model": ollama_model_name,
                    }
                )
                continue

            if action == "hf.refresh_ollama_registry":
                ok, body = self.refresh_models({"provider": "ollama"})
                if not ok:
                    failed.append(
                        {
                            "id": str(row.get("id") or ""),
                            "action": action,
                            "error": str((body or {}).get("error") or "refresh_models_failed"),
                        }
                    )
                    continue
                executed.append({"id": str(row.get("id") or ""), "action": action})
                continue

            if action == "hf.mark_download_only":
                repo_id = str(params.get("repo_id") or "").strip()
                revision = str(params.get("revision") or "").strip()
                target_dir = str(params.get("target_dir") or "").strip()
                if target_dir:
                    try:
                        marker_path = (Path(target_dir).expanduser().resolve() / ".personal-agent-download-only.json").resolve()
                        marker_path.parent.mkdir(parents=True, exist_ok=True)
                        marker_payload = {
                            "repo_id": repo_id,
                            "revision": revision,
                            "status": "download_only",
                        }
                        marker_bytes = (
                            json.dumps(marker_payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
                        ).encode("utf-8")
                        marker_path.write_bytes(marker_bytes)
                    except Exception as exc:
                        failed.append(
                            {
                                "id": str(row.get("id") or ""),
                                "action": action,
                                "error": f"download_marker_failed:{exc.__class__.__name__}",
                            }
                        )
                        continue
                executed.append(
                    {
                        "id": str(row.get("id") or ""),
                        "action": action,
                        "repo_id": repo_id,
                        "revision": revision,
                        "target_dir": target_dir or None,
                    }
                )
                continue

            blocked.append(
                {
                    "id": str(row.get("id") or ""),
                    "action": action,
                    "reason": str(row.get("reason") or "unsupported_action"),
                    "error": "unsupported_action",
                }
            )

        ok = not failed
        return ok, {
            "ok": ok,
            "executed_steps": executed,
            "blocked_steps": blocked,
            "failed_steps": failed,
            "provider_tests": provider_tests,
        }

    def llm_fixit(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        actor = str(payload.get("actor") or "user").strip() or "user"
        now_epoch = int(time.time())
        wizard_state = self._llm_fixit_store.state if isinstance(self._llm_fixit_store.state, dict) else self._llm_fixit_store.empty_state()
        openrouter_last_test = (
            wizard_state.get("openrouter_last_test")
            if isinstance(wizard_state.get("openrouter_last_test"), dict)
            else None
        )
        decision_context = {
            "openrouter_secret_present": self._openrouter_secret_present(),
            "openrouter_last_test": openrouter_last_test,
        }
        status_payload = self.llm_status()
        decision = evaluate_wizard_decision(status_payload, context=decision_context)
        issue_hash = wizard_decision_issue_hash(decision, status_payload)

        def _save_idle_state(*, last_test: dict[str, Any] | None) -> None:
            self._llm_fixit_store.save(
                {
                    "active": False,
                    "issue_hash": None,
                    "issue_code": None,
                    "step": "idle",
                    "question": None,
                    "choices": [],
                    "pending_plan": [],
                    "pending_confirm_token": None,
                    "pending_created_ts": None,
                    "pending_expires_ts": None,
                    "pending_issue_code": None,
                    "last_prompt_ts": now_epoch,
                    "openrouter_last_test": last_test,
                }
            )

        def _save_choice_state(
            *,
            decision_obj: WizardDecision,
            status_obj: dict[str, Any],
            last_test: dict[str, Any] | None,
            step: str = "awaiting_choice",
            question_override: str | None = None,
        ) -> tuple[str, list[dict[str, Any]]]:
            prompt = render_wizard_prompt(decision_obj)
            choices_json = wizard_decision_to_json(decision_obj).get("choices", [])
            issue_hash_value = wizard_decision_issue_hash(decision_obj, status_obj)
            self._llm_fixit_store.save(
                {
                    "active": True,
                    "issue_hash": issue_hash_value,
                    "issue_code": decision_obj.issue_code,
                    "step": step,
                    "question": str(question_override or decision_obj.question or ""),
                    "choices": choices_json,
                    "pending_plan": [],
                    "pending_confirm_token": None,
                    "pending_created_ts": None,
                    "pending_expires_ts": None,
                    "pending_issue_code": None,
                    "last_prompt_ts": now_epoch,
                    "openrouter_last_test": last_test,
                }
            )
            return prompt, choices_json

        openrouter_api_key = str(payload.get("openrouter_api_key") or "").strip()
        if openrouter_api_key:
            ok_secret, secret_body = self.set_provider_secret("openrouter", {"api_key": openrouter_api_key})
            if not ok_secret:
                return True, {
                    "ok": True,
                    "intent": "llm_fixit",
                    "confidence": 0.0,
                    "did_work": False,
                    "error_kind": "needs_clarification",
                    "message": str(secret_body.get("error") or "Unable to save OpenRouter key."),
                    "next_question": "Please send a valid OpenRouter API key.",
                    "actions": [],
                    "errors": [str(secret_body.get("error") or "needs_clarification")],
                    "status": "needs_clarification",
                    "issue_code": decision.issue_code,
                    "trace_id": f"fixit-{now_epoch}",
                }
            ok_test, test_body = self.test_provider("openrouter", {})
            test_summary = self._summarize_provider_test_result(
                provider_id="openrouter",
                ok=ok_test,
                body=test_body if isinstance(test_body, dict) else {},
            )
            status_after = self.llm_status()
            decision_after = evaluate_wizard_decision(
                status_after,
                context={
                    "openrouter_secret_present": self._openrouter_secret_present(),
                    "openrouter_last_test": test_summary,
                },
            )
            if decision_after.status == "ok":
                _save_idle_state(last_test=test_summary)
                return True, {
                    "ok": True,
                    "intent": "llm_fixit",
                    "confidence": 1.0,
                    "did_work": True,
                    "error_kind": None,
                    "message": "OpenRouter key saved and provider test succeeded.",
                    "next_question": None,
                    "actions": [],
                    "errors": [],
                    "status": "ok",
                    "issue_code": decision_after.issue_code,
                    "openrouter_last_test": test_summary,
                    "trace_id": f"fixit-{now_epoch}",
                }
            prompt_after, choices_after = _save_choice_state(
                decision_obj=decision_after,
                status_obj=status_after,
                last_test=test_summary,
            )
            return True, {
                "ok": True,
                "intent": "llm_fixit",
                "confidence": 0.0,
                "did_work": True,
                "error_kind": "needs_clarification",
                "message": prompt_after,
                "next_question": str(decision_after.question or ""),
                "actions": [],
                "errors": ["needs_clarification"],
                "status": "needs_user_choice",
                "issue_code": decision_after.issue_code,
                "choices": choices_after,
                "openrouter_last_test": test_summary,
                "trace_id": f"fixit-{now_epoch}",
            }

        explicit_confirm_flag = "confirm" in payload
        confirm = bool(payload.get("confirm", False))
        answer_value = str(payload.get("answer") or "").strip().lower()
        wizard_step = str(wizard_state.get("step") or "").strip().lower()
        cancel_answers = {"cancel", "no"}
        if wizard_step == "awaiting_confirm":
            cancel_answers.add("2")
        if (explicit_confirm_flag and not confirm) or answer_value in cancel_answers:
            _save_idle_state(last_test=openrouter_last_test)
            return True, {
                "ok": True,
                "intent": "llm_fixit",
                "confidence": 1.0,
                "did_work": False,
                "error_kind": None,
                "message": "Cancelled.",
                "next_question": None,
                "actions": [],
                "errors": [],
                "status": "cancelled",
                "issue_code": str(wizard_state.get("issue_code") or decision.issue_code),
                "trace_id": f"fixit-{now_epoch}",
            }

        if confirm:
            pending_plan = wizard_state.get("pending_plan") if isinstance(wizard_state.get("pending_plan"), list) else []
            expected_token = (
                str(wizard_state.get("pending_confirm_token") or "").strip()
                or str(wizard_state.get("confirm_token") or "").strip()
            )
            if not pending_plan or not expected_token:
                return True, {
                    "ok": True,
                    "intent": "llm_fixit",
                    "confidence": 0.0,
                    "did_work": False,
                    "error_kind": "needs_clarification",
                    "message": "No pending plan to confirm.",
                    "next_question": "Ask me to run fix-it again.",
                    "actions": [],
                    "errors": ["needs_clarification"],
                    "issue_code": str(wizard_state.get("issue_code") or decision.issue_code),
                    "choices": [],
                    "trace_id": f"fixit-{now_epoch}",
                }

            pending_expires_ts = int(wizard_state.get("pending_expires_ts") or 0)
            if pending_expires_ts > 0 and now_epoch > pending_expires_ts:
                self._llm_fixit_store.clear()
                return True, {
                    "ok": True,
                    "intent": "llm_fixit",
                    "confidence": 0.0,
                    "did_work": False,
                    "error_kind": "needs_clarification",
                    "message": "That confirmation expired. Ask me to fix it again.",
                    "next_question": "Run fix-it again now?",
                    "actions": [],
                    "errors": ["needs_clarification"],
                    "issue_code": str(wizard_state.get("issue_code") or decision.issue_code),
                    "choices": [],
                    "trace_id": f"fixit-{now_epoch}",
                }

            recomputed_token = wizard_confirm_token_for_plan_rows(
                [row for row in pending_plan if isinstance(row, dict)]
            )
            if recomputed_token != expected_token:
                self._llm_fixit_store.clear()
                return True, {
                    "ok": True,
                    "intent": "llm_fixit",
                    "confidence": 0.0,
                    "did_work": False,
                    "error_kind": "needs_clarification",
                    "message": "Pending plan changed. Ask me to fix it again.",
                    "next_question": "Run fix-it again now?",
                    "actions": [],
                    "errors": ["needs_clarification"],
                    "issue_code": str(wizard_state.get("issue_code") or decision.issue_code),
                    "choices": [],
                    "trace_id": f"fixit-{now_epoch}",
                }

            provided_token = str(payload.get("confirm_token") or "").strip()
            if provided_token and provided_token != expected_token:
                return True, {
                    "ok": True,
                    "intent": "llm_fixit",
                    "confidence": 0.0,
                    "did_work": False,
                    "error_kind": "needs_clarification",
                    "message": "Confirmation token did not match the pending plan.",
                    "next_question": "Confirm the latest pending plan?",
                    "actions": [],
                    "errors": ["needs_clarification"],
                    "issue_code": str(wizard_state.get("issue_code") or decision.issue_code),
                    "choices": [],
                    "trace_id": f"fixit-{now_epoch}",
                }

            ok_exec, exec_payload = self._execute_llm_fixit_plan(
                plan_rows=[row for row in pending_plan if isinstance(row, dict)],
                actor=actor,
            )
            install_target_model: str | None = None
            rollback_target_model: str | None = None
            for row in [item for item in pending_plan if isinstance(item, dict)]:
                action_name = str(row.get("action") or "").strip()
                if action_name == "ollama.pull_model":
                    model_value = self._normalize_ollama_pull_model(str((row.get("params") or {}).get("model") or ""))
                    if model_value:
                        install_target_model = model_value
                if action_name == "defaults.rollback":
                    rollback_value = str((row.get("params") or {}).get("target_chat_model") or "").strip()
                    if rollback_value:
                        rollback_target_model = rollback_value
            provider_tests = exec_payload.get("provider_tests") if isinstance(exec_payload.get("provider_tests"), dict) else {}
            openrouter_test = (
                provider_tests.get("openrouter")
                if isinstance(provider_tests.get("openrouter"), dict)
                else openrouter_last_test
            )
            if isinstance(openrouter_test, dict):
                status_after = self.llm_status()
                decision_after = evaluate_wizard_decision(
                    status_after,
                    context={
                        "openrouter_secret_present": self._openrouter_secret_present(),
                        "openrouter_last_test": openrouter_test,
                    },
                )
                if decision_after.status != "ok":
                    prompt_after, choices_after = _save_choice_state(
                        decision_obj=decision_after,
                        status_obj=status_after,
                        last_test=openrouter_test,
                    )
                    return True, {
                        "ok": True,
                        "intent": "llm_fixit",
                        "confidence": 0.0,
                        "did_work": bool(exec_payload.get("executed_steps")),
                        "error_kind": "needs_clarification",
                        "message": prompt_after,
                        "next_question": str(decision_after.question or ""),
                        "actions": [],
                        "errors": ["needs_clarification"],
                        "status": "needs_user_choice",
                        "issue_code": decision_after.issue_code,
                        "choices": choices_after,
                        "result": exec_payload,
                        "openrouter_last_test": openrouter_test,
                        "trace_id": f"fixit-{now_epoch}",
                    }
            _save_idle_state(last_test=openrouter_test if isinstance(openrouter_test, dict) else None)
            failed_steps = exec_payload.get("failed_steps") if isinstance(exec_payload.get("failed_steps"), list) else []
            ollama_pull_failure = next(
                (
                    row for row in failed_steps
                    if isinstance(row, dict) and str(row.get("action") or "").strip() == "ollama.pull_model"
                ),
                None,
            )
            rollback_step = next(
                (
                    row for row in (
                        exec_payload.get("executed_steps")
                        if isinstance(exec_payload.get("executed_steps"), list)
                        else []
                    )
                    if isinstance(row, dict) and str(row.get("action") or "").strip() == "defaults.rollback"
                ),
                None,
            )
            defaults_after = self.get_defaults() if install_target_model and ok_exec else {}
            if ok_exec and install_target_model:
                chosen_model = str(defaults_after.get("chat_model") or f"ollama:{install_target_model}")
                message = f"Installed and configured {chosen_model} for chat."
                next_question = None
                error_kind = None
                errors_payload: list[str] = []
            elif ok_exec and isinstance(rollback_step, dict):
                rolled_back_to = str(
                    rollback_step.get("rolled_back_to")
                    or rollback_target_model
                    or self.get_defaults().get("chat_model")
                    or ""
                ).strip()
                rolled_back_from = str(rollback_step.get("rolled_back_from") or "").strip() or None
                message = (
                    f"✅ Rolled back chat model to {rolled_back_to} (was {rolled_back_from})"
                    if rolled_back_from
                    else f"✅ Rolled back chat model to {rolled_back_to}"
                )
                next_question = None
                error_kind = None
                errors_payload = []
            elif isinstance(ollama_pull_failure, dict):
                failure_kind = str(
                    ollama_pull_failure.get("error_kind") or ollama_pull_failure.get("error") or ""
                ).strip().lower()
                if failure_kind in {"ollama_unavailable", "timeout"}:
                    message = "Ollama is unreachable right now. Start Ollama, then run fix-it again."
                    next_question = "After Ollama is running, should I retry local model install?"
                    error_kind = "upstream_down"
                    errors_payload = [failure_kind or "ollama_unavailable"]
                else:
                    message = "Applied partial fixes; some steps still need attention."
                    next_question = "Do you want the detailed failed steps?"
                    error_kind = "upstream_down"
                    errors_payload = ["execution_partial_failure"]
            else:
                message = "Applied safe LLM fixes." if ok_exec else "Applied partial fixes; some steps still need attention."
                next_question = None if ok_exec else "Do you want the detailed failed steps?"
                error_kind = None if ok_exec else "upstream_down"
                errors_payload = [] if ok_exec else ["execution_partial_failure"]
            response_payload = {
                "ok": ok_exec,
                "intent": "llm_fixit",
                "confidence": 1.0 if ok_exec else 0.0,
                "did_work": bool(exec_payload.get("executed_steps")),
                "error_kind": error_kind,
                "message": message,
                "next_question": next_question,
                "actions": [],
                "errors": errors_payload,
                "issue_code": str(wizard_state.get("issue_code") or decision.issue_code),
                "result": exec_payload,
                "trace_id": f"fixit-{now_epoch}",
            }
            if ok_exec and install_target_model:
                response_payload["chat_model"] = defaults_after.get("chat_model")
                response_payload["resolved_default_model"] = defaults_after.get("resolved_default_model")
            if ok_exec and isinstance(rollback_step, dict):
                response_payload["chat_model"] = rollback_step.get("rolled_back_to") or self.get_defaults().get("chat_model")
                response_payload["last_chat_model"] = rollback_step.get("rolled_back_from") or self.get_defaults().get("last_chat_model")
            return ok_exec, response_payload

        answer = str(payload.get("answer") or "").strip()
        if answer:
            state_choices_json = (
                wizard_state.get("choices")
                if isinstance(wizard_state.get("choices"), list)
                else []
            )
            use_state_choices = bool(wizard_state.get("active")) and str(wizard_state.get("step") or "").strip().lower() == "awaiting_choice" and bool(state_choices_json)
            if use_state_choices:
                choices_json = [
                    {
                        "id": str(row.get("id") or "").strip(),
                        "label": str(row.get("label") or "").strip(),
                        "recommended": bool(row.get("recommended", False)),
                    }
                    for row in state_choices_json
                    if isinstance(row, dict)
                    and str(row.get("id") or "").strip()
                    and str(row.get("label") or "").strip()
                ]
                choices = [
                    WizardChoice(
                        id=str(row.get("id") or "").strip(),
                        label=str(row.get("label") or "").strip(),
                        recommended=bool(row.get("recommended", False)),
                    )
                    for row in choices_json
                ]
                issue_code_for_choice = str(wizard_state.get("issue_code") or decision.issue_code)
            else:
                choices = decision.choices
                choices_json = wizard_decision_to_json(decision).get("choices", [])
                issue_code_for_choice = decision.issue_code
            selected = wizard_parse_choice_answer(answer, choices)
            if not selected:
                return True, {
                    "ok": True,
                    "intent": "llm_fixit",
                    "confidence": 0.0,
                    "did_work": False,
                    "error_kind": "needs_clarification",
                    "message": "I did not recognize that option.",
                    "next_question": "Please reply with 1, 2, or 3.",
                    "actions": [],
                    "errors": ["needs_clarification"],
                    "issue_code": issue_code_for_choice,
                    "choices": choices_json,
                    "trace_id": f"fixit-{now_epoch}",
                }
            if issue_code_for_choice == "model_watch.proposal":
                state_proposal_type = str(wizard_state.get("proposal_type") or "").strip().lower()
                if selected == "snooze_model_watch":
                    _save_idle_state(last_test=openrouter_last_test)
                    return True, {
                        "ok": True,
                        "intent": "llm_fixit",
                        "confidence": 1.0,
                        "did_work": False,
                        "error_kind": None,
                        "message": "Okay, I will keep the current model for now.",
                        "next_question": None,
                        "actions": [],
                        "errors": [],
                        "status": "snoozed",
                        "issue_code": issue_code_for_choice,
                        "trace_id": f"fixit-{now_epoch}",
                    }
                if selected == "details":
                    details_message = (
                        str(wizard_state.get("proposal_details") or "").strip()
                        or "No additional model-watch details are available."
                    )
                    return True, {
                        "ok": True,
                        "intent": "llm_fixit",
                        "confidence": 0.0,
                        "did_work": False,
                        "error_kind": "needs_clarification",
                        "message": details_message,
                        "next_question": (
                            "Reply 1 to download/install, or 2 to snooze."
                            if state_proposal_type == "local_download"
                            else "Reply 1 to switch, or 2 to snooze."
                        ),
                        "actions": [],
                        "errors": ["needs_clarification"],
                        "status": "needs_user_choice",
                        "issue_code": issue_code_for_choice,
                        "choices": choices_json,
                        "trace_id": f"fixit-{now_epoch}",
                    }
                if selected in {"switch_to_proposal", "download_install_local"}:
                    pending_plan_rows = (
                        wizard_state.get("pending_plan")
                        if isinstance(wizard_state.get("pending_plan"), list)
                        else []
                    )
                    inferred_local_download = state_proposal_type == "local_download" or any(
                        str((row if isinstance(row, dict) else {}).get("action") or "").strip().startswith("hf.")
                        for row in pending_plan_rows
                    )
                    if not pending_plan_rows:
                        return True, {
                            "ok": True,
                            "intent": "llm_fixit",
                            "confidence": 0.0,
                            "did_work": False,
                            "error_kind": "needs_clarification",
                            "message": "No proposal plan is pending. Run fix-it again.",
                            "next_question": "Do you want me to check model recommendations again?",
                            "actions": [],
                            "errors": ["needs_clarification"],
                            "status": "needs_clarification",
                            "issue_code": issue_code_for_choice,
                            "trace_id": f"fixit-{now_epoch}",
                        }
                    confirm_token = wizard_confirm_token_for_plan_rows(
                        [row for row in pending_plan_rows if isinstance(row, dict)]
                    )
                    self._llm_fixit_store.save(
                        {
                            "active": True,
                            "issue_hash": str(wizard_state.get("issue_hash") or issue_hash),
                            "issue_code": issue_code_for_choice,
                            "step": "awaiting_confirm",
                            "question": (
                                "Apply this download/install plan now?"
                                if inferred_local_download
                                else "Apply this fix-it plan now?"
                            ),
                            "choices": choices_json,
                            "pending_plan": pending_plan_rows,
                            "pending_confirm_token": confirm_token,
                            "pending_created_ts": now_epoch,
                            "pending_expires_ts": now_epoch + 300,
                            "pending_issue_code": issue_code_for_choice,
                            "last_prompt_ts": now_epoch,
                            "openrouter_last_test": openrouter_last_test,
                            "proposal_type": state_proposal_type or ("local_download" if inferred_local_download else "switch_default"),
                            "proposal_details": str(wizard_state.get("proposal_details") or "").strip() or None,
                        }
                    )
                    return True, {
                        "ok": True,
                        "intent": "llm_fixit",
                        "confidence": 0.0,
                        "did_work": False,
                        "error_kind": "needs_clarification",
                        "message": (
                            "I prepared a safe download/install plan. Reply YES to apply, or NO to cancel."
                            if inferred_local_download
                            else "I prepared a safe fix plan. Reply YES to apply, or NO to cancel."
                        ),
                        "next_question": (
                            "Apply this download/install plan now? Reply YES or NO."
                            if inferred_local_download
                            else "Apply this fix-it plan now? Reply YES or NO."
                        ),
                        "actions": [],
                        "errors": ["needs_clarification"],
                        "status": "needs_confirmation",
                        "issue_code": issue_code_for_choice,
                        "confirm_code": 1,
                        "cancel_code": 2,
                        "plan": [row for row in pending_plan_rows if isinstance(row, dict)],
                        "trace_id": f"fixit-{now_epoch}",
                    }
            if issue_code_for_choice == "premium_over_cap":
                if selected == "continue_baseline":
                    self._premium_override_once = False
                    self._premium_override_until_ts = None
                    _save_idle_state(last_test=openrouter_last_test)
                    return True, {
                        "ok": True,
                        "intent": "llm_fixit",
                        "confidence": 1.0,
                        "did_work": False,
                        "error_kind": None,
                        "message": "Okay, I will keep using the baseline model.",
                        "next_question": None,
                        "actions": [],
                        "errors": [],
                        "status": "ok",
                        "issue_code": issue_code_for_choice,
                        "trace_id": f"fixit-{now_epoch}",
                    }
                if selected == "upgrade_once":
                    self._premium_override_once = True
                    self._premium_override_until_ts = None
                    _save_idle_state(last_test=openrouter_last_test)
                    return True, {
                        "ok": True,
                        "intent": "llm_fixit",
                        "confidence": 1.0,
                        "did_work": True,
                        "error_kind": None,
                        "message": "Understood. I will allow one premium upgrade on the next escalated request.",
                        "next_question": None,
                        "actions": [],
                        "errors": [],
                        "status": "ok",
                        "issue_code": issue_code_for_choice,
                        "trace_id": f"fixit-{now_epoch}",
                    }
                if selected == "set_premium_1h":
                    self._premium_override_once = False
                    self._premium_override_until_ts = now_epoch + 3600
                    _save_idle_state(last_test=openrouter_last_test)
                    return True, {
                        "ok": True,
                        "intent": "llm_fixit",
                        "confidence": 1.0,
                        "did_work": True,
                        "error_kind": None,
                        "message": "Understood. Premium upgrades are allowed for the next 1 hour.",
                        "next_question": None,
                        "actions": [],
                        "errors": [],
                        "status": "ok",
                        "issue_code": issue_code_for_choice,
                        "trace_id": f"fixit-{now_epoch}",
                    }
            if selected in {"add_openrouter_key", "update_openrouter_key"}:
                self._llm_fixit_store.save(
                    {
                        "active": True,
                        "issue_hash": issue_hash,
                        "issue_code": issue_code_for_choice,
                        "step": "awaiting_openrouter_key",
                        "question": "Paste your OpenRouter API key now.",
                        "choices": choices_json,
                        "pending_plan": [],
                        "pending_confirm_token": None,
                        "pending_created_ts": None,
                        "pending_expires_ts": None,
                        "pending_issue_code": None,
                        "last_prompt_ts": now_epoch,
                        "openrouter_last_test": openrouter_last_test,
                    }
                )
                return True, {
                    "ok": True,
                    "intent": "llm_fixit",
                    "confidence": 0.0,
                    "did_work": False,
                    "error_kind": "needs_clarification",
                    "message": "Paste your OpenRouter API key now.",
                    "next_question": "Send it in openrouter_api_key and I will test it.",
                    "actions": [],
                    "errors": ["needs_clarification"],
                    "status": "needs_clarification",
                    "issue_code": issue_code_for_choice,
                    "trace_id": f"fixit-{now_epoch}",
                }
            if selected == "rollback_chat_model":
                rollback_target = str(status_payload.get("last_chat_model") or "").strip() or None
                if not rollback_target:
                    return True, {
                        "ok": True,
                        "intent": "llm_fixit",
                        "confidence": 0.0,
                        "did_work": False,
                        "error_kind": "needs_clarification",
                        "message": "No rollback available yet.",
                        "next_question": "Choose another option or run fix-it again.",
                        "actions": [],
                        "errors": ["needs_clarification"],
                        "status": "needs_clarification",
                        "issue_code": issue_code_for_choice,
                        "choices": choices_json,
                        "trace_id": f"fixit-{now_epoch}",
                    }
            plan = wizard_build_plan_for_choice(
                issue_code=issue_code_for_choice,
                choice_id=selected,
                status_payload=status_payload,
            )
            if not plan:
                details_message = "No changes staged. I can show details or wait."
                return True, {
                    "ok": True,
                    "intent": "llm_fixit",
                    "confidence": 1.0,
                    "did_work": False,
                    "error_kind": None,
                    "message": details_message,
                    "next_question": None,
                    "actions": [],
                    "errors": [],
                    "issue_code": issue_code_for_choice,
                    "details": decision.details,
                    "trace_id": f"fixit-{now_epoch}",
                }
            confirm_token = wizard_confirm_token_for_plan(plan)
            plan_json = [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "action": item.action,
                    "reason": item.reason,
                    "params": dict(sorted((item.params or {}).items())),
                    "safe_to_execute": bool(item.safe_to_execute),
                }
                for item in plan
            ]
            install_step = next(
                (
                    row for row in plan_json
                    if str(row.get("action") or "").strip() == "ollama.pull_model"
                ),
                None,
            )
            rollback_step = next(
                (
                    row for row in plan_json
                    if str(row.get("action") or "").strip() == "defaults.rollback"
                ),
                None,
            )
            install_model_name = self._normalize_ollama_pull_model(
                str(
                    (
                        install_step.get("params")
                        if isinstance(install_step, dict) and isinstance(install_step.get("params"), dict)
                        else {}
                    ).get("model")
                    or ""
                )
            )
            rollback_target = str(
                (
                    rollback_step.get("params")
                    if isinstance(rollback_step, dict) and isinstance(rollback_step.get("params"), dict)
                    else {}
                ).get("target_chat_model")
                or ""
            ).strip() or None
            if rollback_target:
                confirm_question = f"I can roll back to {rollback_target}. Reply YES to apply, or NO to cancel."
            elif install_model_name:
                confirm_question = f"I can download {install_model_name}. Reply YES to apply, or NO to cancel."
            else:
                confirm_question = "I prepared a safe fix plan. Reply YES to apply, or NO to cancel."
            state_to_save = {
                "active": True,
                "issue_hash": issue_hash,
                "issue_code": issue_code_for_choice,
                "step": "awaiting_confirm",
                "question": (
                    f"Roll back chat model to {rollback_target} now?"
                    if rollback_target
                    else (
                    f"Apply local model install for {install_model_name} now?"
                    if install_model_name
                    else "Apply this fix-it plan now?"
                    )
                ),
                "choices": choices_json,
                "pending_plan": plan_json,
                "pending_confirm_token": confirm_token,
                "pending_created_ts": now_epoch,
                "pending_expires_ts": now_epoch + 300,
                "pending_issue_code": issue_code_for_choice,
                "last_prompt_ts": now_epoch,
                "openrouter_last_test": openrouter_last_test,
                "proposal_type": str(wizard_state.get("proposal_type") or "").strip().lower() or None,
                "proposal_details": str(wizard_state.get("proposal_details") or "").strip() or None,
            }
            self._llm_fixit_store.save(state_to_save)
            return True, {
                "ok": True,
                "intent": "llm_fixit",
                "confidence": 0.0,
                "did_work": False,
                "error_kind": "needs_clarification",
                "message": confirm_question,
                "next_question": (
                    f"Roll back chat model to {rollback_target}? Reply YES or NO."
                    if rollback_target
                    else (
                    f"Apply local model install for {install_model_name}? Reply YES or NO."
                    if install_model_name
                    else "Apply this fix-it plan now? Reply YES or NO."
                    )
                ),
                "actions": [],
                "errors": ["needs_clarification"],
                "status": "needs_confirmation",
                "issue_code": issue_code_for_choice,
                "confirm_code": 1,
                "cancel_code": 2,
                "plan": plan_json,
                "trace_id": f"fixit-{now_epoch}",
            }

        if decision.status == "ok":
            _save_idle_state(last_test=openrouter_last_test)
            return True, {
                "ok": True,
                "intent": "llm_fixit",
                "confidence": 1.0,
                "did_work": True,
                "error_kind": None,
                "message": decision.message,
                "next_question": None,
                "actions": [],
                "errors": [],
                "status": "ok",
                "issue_code": decision.issue_code,
                "trace_id": f"fixit-{now_epoch}",
            }

        prompt, choices_json = _save_choice_state(
            decision_obj=decision,
            status_obj=status_payload,
            last_test=openrouter_last_test,
        )
        return True, {
            "ok": True,
            "intent": "llm_fixit",
            "confidence": 0.0,
            "did_work": False,
            "error_kind": "needs_clarification",
            "message": prompt,
            "next_question": str(decision.question or ""),
            "actions": [],
            "errors": ["needs_clarification"],
            "status": "needs_user_choice",
            "issue_code": decision.issue_code,
            "choices": choices_json,
            "details": decision.details,
            "trace_id": f"fixit-{now_epoch}",
        }

    def run_llm_health(self, *, trigger: str = "manual") -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        try:
            summary = self._health_monitor.run_once(self.registry_document)
            self._router.set_external_health_state(self._health_monitor.state)
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor="system" if trigger == "scheduler" else "user",
                action="llm.health.run",
                params={"trigger": trigger, "modified_ids": []},
                decision="allow",
                reason="probe_failed",
                dry_run=False,
                outcome="failed",
                error_kind=exc.__class__.__name__,
                duration_ms=duration_ms,
            )
            return False, {"ok": False, "error": "health_probe_failed"}

        probed = summary.get("probed") if isinstance(summary.get("probed"), list) else []
        modified_ids = sorted(
            {
                f"model:{str(item.get('model_id') or '').strip()}"
                for item in probed
                if isinstance(item, dict) and str(item.get("model_id") or "").strip()
            }
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor="system" if trigger == "scheduler" else "user",
            action="llm.health.run",
            params={"trigger": trigger, "counts": summary.get("counts"), "modified_ids": modified_ids},
            decision="allow",
            reason="probe_completed",
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        self._log_request(
            "/llm/health/run",
            True,
            {
                "trigger": trigger,
                "counts": summary.get("counts"),
                "probed": len(summary.get("probed") or []),
            },
        )
        return True, {"ok": True, "health": summary, "trigger": trigger}

    def model_scout_sources(self) -> dict[str, Any]:
        if hasattr(self.model_scout, "sources"):
            payload = getattr(self.model_scout, "sources")()
            return {"ok": True, "sources": payload}
        return {"ok": True, "sources": {"ollama": False, "openrouter": False, "huggingface": False}}

    def llm_autoconfig_plan(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        disable_auth_failed = bool(payload.get("disable_auth_failed_providers", True))
        plan = build_autoconfig_plan(
            self.registry_document,
            self._health_monitor.summary(self.registry_document),
            secret_lookup=self.secret_store.get_secret,
            disable_auth_failed_providers=disable_auth_failed,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        modified_ids = self._plan_modified_ids(plan)
        self.audit_log.append(
            actor=str(payload.get("actor") or "user"),
            action="llm.autoconfig.plan",
            params={"impact": plan.get("impact"), "modified_ids": modified_ids},
            decision="allow",
            reason="planned",
            dry_run=True,
            outcome="planned",
            error_kind=None,
            duration_ms=duration_ms,
        )
        return True, {"ok": True, "plan": plan, "modified_ids": modified_ids}

    def llm_autoconfig_apply(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or "user")
        confirm = bool(payload.get("confirm", False))
        trigger_value = "scheduler" if actor == "scheduler" else "manual"
        raw_plan = build_autoconfig_plan(
            self.registry_document,
            self._health_monitor.summary(self.registry_document),
            secret_lookup=self.secret_store.get_secret,
            disable_auth_failed_providers=bool(payload.get("disable_auth_failed_providers", True)),
        )
        plan, safe_mode_blocked = self._apply_safe_mode_to_plan(action="llm.autoconfig.apply", plan=raw_plan)
        modified_ids = self._plan_modified_ids(plan)
        decision = self._modelops_permission_decision(
            "llm.autoconfig.apply",
            params={
                "default_provider": (plan.get("proposed_defaults") or {}).get("default_provider"),
                "default_model": (plan.get("proposed_defaults") or {}).get("default_model"),
            },
            estimated_download_bytes=0,
            estimated_cost=None,
            risk_level="low",
            dry_run=False,
        )
        if not bool(decision["allow"]):
            duration_ms = int((time.monotonic() - start) * 1000)
            reason = str(decision.get("reason") or "policy_deny")
            self.audit_log.append(
                actor=actor,
                action="llm.autoconfig.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                },
                decision="deny",
                reason=reason,
                dry_run=False,
                outcome="blocked",
                error_kind="policy_deny",
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.autoconfig.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason=reason,
                trigger=trigger_value,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {"ok": False, "error": reason, "plan": plan, "modified_ids": modified_ids}
        if bool(decision["requires_confirmation"]) and not confirm:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.autoconfig.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                },
                decision="deny",
                reason="confirmation_required",
                dry_run=False,
                outcome="blocked",
                error_kind="confirmation_required",
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.autoconfig.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason="confirmation_required",
                trigger=trigger_value,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {"ok": False, "error": "confirmation_required", "plan": plan, "modified_ids": modified_ids}

        if not any(isinstance(item, dict) for item in (plan.get("changes") or [])):
            duration_ms = int((time.monotonic() - start) * 1000)
            noop_reason = "safe_mode_blocked" if safe_mode_blocked else "no_changes"
            self.audit_log.append(
                actor=actor,
                action="llm.autoconfig.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "safe_mode_blocked": safe_mode_blocked,
                },
                decision="allow",
                reason=noop_reason,
                dry_run=False,
                outcome="noop",
                error_kind=None,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.autoconfig.apply",
                actor=actor,
                decision="allow",
                outcome="noop",
                reason=noop_reason,
                trigger=trigger_value,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return True, {
                "ok": True,
                "applied": False,
                "plan": plan,
                "defaults": self.get_defaults(),
                "modified_ids": modified_ids,
                "safe_mode_blocked": bool(safe_mode_blocked),
                "safe_mode_blocked_reason": (safe_mode_blocked[0] if safe_mode_blocked else None),
                "safe_mode_blocked_changes": safe_mode_blocked,
            }

        saved, txn_meta = self._persist_registry_document_transactional(
            lambda current: apply_autoconfig_plan(current, plan)
        )
        if not saved:
            error = txn_meta
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.autoconfig.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": error.get("snapshot_id"),
                    "resulting_registry_hash": None,
                },
                decision="allow",
                reason=str(error.get("error") or "registry_write_failed"),
                dry_run=False,
                outcome="failed",
                error_kind=str(error.get("error") or "registry_write_failed"),
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.autoconfig.apply",
                actor=actor,
                decision="allow",
                outcome="failed",
                reason=str(error.get("error") or "registry_write_failed"),
                trigger=trigger_value,
                snapshot_id=str(error.get("snapshot_id") or "") or None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {**error, "modified_ids": modified_ids}

        duration_ms = int((time.monotonic() - start) * 1000)
        snapshot_id = str(txn_meta.get("snapshot_id") or "") or None
        snapshot_id_after = str(txn_meta.get("snapshot_id_after") or "") or None
        resulting_registry_hash = str(txn_meta.get("resulting_registry_hash") or "") or None
        success_reasons = self._plan_reasons(plan)
        success_reason = success_reasons[0] if success_reasons else "allowed"
        self.audit_log.append(
            actor=actor,
            action="llm.autoconfig.apply",
            params={
                "impact": plan.get("impact"),
                "modified_ids": modified_ids,
                "changed_ids": modified_ids,
                "snapshot_id": snapshot_id,
                "snapshot_id_after": snapshot_id_after,
                "resulting_registry_hash": resulting_registry_hash,
            },
            decision="allow",
            reason=success_reason,
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        self._record_action_ledger(
            action="llm.autoconfig.apply",
            actor=actor,
            decision="allow",
            outcome="success",
            reason=success_reason,
            trigger=trigger_value,
            snapshot_id=snapshot_id,
            snapshot_id_after=snapshot_id_after,
            resulting_registry_hash=resulting_registry_hash,
            changed_ids=modified_ids,
        )
        return True, {
            "ok": True,
            "applied": True,
            "plan": plan,
            "defaults": self.get_defaults(),
            "modified_ids": modified_ids,
            "snapshot_id": snapshot_id,
            "snapshot_id_after": snapshot_id_after,
            "resulting_registry_hash": resulting_registry_hash,
            "safe_mode_blocked": bool(safe_mode_blocked),
            "safe_mode_blocked_reason": (safe_mode_blocked[0] if safe_mode_blocked else None),
            "safe_mode_blocked_changes": safe_mode_blocked,
        }

    def llm_hygiene_plan(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        unavailable_days = int(payload.get("unavailable_days") or self.config.llm_hygiene_unavailable_days)
        remove_empty_disabled = bool(
            payload.get(
                "remove_empty_disabled_providers",
                self.config.llm_hygiene_remove_empty_disabled_providers,
            )
        )
        disable_repeatedly_failing = bool(
            payload.get(
                "disable_repeatedly_failing_providers",
                self.config.llm_hygiene_disable_repeatedly_failing_providers,
            )
        )
        provider_failure_streak = int(
            payload.get("provider_failure_streak") or self.config.llm_hygiene_provider_failure_streak
        )
        provider_inventory = payload.get("provider_inventory") if isinstance(payload.get("provider_inventory"), dict) else None
        plan = build_hygiene_plan(
            self.registry_document,
            self._health_monitor.summary(self.registry_document),
            unavailable_days=max(1, unavailable_days),
            remove_empty_disabled_providers=remove_empty_disabled,
            provider_inventory=provider_inventory,
            disable_repeatedly_failing_providers=disable_repeatedly_failing,
            provider_failure_streak=max(1, provider_failure_streak),
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        modified_ids = self._plan_modified_ids(plan)
        self.audit_log.append(
            actor=str(payload.get("actor") or "user"),
            action="llm.hygiene.plan",
            params={"impact": plan.get("impact"), "modified_ids": modified_ids},
            decision="allow",
            reason="planned",
            dry_run=True,
            outcome="planned",
            error_kind=None,
            duration_ms=duration_ms,
        )
        return True, {"ok": True, "plan": plan, "modified_ids": modified_ids}

    def llm_hygiene_apply(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or "user")
        confirm = bool(payload.get("confirm", False))
        trigger_value = "scheduler" if actor == "scheduler" else "manual"
        unavailable_days = int(payload.get("unavailable_days") or self.config.llm_hygiene_unavailable_days)
        remove_empty_disabled = bool(
            payload.get(
                "remove_empty_disabled_providers",
                self.config.llm_hygiene_remove_empty_disabled_providers,
            )
        )
        disable_repeatedly_failing = bool(
            payload.get(
                "disable_repeatedly_failing_providers",
                self.config.llm_hygiene_disable_repeatedly_failing_providers,
            )
        )
        provider_failure_streak = int(
            payload.get("provider_failure_streak") or self.config.llm_hygiene_provider_failure_streak
        )
        provider_inventory = payload.get("provider_inventory") if isinstance(payload.get("provider_inventory"), dict) else None
        raw_plan = build_hygiene_plan(
            self.registry_document,
            self._health_monitor.summary(self.registry_document),
            unavailable_days=max(1, unavailable_days),
            remove_empty_disabled_providers=remove_empty_disabled,
            provider_inventory=provider_inventory,
            disable_repeatedly_failing_providers=disable_repeatedly_failing,
            provider_failure_streak=max(1, provider_failure_streak),
        )
        plan, safe_mode_blocked = self._apply_safe_mode_to_plan(action="llm.hygiene.apply", plan=raw_plan)
        modified_ids = self._plan_modified_ids(plan)
        decision = self._modelops_permission_decision(
            "llm.hygiene.apply",
            params={},
            estimated_download_bytes=0,
            estimated_cost=None,
            risk_level="low",
            dry_run=False,
        )
        if not bool(decision["allow"]):
            duration_ms = int((time.monotonic() - start) * 1000)
            reason = str(decision.get("reason") or "policy_deny")
            self.audit_log.append(
                actor=actor,
                action="llm.hygiene.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                },
                decision="deny",
                reason=reason,
                dry_run=False,
                outcome="blocked",
                error_kind="policy_deny",
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.hygiene.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason=reason,
                trigger=trigger_value,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {"ok": False, "error": reason, "plan": plan, "modified_ids": modified_ids}
        if bool(decision["requires_confirmation"]) and not confirm:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.hygiene.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                },
                decision="deny",
                reason="confirmation_required",
                dry_run=False,
                outcome="blocked",
                error_kind="confirmation_required",
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.hygiene.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason="confirmation_required",
                trigger=trigger_value,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {"ok": False, "error": "confirmation_required", "plan": plan, "modified_ids": modified_ids}

        if not any(isinstance(item, dict) for item in (plan.get("changes") or [])):
            duration_ms = int((time.monotonic() - start) * 1000)
            noop_reason = "safe_mode_blocked" if safe_mode_blocked else "no_changes"
            self.audit_log.append(
                actor=actor,
                action="llm.hygiene.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "safe_mode_blocked": safe_mode_blocked,
                },
                decision="allow",
                reason=noop_reason,
                dry_run=False,
                outcome="noop",
                error_kind=None,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.hygiene.apply",
                actor=actor,
                decision="allow",
                outcome="noop",
                reason=noop_reason,
                trigger=trigger_value,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return True, {
                "ok": True,
                "applied": False,
                "plan": plan,
                "modified_ids": modified_ids,
                "safe_mode_blocked": bool(safe_mode_blocked),
                "safe_mode_blocked_reason": (safe_mode_blocked[0] if safe_mode_blocked else None),
                "safe_mode_blocked_changes": safe_mode_blocked,
            }

        saved, txn_meta = self._persist_registry_document_transactional(
            lambda current: apply_hygiene_plan(current, plan)
        )
        if not saved:
            error = txn_meta
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.hygiene.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": error.get("snapshot_id"),
                    "resulting_registry_hash": None,
                },
                decision="allow",
                reason=str(error.get("error") or "registry_write_failed"),
                dry_run=False,
                outcome="failed",
                error_kind=str(error.get("error") or "registry_write_failed"),
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.hygiene.apply",
                actor=actor,
                decision="allow",
                outcome="failed",
                reason=str(error.get("error") or "registry_write_failed"),
                trigger=trigger_value,
                snapshot_id=str(error.get("snapshot_id") or "") or None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {**error, "modified_ids": modified_ids}

        duration_ms = int((time.monotonic() - start) * 1000)
        snapshot_id = str(txn_meta.get("snapshot_id") or "") or None
        snapshot_id_after = str(txn_meta.get("snapshot_id_after") or "") or None
        resulting_registry_hash = str(txn_meta.get("resulting_registry_hash") or "") or None
        success_reasons = self._plan_reasons(plan)
        success_reason = success_reasons[0] if success_reasons else "allowed"
        self.audit_log.append(
            actor=actor,
            action="llm.hygiene.apply",
            params={
                "impact": plan.get("impact"),
                "modified_ids": modified_ids,
                "changed_ids": modified_ids,
                "snapshot_id": snapshot_id,
                "snapshot_id_after": snapshot_id_after,
                "resulting_registry_hash": resulting_registry_hash,
            },
            decision="allow",
            reason=success_reason,
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        self._record_action_ledger(
            action="llm.hygiene.apply",
            actor=actor,
            decision="allow",
            outcome="success",
            reason=success_reason,
            trigger=trigger_value,
            snapshot_id=snapshot_id,
            snapshot_id_after=snapshot_id_after,
            resulting_registry_hash=resulting_registry_hash,
            changed_ids=modified_ids,
        )
        return True, {
            "ok": True,
            "applied": True,
            "plan": plan,
            "modified_ids": modified_ids,
            "snapshot_id": snapshot_id,
            "snapshot_id_after": snapshot_id_after,
            "resulting_registry_hash": resulting_registry_hash,
            "safe_mode_blocked": bool(safe_mode_blocked),
            "safe_mode_blocked_reason": (safe_mode_blocked[0] if safe_mode_blocked else None),
            "safe_mode_blocked_changes": safe_mode_blocked,
        }

    def llm_cleanup_plan(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or "user")
        plan = build_registry_cleanup_plan(
            self.registry_document,
            self._router.usage_stats_snapshot(),
            self._health_monitor.summary(self.registry_document),
            self._catalog_store.snapshot(),
            unused_days=max(1, int(payload.get("unused_days") or self.config.llm_registry_prune_unused_days)),
            disable_failing_provider=bool(
                payload.get(
                    "disable_failing_provider",
                    self.config.llm_registry_prune_disable_failing_provider,
                )
            ),
            provider_failure_streak=max(
                1,
                int(payload.get("provider_failure_streak") or self.config.llm_hygiene_provider_failure_streak),
            ),
            apply_prune=False,
        )
        modified_ids = self._plan_modified_ids(plan)
        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor=actor,
            action="llm.cleanup.plan",
            params={"impact": plan.get("impact"), "modified_ids": modified_ids},
            decision="allow",
            reason="planned",
            dry_run=True,
            outcome="planned",
            error_kind=None,
            duration_ms=duration_ms,
        )
        return True, {"ok": True, "plan": plan, "modified_ids": modified_ids}

    def llm_cleanup_apply(
        self,
        payload: dict[str, Any],
        *,
        trigger: str = "manual",
    ) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or ("system" if trigger == "scheduler" else "user"))
        confirm = bool(payload.get("confirm", False))
        unused_days = max(1, int(payload.get("unused_days") or self.config.llm_registry_prune_unused_days))
        disable_failing_provider = bool(
            payload.get(
                "disable_failing_provider",
                self.config.llm_registry_prune_disable_failing_provider,
            )
        )
        provider_failure_streak = max(
            1,
            int(payload.get("provider_failure_streak") or self.config.llm_hygiene_provider_failure_streak),
        )
        raw_preview_plan = build_registry_cleanup_plan(
            self.registry_document,
            self._router.usage_stats_snapshot(),
            self._health_monitor.summary(self.registry_document),
            self._catalog_store.snapshot(),
            unused_days=unused_days,
            disable_failing_provider=disable_failing_provider,
            provider_failure_streak=provider_failure_streak,
            apply_prune=False,
        )
        preview_plan, _preview_blocked = self._apply_safe_mode_to_plan(action="llm.cleanup.apply", plan=raw_preview_plan)
        modified_ids = self._plan_modified_ids(preview_plan)
        decision = self._modelops_permission_decision(
            "llm.registry.prune",
            params={
                "unused_days": unused_days,
                "disable_failing_provider": disable_failing_provider,
            },
            estimated_download_bytes=0,
            estimated_cost=None,
            risk_level="low",
            dry_run=False,
        )
        scheduler_auto_policy = compute_registry_prune_apply_policy(self)
        scheduler_auto_allow = trigger == "scheduler" and bool(scheduler_auto_policy.get("allow_apply_effective"))
        effective_allow = bool(decision.get("allow")) or bool(scheduler_auto_allow)

        if not effective_allow:
            duration_ms = int((time.monotonic() - start) * 1000)
            reason = str(decision.get("reason") or "policy_deny")
            self.audit_log.append(
                actor=actor,
                action="llm.cleanup.apply",
                params={
                    "impact": preview_plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="deny",
                reason=reason,
                dry_run=False,
                outcome="blocked",
                error_kind=reason,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.cleanup.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason=reason,
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {
                "ok": False,
                "error": reason,
                "plan": preview_plan,
                "modified_ids": modified_ids,
            }

        if bool(decision.get("requires_confirmation")) and not scheduler_auto_allow and not confirm:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.cleanup.apply",
                params={
                    "impact": preview_plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="deny",
                reason="confirmation_required",
                dry_run=False,
                outcome="blocked",
                error_kind="confirmation_required",
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.cleanup.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason="confirmation_required",
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {
                "ok": False,
                "error": "confirmation_required",
                "plan": preview_plan,
                "modified_ids": modified_ids,
            }

        raw_plan = build_registry_cleanup_plan(
            self.registry_document,
            self._router.usage_stats_snapshot(),
            self._health_monitor.summary(self.registry_document),
            self._catalog_store.snapshot(),
            unused_days=unused_days,
            disable_failing_provider=disable_failing_provider,
            provider_failure_streak=provider_failure_streak,
            apply_prune=True,
        )
        plan, safe_mode_blocked = self._apply_safe_mode_to_plan(action="llm.cleanup.apply", plan=raw_plan)
        modified_ids = self._plan_modified_ids(plan)
        if not any(isinstance(item, dict) for item in (plan.get("changes") or [])):
            duration_ms = int((time.monotonic() - start) * 1000)
            noop_reason = "safe_mode_blocked" if safe_mode_blocked else "no_changes"
            self.audit_log.append(
                actor=actor,
                action="llm.cleanup.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "safe_mode_blocked": safe_mode_blocked,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="allow",
                reason=noop_reason,
                dry_run=False,
                outcome="noop",
                error_kind=None,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.cleanup.apply",
                actor=actor,
                decision="allow",
                outcome="noop",
                reason=noop_reason,
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return True, {
                "ok": True,
                "applied": False,
                "plan": plan,
                "modified_ids": modified_ids,
                "safe_mode_blocked": bool(safe_mode_blocked),
                "safe_mode_blocked_reason": (safe_mode_blocked[0] if safe_mode_blocked else None),
                "safe_mode_blocked_changes": safe_mode_blocked,
            }

        saved, txn_meta = self._persist_registry_document_transactional(
            lambda current: apply_registry_cleanup_plan(current, plan)
        )
        if not saved:
            error = txn_meta
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.cleanup.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": error.get("snapshot_id"),
                    "resulting_registry_hash": None,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="allow",
                reason=str(error.get("error") or "registry_write_failed"),
                dry_run=False,
                outcome="failed",
                error_kind=str(error.get("error") or "registry_write_failed"),
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.cleanup.apply",
                actor=actor,
                decision="allow",
                outcome="failed",
                reason=str(error.get("error") or "registry_write_failed"),
                trigger=trigger,
                snapshot_id=str(error.get("snapshot_id") or "") or None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {**error, "plan": plan, "modified_ids": modified_ids}

        duration_ms = int((time.monotonic() - start) * 1000)
        snapshot_id = str(txn_meta.get("snapshot_id") or "") or None
        snapshot_id_after = str(txn_meta.get("snapshot_id_after") or "") or None
        resulting_registry_hash = str(txn_meta.get("resulting_registry_hash") or "") or None
        success_reasons = self._plan_reasons(plan)
        success_reason = success_reasons[0] if success_reasons else "allowed"
        self.audit_log.append(
            actor=actor,
            action="llm.cleanup.apply",
            params={
                "impact": plan.get("impact"),
                "modified_ids": modified_ids,
                "changed_ids": modified_ids,
                "snapshot_id": snapshot_id,
                "snapshot_id_after": snapshot_id_after,
                "resulting_registry_hash": resulting_registry_hash,
                "trigger": trigger,
                "scheduler_auto_policy": scheduler_auto_policy,
            },
            decision="allow",
            reason=success_reason,
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        self._record_action_ledger(
            action="llm.cleanup.apply",
            actor=actor,
            decision="allow",
            outcome="success",
            reason=success_reason,
            trigger=trigger,
            snapshot_id=snapshot_id,
            snapshot_id_after=snapshot_id_after,
            resulting_registry_hash=resulting_registry_hash,
            changed_ids=modified_ids,
        )
        return True, {
            "ok": True,
            "applied": True,
            "plan": plan,
            "modified_ids": modified_ids,
            "snapshot_id": snapshot_id,
            "snapshot_id_after": snapshot_id_after,
            "resulting_registry_hash": resulting_registry_hash,
            "safe_mode_blocked": bool(safe_mode_blocked),
            "safe_mode_blocked_reason": (safe_mode_blocked[0] if safe_mode_blocked else None),
            "safe_mode_blocked_changes": safe_mode_blocked,
        }

    def llm_self_heal_plan(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or "user")
        health_summary = self._health_monitor.summary(self.registry_document)
        router_snapshot = self._router.doctor_snapshot()
        plan = build_self_heal_plan(
            self.registry_document,
            health_summary,
            router_snapshot=router_snapshot,
        )
        modified_ids = self._plan_modified_ids(plan)
        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor=actor,
            action="llm.self_heal.plan",
            params={
                "impact": plan.get("impact"),
                "drift": plan.get("drift"),
                "modified_ids": modified_ids,
            },
            decision="allow",
            reason="planned",
            dry_run=True,
            outcome="planned",
            error_kind=None,
            duration_ms=duration_ms,
        )
        return True, {"ok": True, "plan": plan, "drift": plan.get("drift"), "modified_ids": modified_ids}

    def llm_self_heal_apply(
        self,
        payload: dict[str, Any],
        *,
        trigger: str = "manual",
    ) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or ("system" if trigger == "scheduler" else "user"))
        confirm = bool(payload.get("confirm", False))
        health_summary = self._health_monitor.summary(self.registry_document)
        router_snapshot = self._router.doctor_snapshot()
        raw_plan = build_self_heal_plan(
            self.registry_document,
            health_summary,
            router_snapshot=router_snapshot,
        )
        plan, safe_mode_blocked = self._apply_safe_mode_to_plan(action="llm.self_heal.apply", plan=raw_plan)
        modified_ids = self._plan_modified_ids(plan)
        decision = self._modelops_permission_decision(
            "llm.self_heal.apply",
            params={
                "default_provider": (plan.get("proposed_defaults") or {}).get("default_provider"),
                "default_model": (plan.get("proposed_defaults") or {}).get("default_model"),
            },
            estimated_download_bytes=0,
            estimated_cost=None,
            risk_level="low",
            dry_run=False,
        )
        scheduler_auto_policy = compute_self_heal_apply_policy(self)
        scheduler_auto_allow = (
            trigger == "scheduler"
            and bool(scheduler_auto_policy.get("allow_apply_effective"))
        )
        effective_allow = bool(decision.get("allow")) or bool(scheduler_auto_allow)

        if not effective_allow:
            duration_ms = int((time.monotonic() - start) * 1000)
            reason = str(decision.get("reason") or "policy_deny")
            self.audit_log.append(
                actor=actor,
                action="llm.self_heal.apply",
                params={
                    "impact": plan.get("impact"),
                    "drift": plan.get("drift"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="deny",
                reason=reason,
                dry_run=False,
                outcome="blocked",
                error_kind=reason,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.self_heal.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason=reason,
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {
                "ok": False,
                "error": reason,
                "plan": plan,
                "drift": plan.get("drift"),
                "modified_ids": modified_ids,
            }
        if bool(decision.get("requires_confirmation")) and not scheduler_auto_allow and not confirm:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.self_heal.apply",
                params={
                    "impact": plan.get("impact"),
                    "drift": plan.get("drift"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="deny",
                reason="confirmation_required",
                dry_run=False,
                outcome="blocked",
                error_kind="confirmation_required",
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.self_heal.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason="confirmation_required",
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {
                "ok": False,
                "error": "confirmation_required",
                "plan": plan,
                "drift": plan.get("drift"),
                "modified_ids": modified_ids,
            }

        if not any(isinstance(item, dict) for item in (plan.get("changes") or [])):
            duration_ms = int((time.monotonic() - start) * 1000)
            noop_reason = "safe_mode_blocked" if safe_mode_blocked else "no_changes"
            self.audit_log.append(
                actor=actor,
                action="llm.self_heal.apply",
                params={
                    "impact": plan.get("impact"),
                    "drift": plan.get("drift"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "safe_mode_blocked": safe_mode_blocked,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="allow",
                reason=noop_reason,
                dry_run=False,
                outcome="noop",
                error_kind=None,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.self_heal.apply",
                actor=actor,
                decision="allow",
                outcome="noop",
                reason=noop_reason,
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return True, {
                "ok": True,
                "applied": False,
                "plan": plan,
                "drift": plan.get("drift"),
                "defaults": self.get_defaults(),
                "modified_ids": modified_ids,
                "safe_mode_blocked": bool(safe_mode_blocked),
                "safe_mode_blocked_reason": (safe_mode_blocked[0] if safe_mode_blocked else None),
                "safe_mode_blocked_changes": safe_mode_blocked,
            }

        saved, txn_meta = self._persist_registry_document_transactional(
            lambda current: apply_self_heal_plan(current, plan)
        )
        if not saved:
            error = txn_meta
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.self_heal.apply",
                params={
                    "impact": plan.get("impact"),
                    "drift": plan.get("drift"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": error.get("snapshot_id"),
                    "resulting_registry_hash": None,
                    "trigger": trigger,
                    "scheduler_auto_policy": scheduler_auto_policy,
                },
                decision="allow",
                reason=str(error.get("error") or "registry_write_failed"),
                dry_run=False,
                outcome="failed",
                error_kind=str(error.get("error") or "registry_write_failed"),
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.self_heal.apply",
                actor=actor,
                decision="allow",
                outcome="failed",
                reason=str(error.get("error") or "registry_write_failed"),
                trigger=trigger,
                snapshot_id=str(error.get("snapshot_id") or "") or None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {
                **error,
                "plan": plan,
                "drift": plan.get("drift"),
                "modified_ids": modified_ids,
            }

        duration_ms = int((time.monotonic() - start) * 1000)
        snapshot_id = str(txn_meta.get("snapshot_id") or "") or None
        snapshot_id_after = str(txn_meta.get("snapshot_id_after") or "") or None
        resulting_registry_hash = str(txn_meta.get("resulting_registry_hash") or "") or None
        success_reasons = self._plan_reasons(plan)
        success_reason = success_reasons[0] if success_reasons else "allowed"
        self.audit_log.append(
            actor=actor,
            action="llm.self_heal.apply",
            params={
                "impact": plan.get("impact"),
                "drift": plan.get("drift"),
                "modified_ids": modified_ids,
                "changed_ids": modified_ids,
                "snapshot_id": snapshot_id,
                "snapshot_id_after": snapshot_id_after,
                "resulting_registry_hash": resulting_registry_hash,
                "trigger": trigger,
                "scheduler_auto_policy": scheduler_auto_policy,
            },
            decision="allow",
            reason=success_reason,
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        self._record_action_ledger(
            action="llm.self_heal.apply",
            actor=actor,
            decision="allow",
            outcome="success",
            reason=success_reason,
            trigger=trigger,
            snapshot_id=snapshot_id,
            snapshot_id_after=snapshot_id_after,
            resulting_registry_hash=resulting_registry_hash,
            changed_ids=modified_ids,
        )
        return True, {
            "ok": True,
            "applied": True,
            "plan": plan,
            "drift": plan.get("drift"),
            "defaults": self.get_defaults(),
            "modified_ids": modified_ids,
            "snapshot_id": snapshot_id,
            "snapshot_id_after": snapshot_id_after,
            "resulting_registry_hash": resulting_registry_hash,
            "safe_mode_blocked": bool(safe_mode_blocked),
            "safe_mode_blocked_reason": (safe_mode_blocked[0] if safe_mode_blocked else None),
            "safe_mode_blocked_changes": safe_mode_blocked,
        }

    def model_scout_status(self) -> dict[str, Any]:
        allowed, denied = self._model_scout_pack_gate(iface="model_scout.status")
        if not allowed:
            return dict(denied or {"ok": False, "error": "pack_permission_denied"})
        status = self.model_scout.status()
        return {
            "ok": True,
            "enabled": bool(self.config.model_scout_enabled),
            "status": status,
        }

    def model_scout_suggestions(self) -> dict[str, Any]:
        allowed, denied = self._model_scout_pack_gate(iface="model_scout.suggestions")
        if not allowed:
            return dict(denied or {"ok": False, "error": "pack_permission_denied"})
        suggestions = self.model_scout.list_suggestions(limit=200)
        return {
            "ok": True,
            "suggestions": suggestions,
        }

    @staticmethod
    def _empty_model_scout_notify_state() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "initialized": False,
            "last_default_provider": None,
            "last_default_model": None,
            "last_ollama_models": [],
            "last_capabilities_signature": None,
            "last_run_ts": None,
        }

    def _load_model_scout_notify_state(self) -> dict[str, Any]:
        path = self._model_scout_notify_state_path
        default_state = self._empty_model_scout_notify_state()
        if not path.is_file():
            return default_state
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default_state
        if not isinstance(raw, dict):
            return default_state
        state = dict(default_state)
        state["initialized"] = bool(raw.get("initialized", False))
        state["last_default_provider"] = str(raw.get("last_default_provider") or "").strip() or None
        state["last_default_model"] = str(raw.get("last_default_model") or "").strip() or None
        rows = raw.get("last_ollama_models") if isinstance(raw.get("last_ollama_models"), list) else []
        state["last_ollama_models"] = sorted(
            {
                str(item).strip()
                for item in rows
                if str(item).strip()
            }
        )
        state["last_capabilities_signature"] = str(raw.get("last_capabilities_signature") or "").strip() or None
        try:
            state["last_run_ts"] = int(raw.get("last_run_ts") or 0) or None
        except (TypeError, ValueError):
            state["last_run_ts"] = None
        return state

    def _save_model_scout_notify_state(self, state: dict[str, Any]) -> None:
        path = self._model_scout_notify_state_path
        normalized = self._empty_model_scout_notify_state()
        normalized["initialized"] = bool(state.get("initialized", False))
        normalized["last_default_provider"] = str(state.get("last_default_provider") or "").strip() or None
        normalized["last_default_model"] = str(state.get("last_default_model") or "").strip() or None
        raw_models = state.get("last_ollama_models") if isinstance(state.get("last_ollama_models"), list) else []
        normalized["last_ollama_models"] = sorted(
            {
                str(item).strip()
                for item in raw_models
                if str(item).strip()
            }
        )
        normalized["last_capabilities_signature"] = str(state.get("last_capabilities_signature") or "").strip() or None
        try:
            normalized["last_run_ts"] = int(state.get("last_run_ts") or 0) or None
        except (TypeError, ValueError):
            normalized["last_run_ts"] = None
        payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _scout_defaults_from_document(document: dict[str, Any]) -> tuple[str | None, str | None]:
        defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
        provider = str(defaults.get("default_provider") or "").strip().lower() or None
        model = str(defaults.get("default_model") or "").strip() or None
        return provider, model

    @staticmethod
    def _scout_ollama_model_ids(document: dict[str, Any]) -> list[str]:
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        rows: list[str] = []
        for model_id, payload in sorted(models.items()):
            if not isinstance(payload, dict):
                continue
            if str(payload.get("provider") or "").strip().lower() != "ollama":
                continue
            if not bool(payload.get("available", True)):
                continue
            rows.append(str(model_id).strip())
        return sorted({item for item in rows if item})

    @staticmethod
    def _scout_capabilities_signature(document: dict[str, Any]) -> str:
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        provider_rows = []
        for provider_id, payload in sorted(providers.items()):
            if not isinstance(payload, dict):
                continue
            provider_rows.append(
                {
                    "id": str(provider_id).strip().lower(),
                    "enabled": bool(payload.get("enabled", True)),
                    "available": bool(payload.get("available", True)),
                }
            )
        model_rows = []
        for model_id, payload in sorted(models.items()):
            if not isinstance(payload, dict):
                continue
            capabilities = payload.get("capabilities") if isinstance(payload.get("capabilities"), list) else []
            model_rows.append(
                {
                    "id": str(model_id).strip(),
                    "provider": str(payload.get("provider") or "").strip().lower(),
                    "enabled": bool(payload.get("enabled", True)),
                    "available": bool(payload.get("available", True)),
                    "routable": bool(payload.get("routable", False)),
                    "capabilities": sorted(
                        {
                            str(item).strip().lower()
                            for item in capabilities
                            if str(item).strip()
                        }
                    ),
                }
            )
        canonical = json.dumps(
            {"providers": provider_rows, "models": model_rows},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def _scout_baseline_score_for_provider(self, provider_id: str) -> float | None:
        store = getattr(self.model_scout, "store", None)
        if store is None or not hasattr(store, "latest_baseline"):
            return None
        try:
            baseline_payload = store.latest_baseline()
        except Exception:
            baseline_payload = None
        snapshot = (
            baseline_payload.get("snapshot")
            if isinstance(baseline_payload, dict) and isinstance(baseline_payload.get("snapshot"), dict)
            else {}
        )
        local = snapshot.get("local") if isinstance(snapshot.get("local"), dict) else {}
        remote = snapshot.get("remote") if isinstance(snapshot.get("remote"), dict) else {}
        provider_key = str(provider_id or "").strip().lower()
        if provider_key == "ollama":
            try:
                return float(local.get("score")) if local.get("score") is not None else None
            except (TypeError, ValueError):
                return None
        row = remote.get(provider_key) if isinstance(remote.get(provider_key), dict) else {}
        try:
            return float(row.get("score")) if row.get("score") is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _top_model_scout_candidate(result: dict[str, Any]) -> dict[str, Any] | None:
        rows = result.get("new_suggestions") if isinstance(result.get("new_suggestions"), list) else []
        candidates = [row for row in rows if isinstance(row, dict)]
        candidates.sort(
            key=lambda item: (
                -float(item.get("score") or 0.0),
                str(item.get("id") or ""),
            )
        )
        for row in candidates:
            model_id = str(row.get("model_id") or "").strip()
            provider_id = str(row.get("provider_id") or "").strip().lower()
            if model_id and provider_id:
                return row
        return None

    @staticmethod
    def _model_scout_event_priority(reason: str) -> int:
        priority = {
            "better_default_candidate": 0,
            "ollama_model_installed": 1,
            "registry_capabilities_changed": 2,
        }
        return int(priority.get(str(reason or ""), 99))

    def _model_scout_change_events(
        self,
        *,
        result: dict[str, Any],
        state_before: dict[str, Any],
    ) -> list[dict[str, Any]]:
        default_provider, default_model = self._scout_defaults_from_document(self.registry_document)
        events: list[dict[str, Any]] = []
        top_candidate = self._top_model_scout_candidate(result)
        candidate_model = ""
        candidate_provider = ""
        if isinstance(top_candidate, dict):
            candidate_model = str(top_candidate.get("model_id") or "").strip()
            candidate_provider = str(top_candidate.get("provider_id") or "").strip().lower()
        if candidate_model and candidate_model != str(default_model or ""):
            event_payload: dict[str, Any] = {
                "from_model": str(default_model or ""),
                "to_model": candidate_model,
                "reason": "better_default_candidate",
                "provider": candidate_provider or str(default_provider or ""),
            }
            baseline_score = self._scout_baseline_score_for_provider(candidate_provider or str(default_provider or ""))
            try:
                candidate_score = float((top_candidate or {}).get("score")) if top_candidate is not None else None
            except (TypeError, ValueError):
                candidate_score = None
            if baseline_score is not None and candidate_score is not None:
                event_payload["score_delta"] = round(float(candidate_score) - float(baseline_score), 2)
            events.append(event_payload)

        initialized = bool(state_before.get("initialized", False))
        current_ollama = self._scout_ollama_model_ids(self.registry_document)
        previous_ollama = sorted(
            {
                str(item).strip()
                for item in (
                    state_before.get("last_ollama_models")
                    if isinstance(state_before.get("last_ollama_models"), list)
                    else []
                )
                if str(item).strip()
            }
        )
        if initialized:
            added_ollama = sorted(set(current_ollama) - set(previous_ollama))
            if added_ollama:
                added_model = str(added_ollama[0] or "").strip()
                if added_model:
                    events.append(
                        {
                            "from_model": str(default_model or ""),
                            "to_model": added_model,
                            "reason": "ollama_model_installed",
                            "provider": "ollama",
                        }
                    )
            previous_sig = str(state_before.get("last_capabilities_signature") or "").strip()
            current_sig = self._scout_capabilities_signature(self.registry_document)
            if previous_sig and current_sig and previous_sig != current_sig:
                recommended_model = candidate_model or str(default_model or "")
                recommended_provider = candidate_provider or str(default_provider or "")
                events.append(
                    {
                        "from_model": str(default_model or ""),
                        "to_model": recommended_model,
                        "reason": "registry_capabilities_changed",
                        "provider": recommended_provider,
                    }
                )

        deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in events:
            key = (
                str(row.get("reason") or ""),
                str(row.get("provider") or ""),
                str(row.get("from_model") or ""),
                str(row.get("to_model") or ""),
            )
            deduped[key] = row
        ordered = sorted(
            deduped.values(),
            key=lambda row: (
                self._model_scout_event_priority(str(row.get("reason") or "")),
                str(row.get("provider") or ""),
                str(row.get("to_model") or ""),
                str(row.get("from_model") or ""),
            ),
        )
        return ordered

    @staticmethod
    def _model_scout_reason_human(reason: str) -> str:
        normalized = str(reason or "").strip().lower()
        if normalized == "better_default_candidate":
            return "better default candidate"
        if normalized == "ollama_model_installed":
            return "new local model detected"
        if normalized == "registry_capabilities_changed":
            return "registry capabilities changed"
        return "scout update"

    def _format_model_scout_change_message(self, event: dict[str, Any]) -> str:
        from_model = str(event.get("from_model") or "").strip() or "(none)"
        to_model = str(event.get("to_model") or "").strip() or from_model
        provider = str(event.get("provider") or "").strip().lower() or "ollama"
        reason = self._model_scout_reason_human(str(event.get("reason") or ""))
        apply_payload = json.dumps(
            {"default_model": to_model, "default_provider": provider},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        apply_cmd = (
            "curl -sS -X PUT http://127.0.0.1:8765/defaults "
            "-H 'content-type: application/json' "
            f"-d '{apply_payload}'"
        )
        lines = [
            f"Model Scout update: {reason}.",
            f"Current default: {from_model}",
            f"Recommended default: {to_model}",
            f"Apply: {apply_cmd}",
        ]
        return "\n".join(lines)

    def _audit_model_scout_changed(self, *, actor: str, event: dict[str, Any], trigger: str) -> None:
        try:
            self.audit_log.append(
                actor=actor,
                action="llm.model_scout.changed",
                params={
                    "from_model": str(event.get("from_model") or ""),
                    "to_model": str(event.get("to_model") or ""),
                    "reason": str(event.get("reason") or ""),
                    "provider": str(event.get("provider") or ""),
                    "score_delta": event.get("score_delta"),
                },
                decision="allow",
                reason=str(event.get("reason") or "changed"),
                dry_run=False,
                outcome="detected",
                error_kind=None,
                duration_ms=0,
            )
        except Exception:
            pass

    def _notify_model_scout_change(
        self,
        *,
        actor: str,
        trigger: str,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        telegram = self.telegram_status()
        state = str(telegram.get("state") or "").strip().lower()
        if state != "running":
            self.audit_log.append(
                actor=actor,
                action="llm.model_scout.notify",
                params={
                    "trigger": trigger,
                    "reason": str(event.get("reason") or ""),
                    "provider": str(event.get("provider") or ""),
                },
                decision="allow",
                reason="telegram_not_running",
                dry_run=False,
                outcome="skipped",
                error_kind="telegram_not_running",
                duration_ms=0,
            )
            return {"emitted": False, "reason": "telegram_not_running"}

        token, chat_id = self._resolve_telegram_target()
        if not token or not chat_id:
            self.audit_log.append(
                actor=actor,
                action="llm.model_scout.notify",
                params={
                    "trigger": trigger,
                    "reason": str(event.get("reason") or ""),
                    "provider": str(event.get("provider") or ""),
                },
                decision="allow",
                reason="telegram_not_configured_or_no_chat",
                dry_run=False,
                outcome="skipped",
                error_kind="telegram_not_configured_or_no_chat",
                duration_ms=0,
            )
            return {"emitted": False, "reason": "telegram_not_configured_or_no_chat"}

        message = self._format_model_scout_change_message(event)
        start = time.monotonic()
        try:
            self._send_telegram_message(token, chat_id, message)
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.model_scout.notify",
                params={
                    "trigger": trigger,
                    "reason": str(event.get("reason") or ""),
                    "provider": str(event.get("provider") or ""),
                    "chat_id_redacted": self._redact_telegram_chat_id(chat_id),
                },
                decision="allow",
                reason=f"telegram_send_failed:{exc.__class__.__name__}",
                dry_run=False,
                outcome="failed",
                error_kind=f"telegram_send_failed:{exc.__class__.__name__}",
                duration_ms=duration_ms,
            )
            return {"emitted": False, "reason": f"telegram_send_failed:{exc.__class__.__name__}"}

        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor=actor,
            action="llm.model_scout.notify",
            params={
                "trigger": trigger,
                "reason": str(event.get("reason") or ""),
                "provider": str(event.get("provider") or ""),
                "chat_id_redacted": self._redact_telegram_chat_id(chat_id),
            },
            decision="allow",
            reason="sent",
            dry_run=False,
            outcome="sent",
            error_kind=None,
            duration_ms=duration_ms,
        )
        return {"emitted": True, "reason": "sent"}

    def run_model_scout(self, *, trigger: str = "manual") -> tuple[bool, dict[str, Any]]:
        allowed, denied = self._model_scout_pack_gate(iface="model_scout.run")
        if not allowed:
            body = dict(denied or {"ok": False, "error": "pack_permission_denied"})
            self._log_request("/model_scout/run", False, body)
            return False, body
        actor = "system" if trigger == "scheduler" else "user"
        notify_state_before = self._load_model_scout_notify_state()
        result = self.model_scout.run(
            registry_document=self.registry_document,
            router_snapshot=self._router.doctor_snapshot(),
            usage_stats_snapshot=self._router.usage_stats_snapshot(),
            notify_sender=None,
        )
        change_events = self._model_scout_change_events(
            result=result if isinstance(result, dict) else {},
            state_before=notify_state_before,
        )
        for event in change_events:
            self._audit_model_scout_changed(actor=actor, event=event, trigger=trigger)

        notify_result = {"emitted": False, "reason": "no_change"}
        if change_events:
            notify_result = self._notify_model_scout_change(
                actor=actor,
                trigger=trigger,
                event=change_events[0],
            )

        default_provider, default_model = self._scout_defaults_from_document(self.registry_document)
        notify_state_after = {
            "schema_version": 1,
            "initialized": True,
            "last_default_provider": default_provider,
            "last_default_model": default_model,
            "last_ollama_models": self._scout_ollama_model_ids(self.registry_document),
            "last_capabilities_signature": self._scout_capabilities_signature(self.registry_document),
            "last_run_ts": int(time.time()),
        }
        self._save_model_scout_notify_state(notify_state_after)
        self._log_request(
            "/model_scout/run",
            bool(result.get("ok")),
            {
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
                "suggestions": len(result.get("suggestions") or []),
                "new": len(result.get("new_suggestions") or []),
                "changes": len(change_events),
                "notified": bool(notify_result.get("emitted", False)),
            },
        )
        return bool(result.get("ok")), {
            "ok": bool(result.get("ok")),
            **result,
            "model_scout_change_events": change_events,
            "notification_emitted": bool(notify_result.get("emitted", False)),
            "notification_reason": str(notify_result.get("reason") or "no_change"),
        }

    def dismiss_model_scout_suggestion(self, suggestion_id: str) -> tuple[bool, dict[str, Any]]:
        allowed, denied = self._model_scout_pack_gate(iface="model_scout.dismiss")
        if not allowed:
            return False, dict(denied or {"ok": False, "error": "pack_permission_denied"})
        target = urllib.parse.unquote(str(suggestion_id or "")).strip()
        if not target:
            return False, {"ok": False, "error": "suggestion id is required"}
        if not self.model_scout.dismiss(target):
            return False, {"ok": False, "error": "suggestion not found"}
        return True, {"ok": True, "id": target, "status": "dismissed"}

    def mark_model_scout_installed(self, suggestion_id: str) -> tuple[bool, dict[str, Any]]:
        allowed, denied = self._model_scout_pack_gate(iface="model_scout.mark_installed")
        if not allowed:
            return False, dict(denied or {"ok": False, "error": "pack_permission_denied"})
        target = urllib.parse.unquote(str(suggestion_id or "")).strip()
        if not target:
            return False, {"ok": False, "error": "suggestion id is required"}
        if not self.model_scout.mark_installed(target):
            return False, {"ok": False, "error": "suggestion not found"}
        return True, {"ok": True, "id": target, "status": "installed"}

    @staticmethod
    def _model_watch_enabled_provider_set(registry_document: dict[str, Any]) -> frozenset[str]:
        providers = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
        return frozenset(
            str(provider_id).strip().lower()
            for provider_id, row in sorted(providers.items())
            if isinstance(row, dict) and bool(row.get("enabled", False))
        )

    @staticmethod
    def _model_watch_scoring_candidate(
        *,
        model_id: str,
        row: dict[str, Any],
        provider_row: dict[str, Any],
    ) -> dict[str, Any]:
        pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
        provider_id = str(row.get("provider") or "").strip().lower()
        available = bool(row.get("enabled", False)) and bool(row.get("available", False))
        routable = bool(row.get("routable", available))
        if "routable" in row:
            available = available and bool(row.get("routable", False))
            routable = bool(row.get("routable", False))
        context_tokens = None
        try:
            context_tokens = int(row.get("max_context_tokens")) if row.get("max_context_tokens") is not None else None
        except (TypeError, ValueError):
            context_tokens = None
        return {
            "provider": provider_id,
            "model_id": str(model_id).strip(),
            "context_tokens": context_tokens,
            "local": bool(provider_row.get("local", False)),
            "enabled": bool(row.get("enabled", False)),
            "available": bool(available),
            "routable": bool(routable),
            "quality_rank": int(row.get("quality_rank", 0) or 0),
            "price_in": pricing.get("input_per_million_tokens"),
            "price_out": pricing.get("output_per_million_tokens"),
            "health_status": "unknown",
        }

    def _build_model_watch_proposal(
        self,
        *,
        delta_rows: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        self._model_watch_last_proposal_evaluation = {
            "status": "no_delta_candidates",
            "reason": "no_delta_candidates",
            "current_model": None,
            "current_utility": None,
            "min_improvement": round(max(0.0, float(self.config.model_watch_min_improvement)), 4),
            "top_candidate": None,
            "rejected_candidates": [],
            "policy_rejections": [],
        }
        registry_document = self.registry_document if isinstance(self.registry_document, dict) else {}
        defaults = registry_document.get("defaults") if isinstance(registry_document.get("defaults"), dict) else {}
        models = registry_document.get("models") if isinstance(registry_document.get("models"), dict) else {}
        providers = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
        current_model = str(defaults.get("default_model") or "").strip()
        if not current_model:
            self._model_watch_last_proposal_evaluation["reason"] = "default_model_missing"
            return None
        if current_model not in models:
            self._model_watch_last_proposal_evaluation["reason"] = "default_model_not_found"
            return None

        delta_model_ids = sorted(
            {
                str(row.get("model_id") or "").strip()
                for row in delta_rows
                if isinstance(row, dict) and str(row.get("model_id") or "").strip()
            }
        )
        if not delta_model_ids:
            self._model_watch_last_proposal_evaluation["reason"] = "no_delta_candidates"
            return None
        delta_id_set = {item for item in delta_model_ids if item}

        candidate_ids = sorted({current_model, *delta_model_ids})
        candidate_rows: list[dict[str, Any]] = []
        for model_id in candidate_ids:
            row = models.get(model_id) if isinstance(models.get(model_id), dict) else {}
            if not isinstance(row, dict):
                continue
            provider_id = str(row.get("provider") or "").strip().lower()
            provider_row = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
            if not provider_id or not isinstance(provider_row, dict):
                continue
            candidate_rows.append(
                self._model_watch_scoring_candidate(
                    model_id=model_id,
                    row=row,
                    provider_row=provider_row,
                )
            )
        if not candidate_rows:
            self._model_watch_last_proposal_evaluation["reason"] = "no_scoring_candidates"
            return None

        default_policy = self._value_policy("default")
        allow_remote_fallback = bool(defaults.get("allow_remote_fallback", True))
        allowed_rows, rejected_rows = rank_candidates_by_utility(
            candidate_rows,
            policy=default_policy,
            allow_remote_fallback=allow_remote_fallback,
        )
        all_rows_by_id = {str(row.model_id): row for row in [*allowed_rows, *rejected_rows]}
        current_row = all_rows_by_id.get(current_model)
        if current_row is None:
            self._model_watch_last_proposal_evaluation["reason"] = "default_not_routable"
            return None

        delta_allowed = [
            row for row in allowed_rows if str(row.model_id) in delta_id_set and str(row.model_id) != current_model
        ]
        delta_rejected = [
            row for row in rejected_rows if str(row.model_id) in delta_id_set and str(row.model_id) != current_model
        ]
        rejected_candidates = [
            {"model_id": str(row.model_id), "rejected_by": str(row.rejected_by or "unknown")}
            for row in delta_rejected
        ]
        policy_rejections = sorted(
            {
                str(row.rejected_by or "unknown")
                for row in delta_rejected
                if str(row.rejected_by or "").strip()
            }
        )
        top_new = delta_allowed[0] if delta_allowed else None
        improvement_ratio = utility_delta(current=current_row, candidate=top_new)
        quality_delta = (
            round(float(top_new.quality) - float(current_row.quality), 4)
            if top_new is not None
            else 0.0
        )
        expected_cost_delta = (
            round(float(top_new.expected_cost_per_1m) - float(current_row.expected_cost_per_1m), 4)
            if top_new is not None
            else 0.0
        )
        min_improvement = max(0.0, float(self.config.model_watch_min_improvement))
        self._model_watch_last_proposal_evaluation = {
            "status": "evaluated",
            "reason": "evaluated",
            "current_model": current_model,
            "current_utility": round(float(current_row.utility), 6),
            "min_improvement": round(float(min_improvement), 4),
            "top_candidate": (
                {
                    "model_id": str(top_new.model_id),
                    "provider": str(top_new.provider),
                    "utility_delta": round(float(improvement_ratio), 6),
                    "quality_delta": round(float(quality_delta), 6),
                    "expected_cost_delta": round(float(expected_cost_delta), 6),
                    "expected_cost_per_1m": round(float(top_new.expected_cost_per_1m), 6),
                }
                if top_new is not None
                else None
            ),
            "rejected_candidates": rejected_candidates,
            "policy_rejections": policy_rejections,
        }
        if top_new is None:
            self._model_watch_last_proposal_evaluation["reason"] = "no_allowed_delta_candidate"
            return None
        if improvement_ratio < min_improvement:
            self._model_watch_last_proposal_evaluation["reason"] = "improvement_below_threshold"
            return None

        from_model = current_model
        to_model = str(top_new.model_id)
        to_provider = str(top_new.provider)
        score_delta = round(improvement_ratio, 4)
        tradeoffs: list[str] = []
        if expected_cost_delta > 0:
            tradeoffs.append("higher_expected_cost")
        if float(top_new.latency) > float(current_row.latency):
            tradeoffs.append("higher_latency")
        if float(top_new.risk) > float(current_row.risk):
            tradeoffs.append("higher_instability_risk")
        if policy_rejections:
            tradeoffs.append(f"rejected_alternatives:{','.join(policy_rejections)}")
        details_lines = [
            f"Current default: {from_model} (utility {float(current_row.utility):.2f})",
            f"Proposed default: {to_model} (utility {float(top_new.utility):.2f})",
            f"Utility improvement: +{score_delta:.2f} (threshold {min_improvement:.2f})",
            f"Quality delta: {quality_delta:+.2f}",
            f"Expected cost delta per 1M tokens: {expected_cost_delta:+.2f}",
        ]
        if tradeoffs:
            details_lines.append(f"Tradeoff: {tradeoffs[0]}")
        if policy_rejections:
            details_lines.append(f"Policy rejection reason(s): {', '.join(policy_rejections)}")
        self._model_watch_last_proposal_evaluation["status"] = "proposal_created"
        self._model_watch_last_proposal_evaluation["reason"] = "proposal_created"

        plan_rows = [
            {
                "id": "01_defaults.set",
                "kind": "safe_action",
                "action": "defaults.set",
                "reason": f"Switch default model to {to_model}.",
                "params": {
                    "default_provider": to_provider,
                    "default_model": to_model,
                    "allow_remote_fallback": bool(defaults.get("allow_remote_fallback", True)),
                },
                "safe_to_execute": True,
            }
        ]
        prompt = "\n".join(
            [
                "Model Watch found a better default candidate.",
                f"Current default: {from_model}",
                f"Recommended: {to_model} (+{score_delta:.2f})",
                f"Cost delta per 1M tokens: {expected_cost_delta:+.2f}",
                "Reply 1 to switch, 2 to snooze, or 3 for details.",
            ]
        )
        return {
            "issue_code": "model_watch.proposal",
            "from_model": from_model,
            "to_model": to_model,
            "provider": to_provider,
            "score_delta": score_delta,
            "utility_delta": score_delta,
            "quality_delta": round(float(quality_delta), 4),
            "expected_cost_delta": round(float(expected_cost_delta), 4),
            "policy_rejections": policy_rejections,
            "message": prompt,
            "details": "\n".join(details_lines),
            "plan_rows": plan_rows,
            "choices": [
                {"id": "switch_to_proposal", "label": f"Switch to {to_model}", "recommended": True},
                {"id": "snooze_model_watch", "label": "Snooze", "recommended": False},
                {"id": "details", "label": "Show details", "recommended": False},
            ],
        }

    def _persist_model_watch_proposal_state(self, *, proposal: dict[str, Any]) -> dict[str, Any]:
        now_epoch = int(time.time())
        proposal_type = str(proposal.get("proposal_type") or "switch_default").strip().lower()
        prompt_question = (
            "Do you want to download/install this model locally?"
            if proposal_type == "local_download"
            else "Do you want to switch to the proposed model?"
        )
        issue_hash_payload = {
            "issue_code": str(proposal.get("issue_code") or ""),
            "proposal_type": proposal_type,
            "from_model": str(proposal.get("from_model") or ""),
            "to_model": str(proposal.get("to_model") or ""),
            "repo_id": str(proposal.get("repo_id") or ""),
            "revision": str(proposal.get("revision") or ""),
            "score_delta": float(proposal.get("score_delta") or 0.0),
        }
        issue_hash = hashlib.sha256(
            json.dumps(issue_hash_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        state = {
            "active": True,
            "issue_hash": issue_hash,
            "issue_code": str(proposal.get("issue_code") or "model_watch.proposal"),
            "step": "awaiting_choice",
            "question": prompt_question,
            "choices": proposal.get("choices") if isinstance(proposal.get("choices"), list) else [],
            "pending_plan": proposal.get("plan_rows") if isinstance(proposal.get("plan_rows"), list) else [],
            "pending_confirm_token": None,
            "pending_created_ts": None,
            "pending_expires_ts": None,
            "pending_issue_code": None,
            "last_prompt_ts": now_epoch,
            "proposal_type": proposal_type,
            "proposal_details": str(proposal.get("details") or "").strip() or None,
            "openrouter_last_test": (
                self._llm_fixit_store.state.get("openrouter_last_test")
                if isinstance(self._llm_fixit_store.state, dict)
                else None
            ),
        }
        return self._llm_fixit_store.save(state)

    def _notify_model_watch_proposal(
        self,
        *,
        actor: str,
        trigger: str,
        proposal: dict[str, Any],
    ) -> dict[str, Any]:
        proposal_type = str(proposal.get("proposal_type") or "switch_default").strip().lower()
        audit_action = (
            "llm.model_watch.hf_proposal_created"
            if proposal_type == "local_download"
            else "llm.model_watch.proposal_created"
        )
        telegram = self.telegram_status()
        state = str(telegram.get("state") or "").strip().lower()
        if state != "running":
            return {"emitted": False, "reason": "telegram_not_running"}
        token, chat_id = self._resolve_telegram_target()
        if not token or not chat_id:
            return {"emitted": False, "reason": "telegram_not_configured_or_no_chat"}
        start = time.monotonic()
        try:
            self._send_telegram_message(token, chat_id, str(proposal.get("message") or "").strip())
        except Exception as exc:
            return {"emitted": False, "reason": f"telegram_send_failed:{exc.__class__.__name__}"}
        duration_ms = int((time.monotonic() - start) * 1000)
        self._audit_telegram_fixit_prompt_shown(
            chat_id=chat_id,
            status="needs_user_choice",
            issue_code=str(proposal.get("issue_code") or "model_watch.proposal"),
            step="awaiting_choice",
        )
        try:
            self.audit_log.append(
                actor=actor,
                action=audit_action,
                params={
                    "trigger": trigger,
                    "proposal_type": proposal_type,
                    "from_model": str(proposal.get("from_model") or ""),
                    "to_model": str(proposal.get("to_model") or ""),
                    "provider": str(proposal.get("provider") or ""),
                    "score_delta": float(proposal.get("score_delta") or 0.0),
                    "repo_id": str(proposal.get("repo_id") or ""),
                    "revision": str(proposal.get("revision") or ""),
                    "installability": str(proposal.get("installability") or ""),
                    "chat_id_redacted": self._redact_telegram_chat_id(chat_id),
                },
                decision="allow",
                reason="proposal_created",
                dry_run=False,
                outcome="sent",
                error_kind=None,
                duration_ms=duration_ms,
            )
        except Exception:
            pass
        return {"emitted": True, "reason": "sent"}

    def _set_model_watch_hf_status_cache(self, payload: dict[str, Any]) -> None:
        status_payload = payload if isinstance(payload, dict) else {}
        self._model_watch_hf_last_status = {
            "enabled": bool(status_payload.get("enabled", False)),
            "last_run_ts": status_payload.get("last_run_ts"),
            "last_error": str(status_payload.get("last_error") or "").strip() or None,
            "discovered_count": int(status_payload.get("discovered_count") or 0),
            "tracked_repos": int(status_payload.get("tracked_repos") or 0),
            "state_path": str(status_payload.get("state_path") or str(self._model_watch_hf_state_path)),
        }

    def model_watch_hf_status(self) -> dict[str, Any]:
        payload = model_watch_hf_status(self)
        self._set_model_watch_hf_status_cache(payload)
        return payload

    def model_watch_hf_scan(
        self,
        *,
        trigger: str = "manual",
        notify_proposal: bool = True,
        persist_proposal: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        actor = "system" if trigger == "scheduler" else "user"
        scan_payload = scan_hf_watch(self)
        self._set_model_watch_hf_status_cache(model_watch_hf_status(self))
        scan_ok = bool(scan_payload.get("ok", False))
        discovered_count = int(scan_payload.get("discovered_count") or 0)
        scanned_repos = int(scan_payload.get("scanned_repos") or 0)
        try:
            self.audit_log.append(
                actor=actor,
                action="llm.model_watch.hf_scan",
                params={
                    "trigger": trigger,
                    "enabled": bool(scan_payload.get("enabled", False)),
                    "scanned_repos": scanned_repos,
                    "discovered_count": discovered_count,
                },
                decision="allow",
                reason=(
                    "scan_completed"
                    if scan_ok
                    else f"scan_failed:{str(scan_payload.get('detail') or scan_payload.get('error') or 'unknown')}"
                ),
                dry_run=False,
                outcome="success" if scan_ok else "failed",
                error_kind=None if scan_ok else str(scan_payload.get("error") or "hf_scan_failed"),
                duration_ms=0,
            )
        except Exception:
            pass

        if not scan_ok:
            body = {
                "ok": False,
                "error": str(scan_payload.get("error") or "hf_scan_failed"),
                "detail": str(scan_payload.get("detail") or "").strip() or None,
                "trigger": trigger,
                "scan": scan_payload,
            }
            self._log_request(
                "/model_watch/hf/scan",
                False,
                {
                    "ok": False,
                    "trigger": trigger,
                    "error": str(body.get("error") or "hf_scan_failed"),
                },
            )
            return False, body

        proposal = build_hf_local_download_proposal(self, scan_payload=scan_payload)
        notify_result = {"emitted": False, "reason": "no_change"}
        if proposal is None:
            try:
                self.audit_log.append(
                    actor=actor,
                    action="llm.model_watch.hf_no_change",
                    params={
                        "trigger": trigger,
                        "scanned_repos": scanned_repos,
                        "discovered_count": discovered_count,
                    },
                    decision="allow",
                    reason="no_hf_proposal",
                    dry_run=False,
                    outcome="no_change",
                    error_kind=None,
                    duration_ms=0,
                )
            except Exception:
                pass
        else:
            if persist_proposal:
                self._persist_model_watch_proposal_state(proposal=proposal)
            if notify_proposal:
                notify_result = self._notify_model_watch_proposal(
                    actor=actor,
                    trigger=trigger,
                    proposal=proposal,
                )
            try:
                self.audit_log.append(
                    actor=actor,
                    action="llm.model_watch.hf_proposal_created",
                    params={
                        "trigger": trigger,
                        "repo_id": str(proposal.get("repo_id") or ""),
                        "revision": str(proposal.get("revision") or ""),
                        "installability": str(proposal.get("installability") or ""),
                        "notify_reason": str(notify_result.get("reason") or "not_sent"),
                    },
                    decision="allow",
                    reason="proposal_created",
                    dry_run=False,
                    outcome="sent" if bool(notify_result.get("emitted", False)) else "detected",
                    error_kind=None,
                    duration_ms=0,
                )
            except Exception:
                pass

        body = {
            "ok": True,
            "trigger": trigger,
            "scan": scan_payload,
            "proposal_created": bool(proposal is not None),
            "proposal": proposal,
            "proposal_notification_emitted": bool(notify_result.get("emitted", False)),
            "proposal_notification_reason": str(notify_result.get("reason") or "no_change"),
        }
        self._log_request(
            "/model_watch/hf/scan",
            True,
            {
                "ok": True,
                "trigger": trigger,
                "scanned_repos": scanned_repos,
                "discovered_count": discovered_count,
                "proposal_created": bool(proposal is not None),
                "proposal_notification_emitted": bool(notify_result.get("emitted", False)),
            },
        )
        return True, body

    def run_model_watch_once(self, *, trigger: str = "manual") -> tuple[bool, dict[str, Any]]:
        now = int(time.time())
        interval_seconds = max(1, int(self.config.model_watch_interval_seconds))
        actor = "system" if trigger == "scheduler" else "user"
        if not bool(self.config.model_watch_enabled):
            body = {
                "ok": True,
                "skipped": True,
                "reason": "disabled",
                "trigger": trigger,
                "next_check_after_seconds": interval_seconds,
            }
            self._log_request("/model_watch/run", True, body)
            return True, body

        if trigger == "scheduler":
            state = normalize_model_watch_state(self._model_watch_store.load())
            last_run_at = model_watch_last_run_at(state)
            if isinstance(last_run_at, int):
                elapsed = int(now - last_run_at)
                if elapsed < interval_seconds:
                    body = {
                        "ok": True,
                        "skipped": True,
                        "reason": "interval_not_elapsed",
                        "trigger": trigger,
                        "last_run_at": last_run_at,
                        "next_check_after_seconds": max(1, interval_seconds - elapsed),
                    }
                    self._safe_log_event(
                        "model_watch_tick",
                        {
                            "trigger": trigger,
                            "ran": False,
                            "reason": "interval_not_elapsed",
                            "batch_created": False,
                        },
                    )
                    return True, body

        try:
            result = run_watch_once_for_config(self.config, trigger=trigger, now_epoch=now)
        except Exception as exc:
            self._safe_log_event(
                "model_watch_tick",
                {
                    "trigger": trigger,
                    "ran": True,
                    "reason": "error",
                    "error": str(exc),
                    "batch_created": False,
                },
            )
            body = {
                "ok": False,
                "error": "model_watch_run_failed",
                "detail": str(exc),
                "trigger": trigger,
                "next_check_after_seconds": interval_seconds,
            }
            self._log_request("/model_watch/run", False, body)
            return False, body

        scan_ok = True
        try:
            catalog_delta = scan_provider_catalogs(self)
        except Exception as exc:
            scan_ok = False
            catalog_delta = CatalogDelta(
                provider_model_count=0,
                new_models=tuple(),
                changed_models=tuple(),
                providers_considered=tuple(),
                last_run_ts=now,
            )
            try:
                self.audit_log.append(
                    actor=actor,
                    action="llm.model_watch.catalog_scan",
                    params={"trigger": trigger},
                    decision="allow",
                    reason=f"scan_failed:{exc.__class__.__name__}",
                    dry_run=False,
                    outcome="failed",
                    error_kind=exc.__class__.__name__,
                    duration_ms=0,
                )
            except Exception:
                pass
        delta_rows = [
            *[dict(row) for row in catalog_delta.new_models if isinstance(row, dict)],
            *[dict(row) for row in catalog_delta.changed_models if isinstance(row, dict)],
        ]
        if scan_ok:
            try:
                self.audit_log.append(
                    actor=actor,
                    action="llm.model_watch.catalog_scan",
                    params={
                        "trigger": trigger,
                        "providers_considered": list(catalog_delta.providers_considered),
                        "provider_model_count": int(catalog_delta.provider_model_count),
                        "new_models": len(catalog_delta.new_models),
                        "changed_models": len(catalog_delta.changed_models),
                    },
                    decision="allow",
                    reason="scan_completed",
                    dry_run=False,
                    outcome="success",
                    error_kind=None,
                    duration_ms=0,
                )
            except Exception:
                pass

        provider_proposal = self._build_model_watch_proposal(delta_rows=delta_rows)
        proposal_evaluation = (
            dict(self._model_watch_last_proposal_evaluation)
            if isinstance(self._model_watch_last_proposal_evaluation, dict)
            else None
        )
        hf_scan_ok = True
        hf_scan_payload: dict[str, Any] = {
            "ok": True,
            "enabled": bool(self.config.model_watch_hf_enabled),
            "scanned_repos": 0,
            "discovered_count": 0,
            "updates": [],
        }
        hf_proposal: dict[str, Any] | None = None
        if bool(self.config.model_watch_hf_enabled):
            hf_scan_ok, hf_body = self.model_watch_hf_scan(
                trigger=trigger,
                notify_proposal=False,
                persist_proposal=False,
            )
            hf_scan_payload = (
                hf_body.get("scan")
                if isinstance(hf_body.get("scan"), dict)
                else (
                    hf_body
                    if isinstance(hf_body, dict)
                    else hf_scan_payload
                )
            )
            hf_proposal = hf_body.get("proposal") if isinstance(hf_body.get("proposal"), dict) else None

        proposal = provider_proposal if provider_proposal is not None else hf_proposal
        proposal_type = str((proposal or {}).get("proposal_type") or "switch_default").strip().lower()
        notify_result = {"emitted": False, "reason": "no_change"}
        if proposal is not None:
            self._persist_model_watch_proposal_state(proposal=proposal)
            notify_result = self._notify_model_watch_proposal(
                actor=actor,
                trigger=trigger,
                proposal=proposal,
            )
            if not bool(notify_result.get("emitted", False)):
                try:
                    self.audit_log.append(
                        actor=actor,
                        action=(
                            "llm.model_watch.hf_proposal_created"
                            if proposal_type == "local_download"
                            else "llm.model_watch.proposal_created"
                        ),
                        params={
                            "trigger": trigger,
                            "proposal_type": proposal_type,
                            "from_model": str(proposal.get("from_model") or ""),
                            "to_model": str(proposal.get("to_model") or ""),
                            "provider": str(proposal.get("provider") or ""),
                            "score_delta": float(proposal.get("score_delta") or 0.0),
                            "repo_id": str(proposal.get("repo_id") or ""),
                            "revision": str(proposal.get("revision") or ""),
                            "installability": str(proposal.get("installability") or ""),
                            "notify_reason": str(notify_result.get("reason") or "not_sent"),
                        },
                        decision="allow",
                        reason="proposal_created",
                        dry_run=False,
                        outcome="detected",
                        error_kind=None,
                        duration_ms=0,
                    )
                except Exception:
                    pass
        else:
            try:
                self.audit_log.append(
                    actor=actor,
                    action="llm.model_watch.no_change",
                    params={
                        "trigger": trigger,
                        "new_models": len(catalog_delta.new_models),
                        "changed_models": len(catalog_delta.changed_models),
                        "hf_discovered_count": int(hf_scan_payload.get("discovered_count") or 0),
                        "hf_scan_ok": bool(hf_scan_ok),
                        "proposal_eval_reason": (
                            str((proposal_evaluation or {}).get("reason") or "")
                            if isinstance(proposal_evaluation, dict)
                            else ""
                        ),
                    },
                    decision="allow",
                    reason="no_better_candidate",
                    dry_run=False,
                    outcome="no_change",
                    error_kind=None,
                    duration_ms=0,
                )
            except Exception:
                pass

        buzz_results: list[dict[str, Any]] = []
        if bool(self.config.model_watch_buzz_enabled):
            buzz_results = map_buzz_leads_to_catalog(
                leads=model_watch_buzz_scan(
                    sources_allowlist=self.config.model_watch_buzz_sources_allowlist,
                ),
                runtime=self,
            )

        body = {
            "ok": bool(result.get("ok")),
            **result,
            "trigger": trigger,
            "next_check_after_seconds": interval_seconds,
            "catalog_delta": {
                "provider_model_count": int(catalog_delta.provider_model_count),
                "new_models": [dict(row) for row in catalog_delta.new_models],
                "changed_models": [dict(row) for row in catalog_delta.changed_models],
                "providers_considered": list(catalog_delta.providers_considered),
                "last_run_ts": int(catalog_delta.last_run_ts),
            },
            "proposal_created": bool(proposal is not None),
            "proposal": proposal,
            "proposal_evaluation": proposal_evaluation,
            "proposal_type": proposal_type if proposal is not None else None,
            "proposal_notification_emitted": bool(notify_result.get("emitted", False)),
            "proposal_notification_reason": str(notify_result.get("reason") or "no_change"),
            "provider_proposal_created": bool(provider_proposal is not None),
            "hf_enabled": bool(self.config.model_watch_hf_enabled),
            "hf_scan_ok": bool(hf_scan_ok),
            "hf_scan": hf_scan_payload,
            "hf_proposal_created": bool(hf_proposal is not None),
            "hf_proposal": hf_proposal,
            "hf_proposal_selected": bool(proposal is not None and proposal is hf_proposal),
            "buzz_enabled": bool(self.config.model_watch_buzz_enabled),
            "buzz_leads": buzz_results,
        }
        self._log_request(
            "/model_watch/run",
            bool(body.get("ok")),
            {
                "ok": bool(body.get("ok")),
                "batch_id": body.get("batch_id"),
                "new_batch_created": bool(body.get("new_batch_created")),
                "fetched_candidates": int(body.get("fetched_candidates") or 0),
                "catalog_models_considered": int(body.get("catalog_models_considered") or 0),
                "catalog_snapshot_model_count": int(body.get("catalog_snapshot_model_count") or 0),
                "catalog_delta_new_models": len(catalog_delta.new_models),
                "catalog_delta_changed_models": len(catalog_delta.changed_models),
                "proposal_created": bool(proposal is not None),
                "proposal_type": proposal_type if proposal is not None else None,
                "proposal_notification_emitted": bool(notify_result.get("emitted", False)),
                "hf_discovered_count": int(hf_scan_payload.get("discovered_count") or 0),
                "hf_proposal_created": bool(hf_proposal is not None),
            },
        )
        self._safe_log_event(
            "model_watch_tick",
            {
                "trigger": trigger,
                "ran": True,
                "reason": "ok" if bool(body.get("ok")) else "error",
                "batch_created": bool(body.get("new_batch_created")),
                "batch_id": body.get("batch_id"),
            },
        )
        return bool(body.get("ok")), body

    def model_watch_refresh(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        request = payload if isinstance(payload, dict) else {}
        provider = str(request.get("provider") or "openrouter").strip().lower()
        if provider != "openrouter":
            return False, {"ok": False, "error": "provider_not_supported", "provider": provider}

        providers = (
            self.registry_document.get("providers")
            if isinstance(self.registry_document.get("providers"), dict)
            else {}
        )
        provider_payload = providers.get("openrouter") if isinstance(providers.get("openrouter"), dict) else {}
        if not isinstance(provider_payload, dict) or not provider_payload:
            return False, {"ok": False, "error": "provider_not_configured", "provider": "openrouter"}

        base_url = str(provider_payload.get("base_url") or self.config.openrouter_base_url or "").strip().rstrip("/")
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"
        headers = self._provider_request_headers(provider_payload)
        headers["Accept"] = "application/json"
        try:
            parsed = self._http_get_json(base_url + "/models", headers=headers, timeout_seconds=20.0)
            fetched_at = int(time.time())
            snapshot = build_openrouter_snapshot(raw_payload=parsed, fetched_at=fetched_at)
            saved = write_model_watch_catalog_snapshot(self._model_watch_catalog_path, snapshot)
        except Exception as exc:
            body = {
                "ok": False,
                "error": "catalog_refresh_failed",
                "detail": str(exc),
                "provider": "openrouter",
            }
            self._log_request("/model_watch/refresh", False, body)
            return False, body

        body = {
            "ok": True,
            "provider": "openrouter",
            "model_count": int(saved.get("model_count") or 0),
            "fetched_at": int(saved.get("fetched_at") or fetched_at),
            "raw_sha256": saved.get("raw_sha256"),
            "path": str(self._model_watch_catalog_path),
        }
        self._log_request("/model_watch/refresh", True, body)
        return True, body

    @staticmethod
    def _model_watch_candidate_label(candidate: dict[str, Any]) -> str:
        model = str(candidate.get("model") or "").strip()
        candidate_id = str(candidate.get("id") or "").strip()
        return model or candidate_id or "unknown-model"

    def _model_watch_explanation(self, batch: dict[str, Any]) -> dict[str, Any]:
        top = batch.get("top_pick") if isinstance(batch.get("top_pick"), dict) else None
        if not isinstance(top, dict):
            return {"top_pick": None, "reasons": [], "why_not_others": []}

        top_name = self._model_watch_candidate_label(top)
        top_score = float(top.get("score") or 0.0)
        top_subscores = top.get("subscores") if isinstance(top.get("subscores"), dict) else {}
        reasons: list[str] = [f"Top pick score: {top_score:.2f}."]
        for key in ("local_feasibility", "cost_efficiency", "quality_proxy"):
            if key in top_subscores:
                reasons.append(f"{key.replace('_', ' ').title()}: {float(top_subscores.get(key) or 0.0):.2f}.")
        top_tradeoffs = [str(item).strip() for item in (top.get("tradeoffs") or []) if str(item).strip()]
        if top_tradeoffs:
            reasons.append(f"Tradeoff: {top_tradeoffs[0]}")
        reasons = reasons[:4]

        why_not_others: list[str] = []
        other_rows = batch.get("other_candidates") if isinstance(batch.get("other_candidates"), list) else []
        for row in other_rows[:3]:
            if not isinstance(row, dict):
                continue
            label = self._model_watch_candidate_label(row)
            alt_tradeoffs = [str(item).strip() for item in (row.get("tradeoffs") or []) if str(item).strip()]
            if alt_tradeoffs:
                why_not_others.append(f"{label}: {alt_tradeoffs[0]}")
                continue
            alt_subscores = row.get("subscores") if isinstance(row.get("subscores"), dict) else {}
            best_key = None
            best_delta = 0.0
            for key in sorted(set(top_subscores.keys()) | set(alt_subscores.keys())):
                delta = float(top_subscores.get(key, 0.0) or 0.0) - float(alt_subscores.get(key, 0.0) or 0.0)
                if delta > best_delta:
                    best_delta = delta
                    best_key = key
            if best_key:
                why_not_others.append(
                    f"{label}: lower {best_key.replace('_', ' ')} "
                    f"({float(alt_subscores.get(best_key, 0.0) or 0.0):.2f} vs "
                    f"{float(top_subscores.get(best_key, 0.0) or 0.0):.2f})."
                )
            else:
                why_not_others.append(f"{label}: lower deterministic score.")

        return {
            "top_pick": top_name,
            "reasons": reasons,
            "why_not_others": why_not_others,
        }

    def model_watch_latest(self) -> dict[str, Any]:
        snapshot, snapshot_error = load_model_watch_catalog_snapshot(self._model_watch_catalog_path)
        state = normalize_model_watch_state(self._model_watch_store.load())
        latest = summarize_model_watch_batch(latest_model_watch_batch(state))
        if latest is None:
            reason = None
            if snapshot is None:
                reason = "No model catalog snapshot available; run refresh"
            return {
                "ok": True,
                "found": False,
                "reason": reason,
                "catalog_snapshot_error": snapshot_error,
            }
        return {
            "ok": True,
            "found": True,
            "batch": latest,
            "explanation": self._model_watch_explanation(latest),
        }

    def get_permissions(self) -> dict[str, Any]:
        return {"ok": True, "permissions": self.permission_store.load()}

    def update_permissions(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if not isinstance(payload, dict):
            return False, {"ok": False, "error": "payload must be an object"}
        saved = self.permission_store.update(payload)
        return True, {"ok": True, "permissions": saved}

    def get_audit(self, limit: int = 20) -> dict[str, Any]:
        return {"ok": True, "entries": self.audit_log.recent(limit=max(1, int(limit)))}

    def llm_autopilot_ledger(self, *, limit: int = 50) -> dict[str, Any]:
        rows = self._action_ledger.recent(limit=max(1, int(limit)))
        return {"ok": True, "entries": rows}

    def llm_autopilot_ledger_entry(self, ledger_id: str) -> tuple[bool, dict[str, Any]]:
        entry = self._action_ledger.get(ledger_id)
        if not isinstance(entry, dict):
            return False, {"ok": False, "error": "not_found"}
        return True, {"ok": True, "entry": entry}

    def llm_autopilot_explain_last(self) -> dict[str, Any]:
        entry = self._latest_autopilot_apply_entry()
        if not isinstance(entry, dict):
            return {"ok": True, "found": False, "last_apply": None}

        action = str(entry.get("action") or "").strip()
        ts_value = int(entry.get("ts") or 0)
        snapshot_id_before = (
            str(entry.get("snapshot_id_before") or "").strip()
            or str(entry.get("snapshot_id") or "").strip()
            or None
        )
        snapshot_id_after = str(entry.get("snapshot_id_after") or "").strip() or None
        registry_hash_after = str(entry.get("resulting_registry_hash") or "").strip() or None
        changed_ids = sorted(
            {
                str(item).strip()
                for item in (entry.get("changed_ids") or [])
                if str(item).strip()
            }
        )

        audit_entry = None
        for row in self.audit_log.recent(limit=80):
            if not isinstance(row, dict):
                continue
            if str(row.get("action") or "").strip() != action:
                continue
            if str(row.get("outcome") or "").strip() != "success":
                continue
            params = row.get("params_redacted") if isinstance(row.get("params_redacted"), dict) else {}
            audit_snapshot = str(params.get("snapshot_id") or "").strip()
            if snapshot_id_before and audit_snapshot and audit_snapshot != snapshot_id_before:
                continue
            audit_entry = row
            break

        health_payload = self.llm_health_summary()
        health = health_payload.get("health") if isinstance(health_payload.get("health"), dict) else {}
        providers_rows = health.get("providers") if isinstance(health.get("providers"), list) else []
        models_rows = health.get("models") if isinstance(health.get("models"), list) else []
        providers_down = sorted(
            {
                str(row.get("id") or "").strip().lower()
                for row in providers_rows
                if isinstance(row, dict) and str((row.get("health") or {}).get("status") or "").strip().lower() == "down"
            }
        )
        models_down = sorted(
            {
                str(row.get("id") or "").strip()
                for row in models_rows
                if isinstance(row, dict) and str((row.get("health") or {}).get("status") or "").strip().lower() == "down"
            }
        )
        drift = health.get("drift") if isinstance(health.get("drift"), dict) else {}
        drift_reasons = sorted({str(item).strip() for item in (drift.get("reasons") or []) if str(item).strip()})
        drift_details = drift.get("details") if isinstance(drift.get("details"), dict) else {}
        policy = self._autopilot_apply_policy(action)
        safety_state = self._safe_mode_status()
        parsed = urllib.parse.urlparse(str(self.listening_url or ""))
        bind_host = str(parsed.hostname or "").strip()

        rationale_lines: list[str] = []
        rationale_lines.append(f"Applied action {action}.")
        if changed_ids:
            rationale_lines.append(f"Changed IDs: {', '.join(changed_ids)}.")
        reason_value = str(entry.get("reason") or "").strip()
        if reason_value:
            rationale_lines.append(f"Ledger reason: {reason_value}.")
        audit_reason = str((audit_entry or {}).get("reason") or "").strip()
        if audit_reason:
            rationale_lines.append(f"Audit reason: {audit_reason}.")
        if drift_reasons:
            rationale_lines.append(f"Current drift reasons: {', '.join(drift_reasons)}.")
        if not rationale_lines:
            rationale_lines.append("No additional rationale is available from local audit state.")

        evidence = {
            "health": {
                "counts": health.get("counts") if isinstance(health.get("counts"), dict) else {},
                "providers_down": providers_down,
                "models_down": models_down[:10],
            },
            "catalog": {
                "status": self._catalog_store.status(),
            },
            "drift": {
                "has_drift": bool(drift.get("has_drift")),
                "reasons": drift_reasons,
                "details": {
                    "default_provider": drift_details.get("default_provider"),
                    "default_model": drift_details.get("default_model"),
                    "resolved_default_model": drift_details.get("resolved_default_model"),
                    "provider_health_status": drift_details.get("provider_health_status"),
                    "model_health_status": drift_details.get("model_health_status"),
                    "model_routable": drift_details.get("model_routable"),
                },
            },
            "policy": {
                "safe_mode": bool(self._effective_safe_mode()),
                "safe_mode_config_default": bool(self.config.llm_autopilot_safe_mode),
                "safe_mode_override": bool(safety_state.get("safe_mode_override") is True),
                "safe_mode_reason": safety_state.get("safe_mode_reason"),
                "allow_apply_reason": str(policy.get("allow_reason") or "permission_required"),
                "allow_apply_effective": bool(policy.get("allow_apply_effective")),
                "loopback": bool(policy.get("loopback") or self._host_is_loopback(bind_host)),
            },
        }
        safe_evidence = self._sanitize_public_payload(redact_audit_value(evidence))
        safe_rationale_lines = [
            str(self._sanitize_public_payload(str(line or ""))).strip()
            for line in rationale_lines
            if str(line or "").strip()
        ]
        return {
            "ok": True,
            "found": True,
            "last_apply": {
                "action": action,
                "ts": ts_value,
                "snapshot_id_before": snapshot_id_before,
                "snapshot_id_after": snapshot_id_after,
                "registry_hash_after": registry_hash_after,
                "changed_ids": changed_ids,
                "reason": str(self._sanitize_public_payload(reason_value or audit_reason or "allowed")),
                "rationale_lines": safe_rationale_lines,
                "evidence": safe_evidence,
            },
        }

    def llm_autopilot_undo(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or "user")
        entry = self._latest_autopilot_apply_entry()
        if not isinstance(entry, dict):
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.autopilot.undo",
                params={"modified_ids": []},
                decision="deny",
                reason="no_autopilot_apply_found",
                dry_run=False,
                outcome="blocked",
                error_kind="no_autopilot_apply_found",
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.autopilot.undo",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason="no_autopilot_apply_found",
                trigger="manual",
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=[],
            )
            return False, {"ok": False, "error": "no_autopilot_apply_found"}

        snapshot_id_before = (
            str(entry.get("snapshot_id_before") or "").strip()
            or str(entry.get("snapshot_id") or "").strip()
        )
        if not snapshot_id_before:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.autopilot.undo",
                params={"ledger_id": str(entry.get("id") or ""), "modified_ids": []},
                decision="deny",
                reason="snapshot_missing",
                dry_run=False,
                outcome="blocked",
                error_kind="snapshot_missing",
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.autopilot.undo",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason="snapshot_missing",
                trigger="manual",
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=[],
            )
            return False, {"ok": False, "error": "snapshot_missing"}

        rollback_ok, rollback_body = self.llm_registry_rollback(
            {
                "actor": actor,
                "snapshot_id": snapshot_id_before,
                "confirm": True,
            }
        )
        if not rollback_ok:
            error_kind = str(rollback_body.get("error") or "rollback_failed")
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.autopilot.undo",
                params={
                    "source_action": str(entry.get("action") or ""),
                    "source_ledger_id": str(entry.get("id") or ""),
                    "snapshot_id_before": snapshot_id_before,
                    "modified_ids": [],
                },
                decision="deny",
                reason=error_kind,
                dry_run=False,
                outcome="blocked",
                error_kind=error_kind,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.autopilot.undo",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason=error_kind,
                trigger="manual",
                snapshot_id=snapshot_id_before,
                resulting_registry_hash=None,
                changed_ids=[],
            )
            return False, {"ok": False, "error": error_kind}

        resulting_registry_hash = str(rollback_body.get("resulting_registry_hash") or "").strip() or None
        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor=actor,
            action="llm.autopilot.undo",
            params={
                "source_action": str(entry.get("action") or ""),
                "source_ledger_id": str(entry.get("id") or ""),
                "snapshot_id_before": snapshot_id_before,
                "resulting_registry_hash": resulting_registry_hash,
                "modified_ids": [],
            },
            decision="allow",
            reason="allowed",
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        self._record_action_ledger(
            action="llm.autopilot.undo",
            actor=actor,
            decision="allow",
            outcome="success",
            reason="allowed",
            trigger="manual",
            snapshot_id=snapshot_id_before,
            resulting_registry_hash=resulting_registry_hash,
            changed_ids=[],
        )
        return True, {
            "ok": True,
            "rolled_back_to_snapshot_id": snapshot_id_before,
            "resulting_registry_hash": resulting_registry_hash,
        }

    def llm_autopilot_unpause(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or "user")
        confirm = bool(payload.get("confirm") is True)
        if not confirm:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.autopilot.safe_mode.unpause",
                params={"modified_ids": []},
                decision="deny",
                reason="confirm_required",
                dry_run=False,
                outcome="blocked",
                error_kind="confirm_required",
                duration_ms=duration_ms,
            )
            return False, {
                "ok": False,
                "error": "confirm_required",
                "message": 'Set {"confirm": true} to clear safe mode pause.',
            }

        if not self._autopilot_apply_pause_enabled():
            return True, {
                "ok": True,
                "changed": False,
                "paused": False,
                "reason": "not_paused",
            }

        current = dict(
            self._autopilot_safety_state.state
            if isinstance(self._autopilot_safety_state.state, dict)
            else {}
        )
        current["safe_mode_override"] = None
        current["safe_mode_reason"] = None
        current["safe_mode_entered_ts"] = None
        self._autopilot_safety_state.save(current)
        self._safe_mode_last_blocked_reason = None
        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor=actor,
            action="llm.autopilot.safe_mode.unpause",
            params={"modified_ids": []},
            decision="allow",
            reason="allowed",
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        self._record_action_ledger(
            action="llm.autopilot.safe_mode.unpause",
            actor=actor,
            decision="allow",
            outcome="success",
            reason="allowed",
            trigger="manual",
            snapshot_id=None,
            resulting_registry_hash=None,
            changed_ids=[],
        )
        return True, {
            "ok": True,
            "changed": True,
            "paused": False,
            "message": "Safe mode pause cleared. Automatic apply actions can run again.",
        }

    def _bootstrap_plan(self) -> dict[str, Any]:
        document = self.registry_document if isinstance(self.registry_document, dict) else {}
        defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        snapshot = self._router.doctor_snapshot()
        model_rows = snapshot.get("models") if isinstance(snapshot.get("models"), list) else []
        model_lookup = {
            str(row.get("id") or "").strip(): row
            for row in model_rows
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        provider_lookup = {
            str(row.get("id") or "").strip().lower(): row
            for row in (snapshot.get("providers") if isinstance(snapshot.get("providers"), list) else [])
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        raw_default_provider = str(defaults.get("default_provider") or "").strip().lower() or None
        raw_default_model = str(defaults.get("default_model") or "").strip() or None
        default_model_resolved = raw_default_model
        if raw_default_model and raw_default_model not in model_lookup and raw_default_provider and ":" not in raw_default_model:
            scoped = f"{raw_default_provider}:{raw_default_model}"
            if scoped in model_lookup:
                default_model_resolved = scoped

        current_row = model_lookup.get(str(default_model_resolved or ""))
        current_routable = bool((current_row or {}).get("routable", False))
        current_health = str(((current_row or {}).get("health") or {}).get("status") or "unknown").strip().lower()
        current_capabilities = {
            str(item).strip().lower()
            for item in ((current_row or {}).get("capabilities") or [])
            if str(item).strip()
        }
        current_chat = "chat" in current_capabilities
        current_ok = bool(current_row) and current_routable and current_chat and current_health == "ok"

        if current_ok:
            return {
                "ok": True,
                "needs_bootstrap": False,
                "changes": [],
                "reasons": ["already_configured"],
                "selected_candidate": None,
                "impact": {"changes_count": 0},
            }

        candidates: list[dict[str, Any]] = []
        for model_id in sorted(model_lookup.keys()):
            row = model_lookup.get(model_id) if isinstance(model_lookup.get(model_id), dict) else {}
            provider_id = str(row.get("provider") or "").strip().lower()
            provider_payload = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
            provider_row = provider_lookup.get(provider_id) if isinstance(provider_lookup.get(provider_id), dict) else {}
            if not provider_id or not isinstance(provider_payload, dict):
                continue
            if not bool(provider_payload.get("local", False)):
                continue
            if not bool(provider_payload.get("enabled", True)) or not bool(provider_row.get("enabled", True)):
                continue
            if not bool(row.get("enabled", False)) or not bool(row.get("available", False)):
                continue
            if not bool(row.get("routable", False)):
                continue
            capabilities = {
                str(item).strip().lower()
                for item in (row.get("capabilities") or [])
                if str(item).strip()
            }
            if "chat" not in capabilities:
                continue
            health_status = str((row.get("health") or {}).get("status") or "unknown").strip().lower()
            if health_status != "ok":
                continue
            in_cost = row.get("input_cost_per_million_tokens")
            out_cost = row.get("output_cost_per_million_tokens")
            observed_cost = None
            if isinstance(in_cost, (int, float)) and isinstance(out_cost, (int, float)):
                observed_cost = float(in_cost) + float(out_cost)
            max_context = int(row.get("max_context_tokens") or 0) if row.get("max_context_tokens") is not None else 0
            candidates.append(
                {
                    "provider_id": provider_id,
                    "model_id": model_id,
                    "observed_cost": observed_cost,
                    "max_context_tokens": max(0, max_context),
                }
            )

        if not candidates:
            return {
                "ok": True,
                "needs_bootstrap": False,
                "changes": [],
                "reasons": ["no_local_chat_candidate"],
                "selected_candidate": None,
                "impact": {"changes_count": 0},
            }

        def _candidate_sort_key(row: dict[str, Any]) -> tuple[int, float, int, str]:
            cost = row.get("observed_cost")
            if isinstance(cost, float):
                return (0, float(cost), -int(row.get("max_context_tokens") or 0), str(row.get("model_id") or ""))
            return (1, 0.0, -int(row.get("max_context_tokens") or 0), str(row.get("model_id") or ""))

        candidates.sort(key=_candidate_sort_key)
        selected = candidates[0]
        selected_provider = str(selected.get("provider_id") or "").strip().lower() or None
        selected_model = str(selected.get("model_id") or "").strip() or None
        changes: list[dict[str, Any]] = []
        if raw_default_provider != selected_provider:
            changes.append(
                {
                    "kind": "defaults",
                    "field": "default_provider",
                    "before": raw_default_provider,
                    "after": selected_provider,
                    "reason": "bootstrap_local_default",
                }
            )
        if default_model_resolved != selected_model:
            changes.append(
                {
                    "kind": "defaults",
                    "field": "default_model",
                    "before": default_model_resolved,
                    "after": selected_model,
                    "reason": "bootstrap_local_default",
                }
            )
        changes.sort(key=self._plan_change_sort_key)
        rationale = f"picked {selected_model} because local+chat+routable+healthy"
        return {
            "ok": True,
            "needs_bootstrap": bool(changes),
            "changes": changes,
            "reasons": [rationale],
            "selected_candidate": selected,
            "impact": {"changes_count": len(changes)},
        }

    def _apply_bootstrap_plan(self, current: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        updated = copy.deepcopy(current if isinstance(current, dict) else {})
        defaults = updated.get("defaults") if isinstance(updated.get("defaults"), dict) else {}
        for row in sorted(
            (item for item in (plan.get("changes") or []) if isinstance(item, dict)),
            key=self._plan_change_sort_key,
        ):
            if str(row.get("kind") or "").strip().lower() != "defaults":
                continue
            field = str(row.get("field") or "").strip()
            if not field:
                continue
            defaults[field] = row.get("after")
        updated["defaults"] = defaults
        return updated

    def llm_autopilot_bootstrap(
        self,
        payload: dict[str, Any],
        *,
        trigger: str = "manual",
    ) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or ("system" if trigger == "scheduler" else "user"))
        confirm = bool(payload.get("confirm", False))
        raw_plan = self._bootstrap_plan()
        plan, safe_mode_blocked = self._apply_safe_mode_to_plan(action="llm.autopilot.bootstrap.apply", plan=raw_plan)
        modified_ids = self._plan_modified_ids(plan)
        policy = compute_autopilot_bootstrap_apply_policy(self)
        policy_allow = bool(policy.get("allow_apply_effective"))
        decision = self._modelops_permission_decision(
            "llm.autopilot.bootstrap.apply",
            params={
                "default_provider": (plan.get("selected_candidate") or {}).get("provider_id"),
                "default_model": (plan.get("selected_candidate") or {}).get("model_id"),
            },
            estimated_download_bytes=0,
            estimated_cost=None,
            risk_level="low",
            dry_run=False,
        )
        effective_allow = bool(decision.get("allow")) or bool(policy_allow)
        if not effective_allow:
            reason = str(decision.get("reason") or "action_not_permitted")
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.autopilot.bootstrap.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "policy": policy,
                    "trigger": trigger,
                },
                decision="deny",
                reason=reason,
                dry_run=False,
                outcome="blocked",
                error_kind=reason,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.autopilot.bootstrap.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason=reason,
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {"ok": False, "error": reason, "plan": plan, "modified_ids": modified_ids}
        if bool(decision.get("requires_confirmation")) and not policy_allow and not confirm:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.autopilot.bootstrap.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "policy": policy,
                    "trigger": trigger,
                },
                decision="deny",
                reason="confirmation_required",
                dry_run=False,
                outcome="blocked",
                error_kind="confirmation_required",
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.autopilot.bootstrap.apply",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason="confirmation_required",
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {"ok": False, "error": "confirmation_required", "plan": plan, "modified_ids": modified_ids}

        if not any(isinstance(item, dict) for item in (plan.get("changes") or [])):
            noop_reason = "safe_mode_blocked" if safe_mode_blocked else (self._plan_reasons(plan)[0] if self._plan_reasons(plan) else "already_configured")
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.autopilot.bootstrap.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": None,
                    "resulting_registry_hash": None,
                    "safe_mode_blocked": safe_mode_blocked,
                    "policy": policy,
                    "trigger": trigger,
                },
                decision="allow",
                reason=noop_reason,
                dry_run=False,
                outcome="noop",
                error_kind=None,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.autopilot.bootstrap.apply",
                actor=actor,
                decision="allow",
                outcome="noop",
                reason=noop_reason,
                trigger=trigger,
                snapshot_id=None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return True, {
                "ok": True,
                "applied": False,
                "plan": plan,
                "modified_ids": modified_ids,
                "safe_mode_blocked": bool(safe_mode_blocked),
                "safe_mode_blocked_reason": (safe_mode_blocked[0] if safe_mode_blocked else None),
                "safe_mode_blocked_changes": safe_mode_blocked,
            }

        saved, txn_meta = self._persist_registry_document_transactional(
            lambda current: self._apply_bootstrap_plan(current, plan)
        )
        if not saved:
            error = txn_meta
            reason = str(error.get("error") or "registry_write_failed")
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.autopilot.bootstrap.apply",
                params={
                    "impact": plan.get("impact"),
                    "modified_ids": modified_ids,
                    "changed_ids": modified_ids,
                    "snapshot_id": error.get("snapshot_id"),
                    "resulting_registry_hash": None,
                    "policy": policy,
                    "trigger": trigger,
                },
                decision="allow",
                reason=reason,
                dry_run=False,
                outcome="failed",
                error_kind=reason,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.autopilot.bootstrap.apply",
                actor=actor,
                decision="allow",
                outcome="failed",
                reason=reason,
                trigger=trigger,
                snapshot_id=str(error.get("snapshot_id") or "") or None,
                resulting_registry_hash=None,
                changed_ids=modified_ids,
            )
            return False, {**error, "plan": plan, "modified_ids": modified_ids}

        snapshot_id = str(txn_meta.get("snapshot_id") or "") or None
        snapshot_id_after = str(txn_meta.get("snapshot_id_after") or "") or None
        resulting_registry_hash = str(txn_meta.get("resulting_registry_hash") or "") or None
        success_reasons = self._plan_reasons(plan)
        success_reason = success_reasons[0] if success_reasons else "allowed"
        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor=actor,
            action="llm.autopilot.bootstrap.apply",
            params={
                "impact": plan.get("impact"),
                "modified_ids": modified_ids,
                "changed_ids": modified_ids,
                "snapshot_id": snapshot_id,
                "snapshot_id_after": snapshot_id_after,
                "resulting_registry_hash": resulting_registry_hash,
                "policy": policy,
                "trigger": trigger,
            },
            decision="allow",
            reason=success_reason,
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        self._record_action_ledger(
            action="llm.autopilot.bootstrap.apply",
            actor=actor,
            decision="allow",
            outcome="success",
            reason=success_reason,
            trigger=trigger,
            snapshot_id=snapshot_id,
            snapshot_id_after=snapshot_id_after,
            resulting_registry_hash=resulting_registry_hash,
            changed_ids=modified_ids,
        )
        return True, {
            "ok": True,
            "applied": True,
            "plan": plan,
            "modified_ids": modified_ids,
            "snapshot_id": snapshot_id,
            "snapshot_id_after": snapshot_id_after,
            "resulting_registry_hash": resulting_registry_hash,
            "safe_mode_blocked": bool(safe_mode_blocked),
            "safe_mode_blocked_reason": (safe_mode_blocked[0] if safe_mode_blocked else None),
            "safe_mode_blocked_changes": safe_mode_blocked,
        }

    def llm_registry_snapshots(self, *, limit: int = 20) -> dict[str, Any]:
        rows = self._registry_snapshot_store.list_snapshots(limit=max(1, int(limit)))
        return {"ok": True, "snapshots": rows}

    def llm_registry_rollback(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        actor = str(payload.get("actor") or "user")
        snapshot_id = str(payload.get("snapshot_id") or "").strip()
        if not snapshot_id:
            return False, {"ok": False, "error": "snapshot_id is required"}

        decision = self._modelops_permission_decision(
            "llm.registry.rollback",
            params={"snapshot_id": snapshot_id},
            estimated_download_bytes=0,
            estimated_cost=None,
            risk_level="low",
            dry_run=False,
        )
        rollback_policy = compute_registry_rollback_policy(self)
        policy_allow = bool(rollback_policy.get("allow_rollback_effective"))
        effective_allow = bool(decision.get("allow")) or policy_allow

        if not effective_allow:
            duration_ms = int((time.monotonic() - start) * 1000)
            reason = str(decision.get("reason") or "action_not_permitted")
            self.audit_log.append(
                actor=actor,
                action="llm.registry.rollback",
                params={
                    "snapshot_id": snapshot_id,
                    "rollback_policy": rollback_policy,
                },
                decision="deny",
                reason=reason,
                dry_run=False,
                outcome="blocked",
                error_kind=reason,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.registry.rollback",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason=reason,
                trigger="manual",
                snapshot_id=snapshot_id,
                resulting_registry_hash=None,
                changed_ids=[],
            )
            return False, {"ok": False, "error": reason}
        if bool(decision.get("requires_confirmation")) and not policy_allow and not bool(payload.get("confirm", False)):
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.registry.rollback",
                params={"snapshot_id": snapshot_id, "rollback_policy": rollback_policy},
                decision="deny",
                reason="confirmation_required",
                dry_run=False,
                outcome="blocked",
                error_kind="confirmation_required",
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.registry.rollback",
                actor=actor,
                decision="deny",
                outcome="blocked",
                reason="confirmation_required",
                trigger="manual",
                snapshot_id=snapshot_id,
                resulting_registry_hash=None,
                changed_ids=[],
            )
            return False, {"ok": False, "error": "confirmation_required"}

        restore_result = self._registry_snapshot_store.restore_snapshot(
            snapshot_id=snapshot_id,
            registry_path=self.registry_store.path,
        )
        if not bool(restore_result.get("ok")):
            error_kind = str(restore_result.get("error_kind") or "rollback_failed")
            duration_ms = int((time.monotonic() - start) * 1000)
            self.audit_log.append(
                actor=actor,
                action="llm.registry.rollback",
                params={"snapshot_id": snapshot_id, "rollback_policy": rollback_policy},
                decision="allow",
                reason=error_kind,
                dry_run=False,
                outcome="failed",
                error_kind=error_kind,
                duration_ms=duration_ms,
            )
            self._record_action_ledger(
                action="llm.registry.rollback",
                actor=actor,
                decision="allow",
                outcome="failed",
                reason=error_kind,
                trigger="manual",
                snapshot_id=snapshot_id,
                resulting_registry_hash=None,
                changed_ids=[],
            )
            return False, {"ok": False, "error": error_kind, "snapshot_id": snapshot_id}

        self._reload_router()
        resulting_registry_hash = str(restore_result.get("resulting_registry_hash") or "").strip() or None
        duration_ms = int((time.monotonic() - start) * 1000)
        self.audit_log.append(
            actor=actor,
            action="llm.registry.rollback",
            params={
                "snapshot_id": snapshot_id,
                "rollback_policy": rollback_policy,
                "resulting_registry_hash": resulting_registry_hash,
            },
            decision="allow",
            reason="allowed",
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=duration_ms,
        )
        self._record_action_ledger(
            action="llm.registry.rollback",
            actor=actor,
            decision="allow",
            outcome="success",
            reason="allowed",
            trigger="manual",
            snapshot_id=snapshot_id,
            resulting_registry_hash=resulting_registry_hash,
            changed_ids=[],
        )
        return True, {
            "ok": True,
            "snapshot_id": snapshot_id,
            "resulting_registry_hash": resulting_registry_hash,
        }

    def _modelops_apply_defaults(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        return self.update_defaults(
            {
                "default_provider": payload.get("default_provider"),
                "default_model": payload.get("default_model"),
            }
        )

    def _modelops_toggle_enabled(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        target_type = str(payload.get("target_type") or "").strip().lower()
        target_id = str(payload.get("id") or "").strip()
        enabled = bool(payload.get("enabled"))

        if target_type not in {"provider", "model"}:
            return False, {"ok": False, "error": "target_type must be provider or model"}
        if not target_id:
            return False, {"ok": False, "error": "id is required"}

        document = self.registry_document
        if target_type == "provider":
            providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
            provider_id = target_id.lower()
            if provider_id not in providers or not isinstance(providers.get(provider_id), dict):
                return False, {"ok": False, "error": "provider not found"}
            providers[provider_id] = {
                **providers[provider_id],
                "enabled": enabled,
            }
            document["providers"] = providers
            saved, error = self._persist_registry_document(document)
            if not saved:
                assert error is not None
                return False, error
            return True, {"ok": True, "target_type": target_type, "id": provider_id, "enabled": enabled}

        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        if target_id not in models or not isinstance(models.get(target_id), dict):
            return False, {"ok": False, "error": "model not found"}
        models[target_id] = {
            **models[target_id],
            "enabled": enabled,
        }
        document["models"] = models
        saved, error = self._persist_registry_document(document)
        if not saved:
            assert error is not None
            return False, error
        return True, {"ok": True, "target_type": target_type, "id": target_id, "enabled": enabled}

    def _modelops_permission_decision(
        self,
        action: str,
        params: dict[str, Any],
        *,
        estimated_download_bytes: int,
        estimated_cost: float | None,
        risk_level: str | None,
        dry_run: bool,
    ) -> dict[str, Any]:
        permissions = self.permission_store.load()
        request = PermissionRequest(
            action=action,
            params=params,
            estimated_cost=estimated_cost,
            estimated_bytes=estimated_download_bytes,
            risk_level=risk_level,
            dry_run=dry_run,
        )
        decision = self.permission_policy.evaluate(request, permissions)
        return {
            "allow": bool(decision.allow),
            "reason": decision.reason,
            "requires_confirmation": bool(decision.requires_confirmation),
            "mode": permissions.get("mode"),
            "permissions": permissions,
        }

    def _audit_modelops(
        self,
        *,
        actor: str,
        action: str,
        params: dict[str, Any],
        decision: str,
        reason: str,
        dry_run: bool,
        outcome: str,
        error_kind: str | None,
        duration_ms: int,
    ) -> None:
        self.audit_log.append(
            actor=actor,
            action=action,
            params=params,
            decision=decision,
            reason=reason,
            dry_run=dry_run,
            outcome=outcome,
            error_kind=error_kind,
            duration_ms=duration_ms,
        )

    def modelops_plan(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        action = str(payload.get("action") or "").strip()
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        dry_run = bool(payload.get("dry_run", True))
        actor = str(payload.get("actor") or "user")
        try:
            plan = self.modelops_planner.plan(action, params)
        except ValueError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            self._audit_modelops(
                actor=actor,
                action=action,
                params=params,
                decision="deny",
                reason=str(exc),
                dry_run=dry_run,
                outcome="failed",
                error_kind="invalid_request",
                duration_ms=duration_ms,
            )
            return False, {"ok": False, "error": str(exc)}

        estimated_download_bytes = int(plan.get("estimated_download_bytes") or 0)
        decision = self._modelops_permission_decision(
            action,
            params=plan.get("params") if isinstance(plan.get("params"), dict) else {},
            estimated_download_bytes=estimated_download_bytes,
            estimated_cost=payload.get("estimated_cost"),
            risk_level=str(plan.get("risk_level") or payload.get("risk_level") or ""),
            dry_run=dry_run,
        )
        allow = bool(decision["allow"])
        reason = str(decision["reason"] or "")
        duration_ms = int((time.monotonic() - start) * 1000)
        self._audit_modelops(
            actor=actor,
            action=action,
            params=plan.get("params") if isinstance(plan.get("params"), dict) else {},
            decision="allow" if allow else "deny",
            reason=reason,
            dry_run=dry_run,
            outcome="planned" if allow else "blocked",
            error_kind=None if allow else "policy_deny",
            duration_ms=duration_ms,
        )
        return True, {
            "ok": True,
            "action": action,
            "plan": plan,
            "decision": {
                "allow": allow,
                "reason": reason,
                "requires_confirmation": bool(decision["requires_confirmation"]),
                "mode": decision.get("mode"),
            },
        }

    def modelops_execute(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        start = time.monotonic()
        action = str(payload.get("action") or "").strip()
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        dry_run = bool(payload.get("dry_run", False))
        confirm = bool(payload.get("confirm", False))
        actor = str(payload.get("actor") or "user")
        try:
            plan = self.modelops_planner.plan(action, params)
        except ValueError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            self._audit_modelops(
                actor=actor,
                action=action,
                params=params,
                decision="deny",
                reason=str(exc),
                dry_run=dry_run,
                outcome="failed",
                error_kind="invalid_request",
                duration_ms=duration_ms,
            )
            return False, {"ok": False, "error": str(exc)}

        estimated_download_bytes = int(plan.get("estimated_download_bytes") or 0)
        decision = self._modelops_permission_decision(
            action,
            params=plan.get("params") if isinstance(plan.get("params"), dict) else {},
            estimated_download_bytes=estimated_download_bytes,
            estimated_cost=payload.get("estimated_cost"),
            risk_level=str(plan.get("risk_level") or payload.get("risk_level") or ""),
            dry_run=dry_run,
        )
        if not bool(decision["allow"]):
            duration_ms = int((time.monotonic() - start) * 1000)
            self._audit_modelops(
                actor=actor,
                action=action,
                params=plan.get("params") if isinstance(plan.get("params"), dict) else {},
                decision="deny",
                reason=str(decision["reason"] or "policy_deny"),
                dry_run=dry_run,
                outcome="blocked",
                error_kind="policy_deny",
                duration_ms=duration_ms,
            )
            return False, {"ok": False, "error": str(decision["reason"] or "policy_deny"), "plan": plan}

        if bool(decision["requires_confirmation"]) and not dry_run and not confirm:
            duration_ms = int((time.monotonic() - start) * 1000)
            self._audit_modelops(
                actor=actor,
                action=action,
                params=plan.get("params") if isinstance(plan.get("params"), dict) else {},
                decision="deny",
                reason="confirmation_required",
                dry_run=dry_run,
                outcome="blocked",
                error_kind="confirmation_required",
                duration_ms=duration_ms,
            )
            return False, {"ok": False, "error": "confirmation_required", "plan": plan}

        result = self.modelops_executor.execute_plan(plan, dry_run=dry_run)
        duration_ms = int((time.monotonic() - start) * 1000)
        self._audit_modelops(
            actor=actor,
            action=action,
            params=plan.get("params") if isinstance(plan.get("params"), dict) else {},
            decision="allow",
            reason="allowed",
            dry_run=dry_run,
            outcome="success" if bool(result.get("ok")) else "failure",
            error_kind=None if bool(result.get("ok")) else str(result.get("error") or "execution_failed"),
            duration_ms=duration_ms,
        )

        status_ok = bool(result.get("ok"))
        return status_ok, {
            "ok": status_ok,
            "action": action,
            "plan": plan,
            "result": result,
        }

    def _modelops_seen_models_path(self) -> Path:
        explicit = os.getenv("AGENT_MODELOPS_SEEN_MODELS_PATH", "").strip()
        if explicit:
            return Path(explicit).expanduser().resolve()
        runtime_path = self._runtime_state_path(
            self.config,
            configured_path=None,
            filename="modelops_seen_models.json",
        )
        if runtime_path:
            return Path(runtime_path).expanduser().resolve()
        return (Path.home() / ".local" / "share" / "personal-agent" / "modelops_seen_models.json").resolve()

    def _modelops_trace_id(self, prefix: str, payload: dict[str, Any]) -> str:
        canonical_payload = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        seed = f"{prefix}:{canonical_payload}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _normalize_modelops_purposes(payload: dict[str, Any]) -> list[str]:
        allowed = {"chat", "code", "organize", "story"}
        raw = payload.get("purposes") if isinstance(payload.get("purposes"), list) else []
        values = sorted({str(item).strip().lower() for item in raw if str(item).strip()})
        normalized = [item for item in values if item in allowed]
        return normalized or ["chat", "code", "organize", "story"]

    def _active_provider_ids(self) -> list[str]:
        providers = self.registry_document.get("providers") if isinstance(self.registry_document.get("providers"), dict) else {}
        output = [
            provider_id
            for provider_id, row in providers.items()
            if isinstance(row, dict) and bool(row.get("enabled", True))
        ]
        return sorted({str(item).strip().lower() for item in output if str(item).strip()})

    def _collect_modelops_available_models(self) -> tuple[list[ModelInfo], list[str], dict[str, int]]:
        providers = self.registry_document.get("providers") if isinstance(self.registry_document.get("providers"), dict) else {}
        active_providers = self._active_provider_ids()
        warnings: list[str] = []
        provider_counts: dict[str, int] = {}
        rows: list[ModelInfo] = []

        for provider_id in active_providers:
            if provider_id == "ollama":
                discovered = list_models_ollama()
                provider_counts[provider_id] = len(discovered)
                rows.extend(discovered)
                if not discovered:
                    warnings.append("ollama_discovery_unavailable_or_empty")
                continue
            if provider_id == "openrouter":
                provider_payload = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
                api_key = self._provider_api_key(provider_payload if isinstance(provider_payload, dict) else {})
                if not api_key:
                    provider_counts[provider_id] = 0
                    warnings.append("openrouter_not_configured")
                    continue
                discovered = list_models_openrouter(api_key)
                provider_counts[provider_id] = len(discovered)
                rows.extend(discovered)
                if not discovered:
                    warnings.append("openrouter_discovery_unavailable")
                continue
            provider_counts[provider_id] = 0

        deduped: dict[str, ModelInfo] = {}
        for row in rows:
            key = f"{row.provider}:{row.model_id}"
            if key not in deduped:
                deduped[key] = row
        output = sorted(deduped.values(), key=lambda row: (row.provider, row.model_id))
        return output, sorted(set(warnings)), provider_counts

    def llm_models_check(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        trace_id = self._modelops_trace_id("llm_models_check", payload if isinstance(payload, dict) else {})
        purposes = self._normalize_modelops_purposes(payload if isinstance(payload, dict) else {})
        defaults = self.get_defaults()
        current = {
            "provider": str(defaults.get("default_provider") or "").strip().lower(),
            "model_id": str(defaults.get("default_model") or "").strip(),
        }
        available, warnings, provider_counts = self._collect_modelops_available_models()
        recommendations = recommend_models(
            available=available,
            current=current,
            purposes=purposes,
            prefer_local=bool(self.config.prefer_local),
        )
        seen_path = self._modelops_seen_models_path()
        try:
            seen_ids = load_seen_model_ids(seen_path)
        except Exception:
            seen_ids = set()
        current_ids = {f"{row.provider}:{row.model_id}" for row in available}
        new_ids = sorted(current_ids - seen_ids)
        try:
            save_seen_model_ids(seen_path, current_ids)
        except Exception:
            warnings = sorted(set([*warnings, "seen_state_write_failed"]))

        recommendations_by_purpose: dict[str, list[dict[str, Any]]] = {}
        for purpose in purposes:
            rows = recommendations.get(purpose, [])
            current_score = 0.0
            for row in rows:
                if f"{row.provider}:{row.model_id}" == current.get("model_id"):
                    current_score = max(current_score, float(row.score))
            filtered = [
                row
                for row in rows
                if f"{row.provider}:{row.model_id}" in new_ids
                or float(row.score) >= (current_score + 0.10)
            ]
            if not filtered:
                filtered = rows[:1]
            recommendations_by_purpose[purpose] = [recommendation_to_dict(row) for row in filtered[:3]]

        total_candidates = sum(len(rows) for rows in recommendations_by_purpose.values())
        message = f"Found {len(new_ids)} new candidate models. Want details or switch?"
        if total_candidates == 0:
            message = "No new candidate models were found right now."

        return True, {
            "ok": True,
            "intent": "modelops_check",
            "confidence": 1.0,
            "did_work": True,
            "error_kind": None,
            "message": message,
            "next_question": (
                "Do you want details for one recommendation or a switch plan?"
                if total_candidates > 0
                else None
            ),
            "actions": [],
            "errors": [],
            "trace_id": trace_id,
            "envelope": {
                "available_count": len(available),
                "new_models_count": len(new_ids),
                "recommendations_by_purpose": recommendations_by_purpose,
                "current_model": {
                    "provider": current.get("provider"),
                    "model": current.get("model_id"),
                },
                "warnings": warnings,
                "provider_counts": dict(sorted(provider_counts.items())),
            },
        }

    def llm_models_recommend(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        trace_id = self._modelops_trace_id("llm_models_recommend", payload if isinstance(payload, dict) else {})
        provider = str(payload.get("provider") or "").strip().lower()
        model_id = str(payload.get("model_id") or payload.get("model") or "").strip()
        purpose = str(payload.get("purpose") or "chat").strip().lower() or "chat"
        if purpose not in {"chat", "code", "organize", "story"}:
            purpose = "chat"
        if not provider or not model_id:
            return False, {
                "ok": False,
                "intent": "modelops_recommend",
                "confidence": 0.0,
                "did_work": False,
                "error_kind": "bad_request",
                "message": "provider and model_id are required.",
                "next_question": None,
                "actions": [],
                "errors": ["bad_request"],
                "trace_id": trace_id,
                "envelope": {},
            }

        defaults = self.get_defaults()
        current = {
            "provider": str(defaults.get("default_provider") or "").strip().lower(),
            "model_id": str(defaults.get("default_model") or "").strip(),
        }
        available, warnings, _provider_counts = self._collect_modelops_available_models()
        selected = next(
            (row for row in available if row.provider == provider and row.model_id == model_id),
            None,
        )
        if selected is None:
            return False, {
                "ok": False,
                "intent": "modelops_recommend",
                "confidence": 0.0,
                "did_work": False,
                "error_kind": "bad_request",
                "message": "Requested model was not found in discovered providers.",
                "next_question": None,
                "actions": [],
                "errors": ["bad_request"],
                "trace_id": trace_id,
                "envelope": {"warnings": warnings},
            }

        per_purpose = recommend_models(
            available=available,
            current=current,
            purposes=[purpose],
            prefer_local=bool(self.config.prefer_local),
        )
        selected_recommendation = recommend_models(
            available=[selected],
            current=current,
            purposes=[purpose],
            prefer_local=bool(self.config.prefer_local),
        )[purpose][0]
        top_for_purpose = per_purpose[purpose][0] if per_purpose.get(purpose) else selected_recommendation
        return True, {
            "ok": True,
            "intent": "modelops_recommend",
            "confidence": 1.0,
            "did_work": True,
            "error_kind": None,
            "message": f"Recommendation details for {provider}:{model_id} ({purpose}).",
            "next_question": "Do you want a switch plan for this model?",
            "actions": [],
            "errors": [],
            "trace_id": trace_id,
            "envelope": {
                "purpose": purpose,
                "selected": recommendation_to_dict(selected_recommendation),
                "top_for_purpose": recommendation_to_dict(top_for_purpose),
                "warnings": warnings,
            },
        }

    def llm_models_switch(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        trace_id = self._modelops_trace_id("llm_models_switch", payload if isinstance(payload, dict) else {})
        provider = str(payload.get("provider") or "").strip().lower()
        model_id = str(payload.get("model_id") or payload.get("model") or "").strip()
        purpose = str(payload.get("purpose") or "chat").strip().lower() or "chat"
        confirm = bool(payload.get("confirm", False))
        if purpose not in {"chat", "code", "organize", "story"}:
            purpose = "chat"
        if not provider or not model_id:
            return False, {
                "ok": False,
                "intent": "modelops_switch",
                "confidence": 0.0,
                "did_work": False,
                "error_kind": "bad_request",
                "message": "provider and model_id are required.",
                "next_question": None,
                "actions": [],
                "errors": ["bad_request"],
                "trace_id": trace_id,
                "envelope": {},
            }

        defaults_before = self.get_defaults()
        available, warnings, _provider_counts = self._collect_modelops_available_models()
        available_ids = {f"{row.provider}:{row.model_id}" for row in available}
        canonical_model = f"{provider}:{model_id}"
        operations: list[dict[str, Any]] = []
        if provider == "ollama" and canonical_model not in available_ids:
            operations.append(
                {
                    "id": "pull_model",
                    "action": "modelops.pull_ollama_model",
                    "params": {"model": model_id},
                }
            )
        operations.append(
            {
                "id": "set_default_model",
                "action": "modelops.set_default_model",
                "params": {"default_provider": provider, "default_model": canonical_model},
            }
        )

        if not confirm:
            return True, {
                "ok": True,
                "intent": "modelops_switch",
                "confidence": 1.0,
                "did_work": False,
                "error_kind": "needs_clarification",
                "message": "I can apply this change. Reply 'confirm' to proceed.",
                "next_question": "I can apply this change. Reply 'confirm' to proceed.",
                "actions": [],
                "errors": ["needs_clarification"],
                "trace_id": trace_id,
                "envelope": {
                    "purpose": purpose,
                    "target": {"provider": provider, "model_id": model_id},
                    "plan_steps": operations,
                    "warnings": warnings,
                },
            }

        execution_results: list[dict[str, Any]] = []
        for operation in operations:
            ok, body = self.modelops_execute(
                {
                    "action": operation.get("action"),
                    "params": operation.get("params"),
                    "confirm": True,
                    "dry_run": False,
                    "actor": "modelops_advisor",
                }
            )
            execution_results.append(
                {
                    "id": operation.get("id"),
                    "action": operation.get("action"),
                    "ok": bool(ok),
                    "body": body if isinstance(body, dict) else {},
                }
            )
            if not ok:
                message = str((body if isinstance(body, dict) else {}).get("error") or "Model switch failed.")
                return False, {
                    "ok": False,
                    "intent": "modelops_switch",
                    "confidence": 0.0,
                    "did_work": False,
                    "error_kind": classify_error_kind(payload={"ok": False, "error": message}),
                    "message": message,
                    "next_question": None,
                    "actions": [],
                    "errors": ["switch_failed"],
                    "trace_id": trace_id,
                    "envelope": {
                        "purpose": purpose,
                        "target": {"provider": provider, "model_id": model_id},
                        "execution": execution_results,
                        "rollback": {
                            "default_provider": defaults_before.get("default_provider"),
                            "default_model": defaults_before.get("default_model"),
                        },
                    },
                }

        rollback_payload = {
            "action": "modelops.set_default_model",
            "params": {
                "default_provider": defaults_before.get("default_provider"),
                "default_model": defaults_before.get("default_model"),
            },
            "confirm": True,
        }
        self._record_memory_event(
            text=f"Switched default model to {canonical_model} for {purpose}.",
            tags={
                "project": "personal-agent",
                "topic": "modelops",
                "provider": provider,
                "purpose": purpose,
            },
            source_kind="api",
            source_ref="/llm/models/switch",
        )
        return True, {
            "ok": True,
            "intent": "modelops_switch",
            "confidence": 1.0,
            "did_work": True,
            "error_kind": None,
            "message": f"Switched to {canonical_model} for {purpose}.",
            "next_question": None,
            "actions": [],
            "errors": [],
            "trace_id": trace_id,
            "envelope": {
                "purpose": purpose,
                "target": {"provider": provider, "model_id": model_id},
                "execution": execution_results,
                "rollback_suggestion": rollback_payload,
                "warnings": warnings,
            },
        }

    def webui_dev_landing_html(self) -> str:
        return (
            "<!doctype html>"
            "<html><head><meta charset='utf-8'><meta name='personal-agent-webui' content='1'>"
            "<title>Personal Agent Web UI (Dev)</title></head>"
            "<body style='font-family: sans-serif; padding: 2rem;'>"
            "<h1>Personal Agent Web UI (Dev Mode)</h1>"
            f"<p>WEBUI_DEV_PROXY=1 is enabled. Open <a href='{self.webui_dev_url}'>{self.webui_dev_url}</a>.</p>"
            "<p>The API is still available at this host for /health, /providers, /chat, and related endpoints.</p>"
            "</body></html>"
        )

    def webui_missing_html(self) -> str:
        return (
            "<!doctype html>"
            "<html><head><meta charset='utf-8'><meta name='personal-agent-webui' content='1'>"
            "<title>Personal Agent Web UI</title></head>"
            "<body style='font-family: sans-serif; padding: 2rem;'>"
            "<h1>Personal Agent Web UI</h1>"
            "<p>UI assets are missing.</p>"
            "<p>Build them with:</p>"
            "<pre>./scripts/build_webui.sh</pre>"
            "</body></html>"
        )

    def resolve_webui_file(self, request_path: str) -> tuple[Path, str] | None:
        clean_path = (request_path or "/").split("?", 1)[0].split("#", 1)[0]
        rel_path = clean_path.lstrip("/") or "index.html"

        # Keep serving scope narrow and explicit.
        if rel_path != "index.html" and not rel_path.startswith("assets/") and "/" in rel_path:
            return None
        if ".." in rel_path.split("/"):
            return None

        candidate = (self.webui_dist_path / rel_path).resolve()
        try:
            candidate.relative_to(self.webui_dist_path)
        except ValueError:
            return None

        if not candidate.is_file():
            return None

        if rel_path.startswith("assets/"):
            cache_control = "public, max-age=31536000, immutable"
        elif rel_path == "index.html":
            cache_control = "no-cache"
        else:
            cache_control = "public, max-age=3600"
        return candidate, cache_control


class APIServerHandler(BaseHTTPRequestHandler):
    runtime: AgentRuntime

    def log_message(self, format: str, *args) -> None:  # pragma: no cover - avoid noisy stdout in tests
        _ = format
        _ = args

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        cache_control: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        self._last_json_error = None
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        content_type = str(self.headers.get("Content-Type") or "").strip().lower()
        if "application/json" not in content_type:
            self._last_json_error = "content_type_not_json"
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                return parsed
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._last_json_error = "invalid_json_body"
            return {}
        self._last_json_error = "invalid_json_body"
        return {}

    def _request_trace_id(self, payload: dict[str, Any] | None = None) -> str:
        body = payload if isinstance(payload, dict) else {}
        for key in ("request_id", "trace_id"):
            value = str(body.get(key) or "").strip()
            if value:
                return value
        seed = f"{time.time_ns()}|{os.getpid()}|{getattr(self, 'path', '')}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _message_content_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text_value = part.get("text")
                if isinstance(text_value, str):
                    stripped = text_value.strip()
                    if stripped:
                        parts.append(stripped)
            if parts:
                return " ".join(parts)
        return ""

    @staticmethod
    def _extract_user_text_for_low_confidence(payload: dict[str, Any]) -> str:
        body = payload if isinstance(payload, dict) else {}
        for key in ("text", "message", "content", "input", "query"):
            raw = body.get(key)
            if not isinstance(raw, str):
                continue
            value = raw.strip()
            if value:
                return value
        raw_messages = body.get("messages")
        if isinstance(raw_messages, list):
            for row in reversed(raw_messages):
                if not isinstance(row, dict):
                    continue
                role = str(row.get("role") or "").strip().lower() or "user"
                if role != "user":
                    continue
                value = APIServerHandler._message_content_text(row.get("content"))
                if value:
                    return value
        return ""

    @staticmethod
    def _has_explicit_user_message(payload: dict[str, Any]) -> bool:
        body = payload if isinstance(payload, dict) else {}
        raw_messages = body.get("messages")
        if not isinstance(raw_messages, list):
            return False
        for row in raw_messages:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role") or "").strip().lower()
            if role != "user":
                continue
            if APIServerHandler._message_content_text(row.get("content")):
                return True
        return False

    @staticmethod
    def _extract_recent_context_for_thread_integrity(payload: dict[str, Any]) -> tuple[str | None, str | None]:
        body = payload if isinstance(payload, dict) else {}
        raw_messages = body.get("messages")
        if not isinstance(raw_messages, list):
            return None, None
        user_messages: list[str] = []
        last_assistant_message: str | None = None
        for row in raw_messages:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role") or "").strip().lower()
            content_text = APIServerHandler._message_content_text(row.get("content"))
            normalized = normalize_thread_text(content_text)
            if not normalized:
                continue
            if role == "user":
                user_messages.append(normalized)
                continue
            if role == "assistant":
                last_assistant_message = normalized
        last_user_message = user_messages[-2] if len(user_messages) >= 2 else None
        return last_user_message, last_assistant_message

    @staticmethod
    def _thread_integrity_payload(
        *,
        intent: str,
        trace_id: str,
        user_text_norm: str,
        drift_reason: str,
        thread_debug: dict[str, Any],
    ) -> dict[str, Any]:
        plan = build_thread_integrity_plan(
            user_text_norm=user_text_norm,
            drift_reason=drift_reason,
            intent=intent,
        )
        envelope = validate_envelope(
            {
                "ok": True,
                "intent": intent,
                "confidence": 0.0,
                "did_work": False,
                "error_kind": "needs_clarification",
                "message": str(plan.message or "").strip(),
                "next_question": str(plan.next_question or "").strip(),
                "actions": [],
                "errors": ["needs_clarification"],
                "trace_id": trace_id,
            }
        )
        envelope_payload = dict(envelope)
        envelope_payload["clarification"] = {
            "reason": plan.reason,
            "hints": list(plan.hints),
            "suggested_intents": list(plan.suggested_intents),
        }
        envelope_payload["thread_integrity"] = {
            "reason": str(drift_reason or "").strip().lower() or "topic_shift",
            "debug": dict(thread_debug),
        }
        return {
            "ok": bool(envelope_payload.get("ok", False)),
            "intent": envelope_payload.get("intent", "chat"),
            "confidence": float(envelope_payload.get("confidence", 0.0)),
            "did_work": bool(envelope_payload.get("did_work", False)),
            "error_kind": envelope_payload.get("error_kind", "internal_error"),
            "message": envelope_payload.get("message") or "Internal error.",
            "next_question": envelope_payload.get("next_question"),
            "actions": envelope_payload.get("actions", []),
            "errors": envelope_payload.get("errors", ["internal_error"]),
            "trace_id": envelope_payload.get("trace_id") or "trace_unknown",
            "envelope": envelope_payload,
        }

    @staticmethod
    def _clarification_payload(
        *,
        intent: str,
        trace_id: str,
        raw_text: str,
        norm_text: str,
        detector_reason: str,
    ) -> dict[str, Any]:
        plan = build_clarification_plan(
            raw_text=raw_text,
            norm_text=norm_text,
            detector_reason=detector_reason,
            intent=intent,
        )
        envelope = validate_envelope(
            {
                "ok": True,
                "intent": intent,
                "confidence": 0.0,
                "did_work": False,
                "error_kind": "needs_clarification",
                "message": str(plan.message or "").strip(),
                "next_question": str(plan.next_question or "").strip(),
                "actions": [],
                "errors": ["needs_clarification"],
                "trace_id": trace_id,
            }
        )
        envelope_payload = dict(envelope)
        envelope_payload["clarification"] = {
            "reason": plan.reason,
            "hints": list(plan.hints),
            "suggested_intents": list(plan.suggested_intents),
        }
        return {
            "ok": bool(envelope_payload.get("ok", False)),
            "intent": envelope_payload.get("intent", "chat"),
            "confidence": float(envelope_payload.get("confidence", 0.0)),
            "did_work": bool(envelope_payload.get("did_work", False)),
            "error_kind": envelope_payload.get("error_kind", "internal_error"),
            "message": envelope_payload.get("message") or "Internal error.",
            "next_question": envelope_payload.get("next_question"),
            "actions": envelope_payload.get("actions", []),
            "errors": envelope_payload.get("errors", ["internal_error"]),
            "trace_id": envelope_payload.get("trace_id") or "trace_unknown",
            "envelope": envelope_payload,
        }

    @staticmethod
    def _ambiguity_payload(
        *,
        intent: str,
        trace_id: str,
        mode: str,
        message: str,
        reason: str,
    ) -> dict[str, Any]:
        normalized_mode = str(mode or "").strip().lower() or "clarify"
        next_question = (
            str(message or "").strip()
            if normalized_mode == "clarify"
            else "Reply 1, 2, or 3?"
        )
        envelope = validate_envelope(
            {
                "ok": True,
                "intent": intent,
                "confidence": 0.0,
                "did_work": False,
                "error_kind": "needs_clarification",
                "message": str(message or "").strip(),
                "next_question": next_question,
                "actions": [],
                "errors": ["needs_clarification"],
                "trace_id": trace_id,
            }
        )
        envelope_payload = dict(envelope)
        envelope_payload["clarification"] = {
            "reason": str(reason or "").strip().lower() or "ambiguous",
            "hints": [],
            "suggested_intents": [],
        }
        envelope_payload["clarify_suggest"] = {
            "mode": normalized_mode,
            "reason": str(reason or "").strip().lower() or "ambiguous",
        }
        return {
            "ok": bool(envelope_payload.get("ok", False)),
            "intent": envelope_payload.get("intent", "chat"),
            "confidence": float(envelope_payload.get("confidence", 0.0)),
            "did_work": bool(envelope_payload.get("did_work", False)),
            "error_kind": envelope_payload.get("error_kind", "internal_error"),
            "message": envelope_payload.get("message") or "Internal error.",
            "next_question": envelope_payload.get("next_question"),
            "actions": envelope_payload.get("actions", []),
            "errors": envelope_payload.get("errors", ["internal_error"]),
            "trace_id": envelope_payload.get("trace_id") or "trace_unknown",
            "envelope": envelope_payload,
        }

    @staticmethod
    def _with_replaced_message(payload: dict[str, Any], message: str) -> dict[str, Any]:
        updated = dict(payload if isinstance(payload, dict) else {})
        text = str(message or "").strip() or "Internal error."
        updated["message"] = text
        updated["next_question"] = text
        envelope = updated.get("envelope") if isinstance(updated.get("envelope"), dict) else {}
        envelope_payload = dict(envelope)
        envelope_payload["message"] = text
        envelope_payload["next_question"] = text
        updated["envelope"] = envelope_payload
        return updated

    @staticmethod
    def _intent_assessment_obj(assessment: IntentAssessment) -> dict[str, Any]:
        return {
            "decision": assessment.decision,
            "confidence": round(float(assessment.confidence), 4),
            "candidates": [
                {
                    "intent": row.intent,
                    "score": round(float(row.score), 4),
                    "reason": row.reason,
                }
                for row in assessment.candidates
            ],
            "next_question": assessment.next_question,
        }

    @staticmethod
    def _intent_ambiguity_payload(
        *,
        intent: str,
        trace_id: str,
        assessment: IntentAssessment,
    ) -> dict[str, Any]:
        question = (
            str(assessment.next_question or "").strip()
            or "I can do that. Which of these is your goal: chat, ask, or model check/switch?"
        )
        envelope = validate_envelope(
            {
                "ok": True,
                "intent": intent,
                "confidence": 0.0,
                "did_work": False,
                "error_kind": "needs_clarification",
                "message": question,
                "next_question": question,
                "actions": [],
                "errors": ["needs_clarification"],
                "trace_id": trace_id,
            }
        )
        envelope_payload = dict(envelope)
        envelope_payload["clarification"] = {
            "reason": "intent_ambiguity",
            "hints": [
                "Tell me whether you want chat, ask, or model check/switch.",
            ],
            "suggested_intents": [],
        }
        envelope_payload["intent_assessment"] = APIServerHandler._intent_assessment_obj(assessment)
        return {
            "ok": bool(envelope_payload.get("ok", False)),
            "intent": envelope_payload.get("intent", "chat"),
            "confidence": float(envelope_payload.get("confidence", 0.0)),
            "did_work": bool(envelope_payload.get("did_work", False)),
            "error_kind": envelope_payload.get("error_kind", "internal_error"),
            "message": envelope_payload.get("message") or "Internal error.",
            "next_question": envelope_payload.get("next_question"),
            "actions": envelope_payload.get("actions", []),
            "errors": envelope_payload.get("errors", ["internal_error"]),
            "trace_id": envelope_payload.get("trace_id") or "trace_unknown",
            "envelope": envelope_payload,
        }

    def _error_envelope_payload(
        self,
        *,
        intent: str,
        message: str,
        trace_id: str,
        error_code: str,
        error_kind: str = "internal_error",
        status: int = 500,
    ) -> tuple[int, dict[str, Any]]:
        envelope = failure(
            message,
            intent=intent,
            trace_id=trace_id,
            errors=[error_code],
            error_kind=classify_error_kind(
                payload={"ok": False, "error": error_code, "message": message},
                context={"intent": intent, "error_kind": error_kind},
            ),
        )
        return status, {
            "ok": False,
            "error": error_code,
            "error_kind": envelope["error_kind"],
            "message": envelope["message"],
            "trace_id": envelope["trace_id"],
            "envelope": envelope,
        }

    def _send_method_not_allowed(
        self,
        *,
        method: str,
        path: str,
        allowed_methods: list[str],
    ) -> None:
        trace_id = self._request_trace_id()
        normalized_allowed = sorted(
            {
                str(item or "").strip().upper()
                for item in allowed_methods
                if str(item or "").strip()
            }
        )
        allowed_label = ", ".join(normalized_allowed) if normalized_allowed else "the documented method"
        status, payload = self._error_envelope_payload(
            intent=f"http.{str(method or '').strip().lower() or 'post'}",
            message=f"Method {str(method or '').strip().upper() or 'POST'} is not supported for {path}. Use {allowed_label}.",
            trace_id=trace_id,
            error_code="method_not_allowed",
            error_kind="bad_request",
            status=405,
        )
        payload = dict(payload)
        payload["allowed_methods"] = normalized_allowed
        envelope_payload = payload.get("envelope") if isinstance(payload.get("envelope"), dict) else {}
        envelope_dict = dict(envelope_payload)
        envelope_dict["allowed_methods"] = normalized_allowed
        payload["envelope"] = envelope_dict
        self._send_json(status, payload)

    def _handle_internal_error(self, method: str, exc: Exception) -> None:
        error_path = str(getattr(self, "path", "") or "")
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
        log_payload = {
            "event": "api_server.internal_error",
            "method": str(method or "").strip().upper(),
            "path": error_path,
            "error": exc.__class__.__name__,
            "message": str(exc),
            "traceback": tb,
        }
        print(json.dumps(log_payload, ensure_ascii=True, sort_keys=True), file=sys.stderr, flush=True)
        if hasattr(self, "runtime") and hasattr(self.runtime, "_safe_log_event"):
            try:
                self.runtime._safe_log_event("api_server.internal_error", log_payload)  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            trace_id = self._request_trace_id()
            status, payload = self._error_envelope_payload(
                intent=f"http.{str(method).strip().lower()}",
                message="I hit an internal error, but I’m still running. Try one of these:",
                trace_id=trace_id,
                error_code="internal_error",
                error_kind="internal_error",
                status=500,
            )
            self._send_json(status, payload)
        except OSError:
            pass

    def _path_parts(self) -> tuple[str, list[str]]:
        parsed = urllib.parse.urlparse(self.path)
        clean_path = parsed.path or "/"
        parts = [part for part in clean_path.split("/") if part]
        return clean_path, parts

    def _request_client_host(self) -> str:
        try:
            host = str((self.client_address or ("", 0))[0]).strip()
        except Exception:
            host = ""
        return host

    def _request_is_loopback(self) -> bool:
        host = self._request_client_host()
        if not host:
            return True
        return bool(self.runtime._host_is_loopback(host))

    def do_OPTIONS(self) -> None:  # noqa: N802
        try:
            self._send_json(200, {"ok": True})
        except Exception as exc:
            self._handle_internal_error("OPTIONS", exc)

    def do_GET(self) -> None:  # noqa: N802
        try:
            path, parts = self._path_parts()
            if path == "/ready":
                self._send_json(200, self.runtime.ready_status())
                return
            if path == "/health":
                self._send_json(200, self.runtime.health())
                return
            if path == "/version":
                self._send_json(200, self.runtime.version_info())
                return
            if path == "/telegram/status":
                self._send_json(200, self.runtime.telegram_status())
                return
            if path in {"/model", "/llm/model"}:
                self._send_json(200, self.runtime.model_status())
                return
            if path == "/models":
                self._send_json(200, self.runtime.models())
                return
            if path == "/config":
                self._send_json(200, self.runtime.get_config())
                return
            if path == "/defaults":
                self._send_json(200, self.runtime.get_defaults())
                return
            if path == "/providers":
                self._send_json(200, self.runtime.list_providers())
                return
            if path == "/permissions":
                self._send_json(200, self.runtime.get_permissions())
                return
            if path == "/packs":
                self._send_json(200, self.runtime.list_packs())
                return
            if path == "/audit":
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                limit_raw = query.get("limit", [20])[0]
                try:
                    limit = int(limit_raw)
                except (TypeError, ValueError):
                    limit = 20
                self._send_json(200, self.runtime.get_audit(limit=limit))
                return
            if path == "/llm/autopilot/ledger":
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                limit_raw = query.get("limit", [50])[0]
                try:
                    limit = int(limit_raw)
                except (TypeError, ValueError):
                    limit = 50
                self._send_json(200, self.runtime.llm_autopilot_ledger(limit=limit))
                return
            if len(parts) == 4 and parts[0] == "llm" and parts[1] == "autopilot" and parts[2] == "ledger":
                ok, body = self.runtime.llm_autopilot_ledger_entry(parts[3])
                self._send_json(200 if ok else 404, body)
                return
            if path == "/llm/autopilot/explain_last":
                self._send_json(200, self.runtime.llm_autopilot_explain_last())
                return
            if path == "/llm/registry/snapshots":
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                limit_raw = query.get("limit", [20])[0]
                try:
                    limit = int(limit_raw)
                except (TypeError, ValueError):
                    limit = 20
                self._send_json(200, self.runtime.llm_registry_snapshots(limit=limit))
                return
            if path == "/model_scout/status":
                self._send_json(200, self.runtime.model_scout_status())
                return
            if path == "/model_scout/suggestions":
                self._send_json(200, self.runtime.model_scout_suggestions())
                return
            if path == "/model_scout/sources":
                self._send_json(200, self.runtime.model_scout_sources())
                return
            if path == "/model_watch/latest":
                self._send_json(200, self.runtime.model_watch_latest())
                return
            if path == "/model_watch/hf/status":
                self._send_json(200, self.runtime.model_watch_hf_status())
                return
            if path == "/llm/health":
                self._send_json(200, self.runtime.llm_health_summary())
                return
            if path == "/llm/status":
                self._send_json(200, self.runtime.llm_status())
                return
            if path == "/llm/catalog":
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                provider_id = str(query.get("provider_id", [""])[0] or "").strip().lower() or None
                limit_raw = query.get("limit", [200])[0]
                try:
                    limit = int(limit_raw)
                except (TypeError, ValueError):
                    limit = 200
                self._send_json(200, self.runtime.llm_catalog(provider_id=provider_id, limit=limit))
                return
            if path == "/llm/catalog/status":
                self._send_json(200, self.runtime.llm_catalog_status())
                return
            if path == "/llm/notifications":
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                limit_raw = query.get("limit", [20])[0]
                try:
                    limit = int(limit_raw)
                except (TypeError, ValueError):
                    limit = 20
                self._send_json(200, self.runtime.llm_notifications(limit=limit))
                return
            if path == "/llm/notifications/status":
                self._send_json(200, self.runtime.llm_notifications_status())
                return
            if path == "/llm/notifications/last_change":
                self._send_json(200, self.runtime.llm_notifications_last_change())
                return
            if path == "/llm/notifications/policy":
                self._send_json(200, self.runtime.llm_notifications_policy())
                return
            if path == "/llm/support/bundle":
                self._send_json(200, self.runtime.llm_support_bundle())
                return
            if path == "/llm/support/diagnose":
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                target = str(query.get("id", [""])[0] or "").strip()
                if not target:
                    self._send_json(400, {"ok": False, "error": "id is required"})
                    return
                ok, body = self.runtime.llm_support_diagnose(target)
                self._send_json(200 if ok else 404, body)
                return

            if self._try_serve_webui(path):
                return
            self._send_json(404, {"ok": False, "error": "not_found", "path": path, "parts": parts})
        except Exception as exc:
            self._handle_internal_error("GET", exc)

    def _try_serve_webui(self, path: str) -> bool:
        if path == "/" and self.runtime.webui_dev_proxy:
            html = self.runtime.webui_dev_landing_html().encode("utf-8")
            self._send_bytes(
                200,
                html,
                content_type="text/html; charset=utf-8",
                cache_control="no-store",
            )
            return True

        if path == "/":
            resolved = self.runtime.resolve_webui_file(path)
            if resolved is not None:
                self._send_static_file(*resolved)
            else:
                fallback = self.runtime.webui_missing_html().encode("utf-8")
                self._send_bytes(
                    200,
                    fallback,
                    content_type="text/html; charset=utf-8",
                    cache_control="no-store",
                )
            return True

        if path.startswith("/assets/") or (path.count("/") == 1 and "." in path.rsplit("/", 1)[-1]):
            resolved = self.runtime.resolve_webui_file(path)
            if resolved is None:
                return False
            self._send_static_file(*resolved)
            return True

        return False

    def _send_static_file(self, file_path: Path, cache_control: str) -> None:
        try:
            body = file_path.read_bytes()
        except OSError:
            self._send_json(500, {"ok": False, "error": "static_read_failed"})
            return

        guessed_content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = guessed_content_type or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type = f"{content_type}; charset=utf-8"

        self._send_bytes(200, body, content_type=content_type, cache_control=cache_control)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path, parts = self._path_parts()
            original_path = path
            if path in {"/model", "/llm/model"}:
                self._send_method_not_allowed(method="POST", path=path, allowed_methods=["GET"])
                return
            payload = self._read_json()
            intent_assessment: IntentAssessment | None = None
            memory_debug_payload: dict[str, Any] | None = None
            json_error = str(getattr(self, "_last_json_error", "") or "").strip().lower() or None
            if path in {"/chat", "/ask", "/done"} and json_error in {"content_type_not_json", "invalid_json_body"}:
                trace_id = self._request_trace_id(payload)
                if json_error == "content_type_not_json":
                    message = "Request body must use Content-Type: application/json."
                else:
                    message = "Request body is not valid JSON."
                next_question = bad_request_next_question(error_message=message, json_error=json_error)
                envelope = failure(
                    message,
                    intent=path.lstrip("/"),
                    error_kind="bad_request",
                    trace_id=trace_id,
                    errors=["bad_request"],
                    next_question=next_question,
                )
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "error": "bad_request",
                        "error_kind": envelope["error_kind"],
                        "message": envelope["message"],
                        "next_question": envelope["next_question"],
                        "trace_id": envelope["trace_id"],
                        "envelope": envelope,
                    },
                )
                return
            if path in {"/chat", "/ask"}:
                trace_id = self._request_trace_id(payload)
                input_text = self._extract_user_text_for_low_confidence(payload)
                low_confidence = detect_low_confidence(input_text)
                intent_name = "ask" if path == "/ask" else "chat"
                norm_text = str(low_confidence.debug.get("norm") or "")
                handled_choice, choice_payload = self.runtime.consume_clarify_recovery_choice(
                    source="api",
                    text=input_text,
                )
                if handled_choice and isinstance(choice_payload, dict):
                    message = str(choice_payload.get("message") or "").strip() or "Done."
                    response = {
                        "ok": bool(choice_payload.get("ok", True)),
                        "intent": str(choice_payload.get("intent") or intent_name),
                        "confidence": float(choice_payload.get("confidence", 0.0)),
                        "did_work": bool(choice_payload.get("did_work", False)),
                        "error_kind": (
                            str(choice_payload.get("error_kind") or "").strip()
                            or ("needs_clarification" if not bool(choice_payload.get("did_work", False)) else None)
                        ),
                        "message": message,
                        "next_question": choice_payload.get("next_question"),
                        "actions": [],
                        "errors": [
                            str(item).strip()
                            for item in (
                                choice_payload.get("errors")
                                if isinstance(choice_payload.get("errors"), list)
                                else []
                            )
                            if str(item).strip()
                        ],
                        "trace_id": trace_id,
                    }
                    response["envelope"] = validate_envelope(
                        {
                            "ok": bool(response.get("ok", True)),
                            "intent": str(response.get("intent") or intent_name),
                            "confidence": float(response.get("confidence", 0.0)),
                            "did_work": bool(response.get("did_work", False)),
                            "error_kind": response.get("error_kind"),
                            "message": message,
                            "next_question": response.get("next_question"),
                            "actions": [],
                            "errors": (
                                list(response.get("errors") or [])
                                if isinstance(response.get("errors"), list)
                                else []
                            ),
                            "trace_id": trace_id,
                        }
                    )
                    self._send_json(200, response)
                    return
                if low_confidence.is_low_confidence and not self._has_explicit_user_message(payload):
                    clarification_payload = self._clarification_payload(
                        intent=intent_name,
                        trace_id=trace_id,
                        raw_text=input_text,
                        norm_text=norm_text,
                        detector_reason=low_confidence.reason,
                    )
                    if (
                        intent_name == "chat"
                        and str(low_confidence.reason or "").strip().lower() == "empty"
                    ):
                        greeting = self.runtime.consume_bootstrap_greeting_if_needed()
                        if greeting:
                            clarification_payload = self._with_replaced_message(clarification_payload, greeting)
                    self._send_json(
                        200,
                        clarification_payload,
                    )
                    return
                last_user_text_norm, last_assistant_text_norm = self._extract_recent_context_for_thread_integrity(payload)
                thread_integrity = detect_thread_drift(
                    user_text_raw=input_text,
                    user_text_norm=norm_text,
                    intent=intent_name,
                    last_user_text_norm=last_user_text_norm,
                    last_assistant_text_norm=last_assistant_text_norm,
                )
                if thread_integrity.is_thread_drift:
                    self._send_json(
                        200,
                        self._thread_integrity_payload(
                            intent=intent_name,
                            trace_id=trace_id,
                            user_text_norm=norm_text,
                            drift_reason=thread_integrity.reason,
                            thread_debug=thread_integrity.debug,
                        ),
                    )
                    return
                ambiguity = classify_ambiguity(input_text)
                if ambiguity.ambiguous:
                    availability = self.runtime.llm_availability_state()
                    if bool(availability.get("available", False)):
                        clarify_message = build_clarify_message(input_text)
                        self._send_json(
                            200,
                            self._ambiguity_payload(
                                intent=intent_name,
                                trace_id=trace_id,
                                mode="clarify",
                                message=clarify_message,
                                reason=ambiguity.reason,
                            ),
                        )
                        return
                    reason = str(availability.get("reason") or "llm_unavailable").strip().lower()
                    self.runtime.set_clarify_recovery_prompt(source="api", reason=reason)
                    suggest_message = build_suggest_message(availability_reason=reason)
                    self._send_json(
                        200,
                        self._ambiguity_payload(
                            intent=intent_name,
                            trace_id=trace_id,
                            mode="suggest",
                            message=suggest_message,
                            reason=reason,
                        ),
                    )
                    return
                intent_assessment = assess_intent_deterministic(
                    user_text_raw=input_text,
                    user_text_norm=norm_text,
                    route_intent=intent_name,
                    context={"low_confidence": False, "thread_integrity": False},
                )
                if bool(self.runtime.config.intent_llm_rerank_enabled):
                    reranked_candidates = rerank_intents_with_llm(
                        candidates=list(intent_assessment.candidates),
                        user_text=norm_text,
                        runtime=self.runtime,
                    )
                    intent_assessment = rebuild_assessment_from_candidates(
                        candidates=reranked_candidates,
                        debug={
                            **dict(intent_assessment.debug),
                            "reranked": True,
                        },
                    )
                if intent_assessment.decision == "clarify":
                    self._send_json(
                        200,
                        self._intent_ambiguity_payload(
                            intent=intent_name,
                            trace_id=trace_id,
                            assessment=intent_assessment,
                        ),
                    )
                    return
            if path in {"/ask", "/done"}:
                command = path
                prompt_text = (
                    str(payload.get("text") or "").strip()
                    or str(payload.get("message") or "").strip()
                    or str(payload.get("query") or "").strip()
                )
                forwarded_payload = dict(payload)
                raw_messages = payload.get("messages")
                if isinstance(raw_messages, list) and raw_messages:
                    normalized_messages: list[dict[str, str]] = []
                    last_user_idx = -1
                    for idx, row in enumerate(raw_messages):
                        if not isinstance(row, dict):
                            continue
                        role = str(row.get("role") or "user").strip() or "user"
                        content = str(row.get("content") or "")
                        normalized_messages.append({"role": role, "content": content})
                        if role == "user":
                            last_user_idx = len(normalized_messages) - 1
                    if normalized_messages:
                        if last_user_idx >= 0:
                            content = str(normalized_messages[last_user_idx].get("content") or "").strip()
                            if not content.startswith(command):
                                normalized_messages[last_user_idx]["content"] = f"{command} {content}".strip()
                        else:
                            normalized_messages.append({"role": "user", "content": command})
                        forwarded_payload["messages"] = normalized_messages
                elif prompt_text:
                    forwarded_payload["messages"] = [{"role": "user", "content": f"{command} {prompt_text}".strip()}]
                else:
                    trace_id = self._request_trace_id(payload)
                    envelope = failure(
                        "message is required",
                        intent=command.lstrip("/"),
                        error_kind="bad_request",
                        trace_id=trace_id,
                        errors=["bad_request"],
                    )
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": "bad_request",
                            "error_kind": envelope["error_kind"],
                            "message": envelope["message"],
                            "trace_id": envelope["trace_id"],
                            "envelope": envelope,
                        },
                    )
                    return
                payload = forwarded_payload
                path = "/chat"

            if path == "/chat":
                trace_id = self._request_trace_id(payload)
                memory_context_payload = self.runtime.build_memory_context_for_payload(
                    payload,
                    intent=str(original_path).lstrip("/") or "chat",
                    trace_id=trace_id,
                )
                if isinstance(memory_context_payload, dict):
                    selected_ids_raw = memory_context_payload.get("selected_ids")
                    if not isinstance(selected_ids_raw, list):
                        selected_ids_raw = []
                    levels_raw = memory_context_payload.get("levels")
                    if not isinstance(levels_raw, dict):
                        levels_raw = {}
                    debug_raw = memory_context_payload.get("debug")
                    if not isinstance(debug_raw, dict):
                        debug_raw = {}
                    memory_debug_payload = {
                        "selected_ids": [str(item) for item in selected_ids_raw],
                        "levels": dict(levels_raw),
                        "debug": dict(debug_raw),
                    }
                    context_text = str(memory_context_payload.get("merged_context_text") or "").strip()
                    if context_text:
                        payload = dict(payload)
                        payload["memory_context_text"] = context_text
                chat_result: tuple[bool, dict[str, Any]] | None = None

                def _run_chat() -> dict[str, Any]:
                    nonlocal chat_result
                    payload_with_trace = dict(payload)
                    payload_with_trace.setdefault("trace_id", trace_id)
                    payload_with_trace.setdefault("source_surface", "api")
                    chat_result = self.runtime.chat(payload_with_trace)
                    ok_local, body_local = chat_result
                    assistant = (
                        body_local.get("assistant")
                        if isinstance(body_local.get("assistant"), dict)
                        else {}
                    )
                    assistant_text = str((assistant or {}).get("content") or "").strip()
                    if not assistant_text:
                        assistant_text = str(body_local.get("message") or body_local.get("error") or "").strip()
                    if ok_local:
                        return ok_result(
                            intent="chat",
                            message=assistant_text or "Done.",
                            confidence=1.0,
                            did_work=True,
                            trace_id=trace_id,
                        )
                    meta_local = body_local.get("meta") if isinstance(body_local.get("meta"), dict) else {}
                    provider_local = str(meta_local.get("provider") or "").strip().lower()
                    model_local = str(meta_local.get("model") or "").strip()
                    attempts_local = meta_local.get("attempts") if isinstance(meta_local.get("attempts"), list) else []
                    if (not provider_local or not model_local) and attempts_local:
                        first_attempt = attempts_local[0] if isinstance(attempts_local[0], dict) else {}
                        provider_local = provider_local or str(first_attempt.get("provider") or "").strip().lower()
                        model_local = model_local or str(first_attempt.get("model") or "").strip()
                    error_kind = classify_error_kind(
                        payload=body_local,
                        context={
                            "route": "/chat",
                            "provider": provider_local,
                            "model": model_local,
                            "health_state": self.runtime._health_monitor.state,  # noqa: SLF001
                        },
                    )
                    base_message = assistant_text or "I couldn't complete that request."
                    friendly_message = friendly_error_message(
                        error_kind=error_kind,
                        current_message=base_message,
                        context={
                            "route": "/chat",
                            "provider": provider_local,
                            "model": model_local,
                            "health_state": self.runtime._health_monitor.state,  # noqa: SLF001
                        },
                        now_epoch=int(time.time()),
                    ) or base_message
                    next_question = (
                        bad_request_next_question(error_message=base_message)
                        if error_kind == "bad_request"
                        else None
                    )
                    return failure(
                        friendly_message,
                        intent="chat",
                        confidence=0.0,
                        error_kind=error_kind,
                        trace_id=trace_id,
                        errors=[str(body_local.get("error") or "chat_failed")],
                        next_question=next_question,
                    )

                envelope = run_with_fallback(
                    fn=_run_chat,
                    context={
                        "intent": "chat",
                        "route": "/chat",
                        "trace_id": trace_id,
                        "log_path": self.runtime.config.log_path,
                        "health_state": self.runtime._health_monitor.state,  # noqa: SLF001
                        "actions": [
                            {"label": "Retry request", "command": "Retry the same /chat request once"},
                            {"label": "Check health", "command": "GET /health"},
                        ],
                    },
                )

                if chat_result is None:
                    status, error_payload = self._error_envelope_payload(
                        intent="chat",
                        message=envelope["message"],
                        trace_id=envelope["trace_id"],
                        error_code="internal_error",
                        error_kind=str(envelope.get("error_kind") or "internal_error"),
                        status=500,
                    )
                    if memory_debug_payload is not None:
                        envelope_inner = (
                            error_payload.get("envelope")
                            if isinstance(error_payload.get("envelope"), dict)
                            else {}
                        )
                        envelope_inner = dict(envelope_inner)
                        envelope_inner["memory"] = memory_debug_payload
                        error_payload["envelope"] = envelope_inner
                    self._send_json(status, error_payload)
                    return

                ok, body = chat_result
                if not isinstance(body, dict):
                    body = {}
                assistant = body.get("assistant") if isinstance(body.get("assistant"), dict) else {}
                assistant_text = str((assistant or {}).get("content") or "").strip()
                if not assistant_text:
                    assistant_text = str(envelope.get("message") or "").strip() or "I couldn't complete that request."
                    body["assistant"] = {"role": "assistant", "content": assistant_text}
                if ok and original_path == "/chat":
                    greeting = self.runtime.consume_bootstrap_greeting_if_needed()
                    if greeting:
                        combined = greeting if not assistant_text else f"{greeting}\n\n{assistant_text}"
                        body["assistant"] = {"role": "assistant", "content": combined}
                        body["message"] = combined
                if not ok:
                    body["message"] = str(body.get("message") or "").strip() or str(envelope.get("message") or "").strip() or assistant_text
                    body["error_kind"] = str(
                        body.get("error_kind")
                        or envelope.get("error_kind")
                        or classify_error_kind(
                            payload=body,
                            context={"route": "/chat", "health_state": self.runtime._health_monitor.state},  # noqa: SLF001
                        )
                    )
                    if body["error_kind"] in {"upstream_down", "payment_required"}:
                        body["message"] = friendly_error_message(
                            error_kind=str(body["error_kind"]),
                            current_message=body["message"],
                            context={
                                "route": "/chat",
                                "provider": str((body.get("meta") or {}).get("provider") or "").strip().lower()
                                if isinstance(body.get("meta"), dict)
                                else "",
                                "model": str((body.get("meta") or {}).get("model") or "").strip()
                                if isinstance(body.get("meta"), dict)
                                else "",
                                "health_state": self.runtime._health_monitor.state,  # noqa: SLF001
                            },
                            now_epoch=int(time.time()),
                        ) or body["message"]
                        body["assistant"] = {"role": "assistant", "content": body["message"]}
                    if str(envelope.get("next_question") or "").strip():
                        body["next_question"] = envelope.get("next_question")
                    body.setdefault("trace_id", envelope.get("trace_id"))
                    body.setdefault("envelope", envelope)
                if intent_assessment is not None:
                    envelope_payload = body.get("envelope") if isinstance(body.get("envelope"), dict) else {}
                    envelope_payload = dict(envelope_payload)
                    envelope_payload["intent_assessment"] = self._intent_assessment_obj(intent_assessment)
                    body["envelope"] = envelope_payload
                if memory_debug_payload is not None:
                    envelope_payload = body.get("envelope") if isinstance(body.get("envelope"), dict) else {}
                    envelope_payload = dict(envelope_payload)
                    envelope_payload["memory"] = memory_debug_payload
                    body["envelope"] = envelope_payload
                self._send_json(200 if ok else 400, body)
                return

            if path == "/providers":
                ok, body = self.runtime.add_provider(payload)
                self._send_json(200 if ok else 400, body)
                return

            if path == "/providers/test":
                # backward-compatible endpoint
                provider_id = str(payload.get("provider") or "").strip().lower()
                if not provider_id:
                    self._send_json(400, {"ok": False, "error": "provider is required"})
                    return
                ok, body = self.runtime.test_provider(provider_id, payload)
                self._send_json(200 if ok else 400, body)
                return

            if path == "/models/refresh":
                ok, body = self.runtime.refresh_models(payload)
                self._send_json(200 if ok else 400, body)
                return

            if path == "/defaults/rollback":
                if not self._request_is_loopback():
                    self._send_json(
                        403,
                        {
                            "ok": False,
                            "error": "forbidden",
                            "error_kind": "forbidden",
                            "message": "This endpoint is restricted to loopback requests.",
                        },
                    )
                    return
                ok, body = self.runtime.rollback_defaults()
                self._send_json(200 if ok else 400, body)
                return

            if path == "/telegram/secret":
                ok, body = self.runtime.set_telegram_secret(payload)
                self._send_json(200 if ok else 400, body)
                return

            if path == "/telegram/test":
                ok, body = self.runtime.test_telegram()
                self._send_json(200 if ok else 400, body)
                return
            if path == "/model_scout/run":
                ok, body = self.runtime.run_model_scout()
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/scout/run":
                ok, body = self.runtime.run_model_scout()
                self._send_json(200 if ok else 400, body)
                return
            if path == "/model_watch/run":
                ok, body = self.runtime.run_model_watch_once(trigger="manual")
                if not ok:
                    trace_id = self._request_trace_id(payload)
                    body = body if isinstance(body, dict) else {}
                    error_kind = classify_error_kind(payload=body, context={"route": "/model_watch/run"})
                    message = (
                        str(body.get("message") or "").strip()
                        or str(body.get("detail") or "").strip()
                        or str(body.get("error") or "").strip()
                        or "Model watch run failed."
                    )
                    envelope = failure(
                        message,
                        intent="model_watch.run",
                        error_kind=error_kind,
                        trace_id=trace_id,
                        errors=[str(body.get("error") or "model_watch_run_failed")],
                    )
                    body = {
                        **body,
                        "ok": False,
                        "error_kind": envelope["error_kind"],
                        "message": envelope["message"],
                        "trace_id": envelope["trace_id"],
                        "envelope": envelope,
                    }
                self._send_json(200 if ok else 400, body)
                return
            if path == "/model_watch/refresh":
                ok, body = self.runtime.model_watch_refresh(payload)
                if not ok:
                    trace_id = self._request_trace_id(payload)
                    body = body if isinstance(body, dict) else {}
                    error_kind = classify_error_kind(payload=body, context={"route": "/model_watch/refresh"})
                    message = (
                        str(body.get("message") or "").strip()
                        or str(body.get("detail") or "").strip()
                        or str(body.get("error") or "").strip()
                        or "Model watch refresh failed."
                    )
                    envelope = failure(
                        message,
                        intent="model_watch.refresh",
                        error_kind=error_kind,
                        trace_id=trace_id,
                        errors=[str(body.get("error") or "model_watch_refresh_failed")],
                    )
                    body = {
                        **body,
                        "ok": False,
                        "error_kind": envelope["error_kind"],
                        "message": envelope["message"],
                        "trace_id": envelope["trace_id"],
                        "envelope": envelope,
                    }
                self._send_json(200 if ok else 400, body)
                return
            if path == "/model_watch/hf/scan":
                ok, body = self.runtime.model_watch_hf_scan(trigger="manual")
                if not ok:
                    trace_id = self._request_trace_id(payload)
                    body = body if isinstance(body, dict) else {}
                    error_kind = classify_error_kind(payload=body, context={"route": "/model_watch/hf/scan"})
                    message = (
                        str(body.get("message") or "").strip()
                        or str(body.get("detail") or "").strip()
                        or str(body.get("error") or "").strip()
                        or "Model watch HF scan failed."
                    )
                    envelope = failure(
                        message,
                        intent="model_watch.hf.scan",
                        error_kind=error_kind,
                        trace_id=trace_id,
                        errors=[str(body.get("error") or "model_watch_hf_scan_failed")],
                    )
                    body = {
                        **body,
                        "ok": False,
                        "error_kind": envelope["error_kind"],
                        "message": envelope["message"],
                        "trace_id": envelope["trace_id"],
                        "envelope": envelope,
                    }
                self._send_json(200 if ok else 400, body)
                return
            if path == "/packs/install":
                ok, body = self.runtime.packs_install(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/packs/approve":
                ok, body = self.runtime.packs_approve(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/packs/enable":
                ok, body = self.runtime.packs_enable(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/bootstrap/run":
                trace_id = self._request_trace_id(payload)
                if payload.get("force") is not True:
                    question = 'Include {"force": true} to run bootstrap snapshot?'
                    envelope = validate_envelope(
                        {
                            "ok": True,
                            "intent": "bootstrap",
                            "confidence": 0.0,
                            "did_work": False,
                            "error_kind": "needs_clarification",
                            "message": question,
                            "next_question": question,
                            "actions": [],
                            "errors": ["needs_clarification"],
                            "trace_id": trace_id,
                        }
                    )
                    self._send_json(
                        200,
                        {
                            "ok": bool(envelope.get("ok", False)),
                            "intent": envelope.get("intent", "bootstrap"),
                            "confidence": float(envelope.get("confidence", 0.0)),
                            "did_work": bool(envelope.get("did_work", False)),
                            "error_kind": envelope.get("error_kind", "needs_clarification"),
                            "message": envelope.get("message") or question,
                            "next_question": envelope.get("next_question"),
                            "actions": envelope.get("actions", []),
                            "errors": envelope.get("errors", ["needs_clarification"]),
                            "trace_id": envelope.get("trace_id") or trace_id,
                            "envelope": envelope,
                        },
                    )
                    return

                promote_semantic = bool(payload.get("promote_semantic", True))
                reason = str(payload.get("reason") or "").strip() or None
                ok, body = self.runtime.run_memory_v2_bootstrap(
                    source_ref=f"api_bootstrap_run:{trace_id}",
                    promote_semantic=promote_semantic,
                    reason=reason,
                )
                if not ok:
                    body = body if isinstance(body, dict) else {}
                    error_kind = classify_error_kind(payload=body, context={"route": "/bootstrap/run"})
                    message = (
                        str(body.get("message") or "").strip()
                        or str(body.get("detail") or "").strip()
                        or str(body.get("error") or "").strip()
                        or "Bootstrap run failed."
                    )
                    envelope = failure(
                        message,
                        intent="bootstrap",
                        error_kind=error_kind,
                        trace_id=trace_id,
                        errors=[str(body.get("error") or "bootstrap_failed")],
                    )
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": str(body.get("error") or "bootstrap_failed"),
                            "error_kind": envelope["error_kind"],
                            "message": envelope["message"],
                            "trace_id": envelope["trace_id"],
                            "envelope": envelope,
                        },
                    )
                    return

                body = body if isinstance(body, dict) else {}
                ingest = body.get("ingest") if isinstance(body.get("ingest"), dict) else {}
                semantic_updates = ingest.get("semantic_updates") if isinstance(ingest.get("semantic_updates"), dict) else {}
                inserted = semantic_updates.get("inserted") if isinstance(semantic_updates.get("inserted"), list) else []
                episodic_ids = ingest.get("episodic_ids") if isinstance(ingest.get("episodic_ids"), list) else []
                message = (
                    "Bootstrap snapshot captured. "
                    f"Updated {len(inserted)} semantic facts; recorded {len(episodic_ids)} episodic sections."
                )
                envelope = ok_result(
                    intent="bootstrap",
                    message=message,
                    confidence=1.0,
                    did_work=True,
                    trace_id=trace_id,
                )
                envelope_payload = dict(envelope)
                envelope_payload["bootstrap"] = {
                    "completed": True,
                    "snapshot_ts": int(body.get("created_at_ts") or int(time.time())),
                    "episodic_ids": [str(item) for item in episodic_ids],
                    "semantic_updates": semantic_updates,
                    "notes": [str(item) for item in (body.get("notes") if isinstance(body.get("notes"), list) else [])],
                }
                self._send_json(
                    200,
                    {
                        "ok": bool(envelope_payload.get("ok", False)),
                        "intent": envelope_payload.get("intent", "bootstrap"),
                        "confidence": float(envelope_payload.get("confidence", 0.0)),
                        "did_work": bool(envelope_payload.get("did_work", False)),
                        "error_kind": envelope_payload.get("error_kind"),
                        "message": envelope_payload.get("message") or message,
                        "next_question": envelope_payload.get("next_question"),
                        "actions": envelope_payload.get("actions", []),
                        "errors": envelope_payload.get("errors", []),
                        "trace_id": envelope_payload.get("trace_id") or trace_id,
                        "envelope": envelope_payload,
                    },
                )
                return
            if path == "/llm/models/check":
                ok, body = self.runtime.llm_models_check(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/models/recommend":
                ok, body = self.runtime.llm_models_recommend(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/models/switch":
                ok, body = self.runtime.llm_models_switch(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/fixit":
                ok, body = self.runtime.llm_fixit(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/health/run":
                ok, body = self.runtime.run_llm_health(trigger="manual")
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/catalog/run":
                provider_filter = str(payload.get("provider_id") or "").strip().lower() or None
                ok, body = self.runtime.run_llm_catalog_refresh(trigger="manual", provider_filter=provider_filter)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/capabilities/reconcile/plan":
                ok, body = self.runtime.llm_capabilities_reconcile_plan(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/capabilities/reconcile/apply":
                ok, body = self.runtime.llm_capabilities_reconcile_apply(payload, trigger="manual")
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/autoconfig/plan":
                ok, body = self.runtime.llm_autoconfig_plan(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/autoconfig/apply":
                ok, body = self.runtime.llm_autoconfig_apply(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/hygiene/plan":
                ok, body = self.runtime.llm_hygiene_plan(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/hygiene/apply":
                ok, body = self.runtime.llm_hygiene_apply(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/cleanup/plan":
                ok, body = self.runtime.llm_cleanup_plan(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/cleanup/apply":
                ok, body = self.runtime.llm_cleanup_apply(payload, trigger="manual")
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/self_heal/plan":
                ok, body = self.runtime.llm_self_heal_plan(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/self_heal/apply":
                ok, body = self.runtime.llm_self_heal_apply(payload, trigger="manual")
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/notifications/test":
                ok, body = self.runtime.llm_notifications_test(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/notifications/mark_read":
                ok, body = self.runtime.llm_notifications_mark_read(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/notifications/prune":
                ok, body = self.runtime.llm_notifications_prune(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/support/remediate/plan":
                ok, body = self.runtime.llm_support_remediate_plan(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/support/remediate/execute":
                ok, body = self.runtime.llm_support_remediate_execute(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/registry/rollback":
                ok, body = self.runtime.llm_registry_rollback(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/autopilot/undo":
                ok, body = self.runtime.llm_autopilot_undo(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/autopilot/unpause":
                ok, body = self.runtime.llm_autopilot_unpause(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/autopilot/bootstrap":
                ok, body = self.runtime.llm_autopilot_bootstrap(payload, trigger="manual")
                self._send_json(200 if ok else 400, body)
                return
            if path == "/modelops/plan":
                ok, body = self.runtime.modelops_plan(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/modelops/execute":
                ok, body = self.runtime.modelops_execute(payload)
                self._send_json(200 if ok else 400, body)
                return

            if len(parts) == 3 and parts[0] == "providers" and parts[2] == "secret":
                provider_id = parts[1]
                ok, body = self.runtime.set_provider_secret(provider_id, payload)
                self._send_json(200 if ok else 400, body)
                return

            if len(parts) == 3 and parts[0] == "providers" and parts[2] == "models":
                provider_id = parts[1]
                ok, body = self.runtime.add_provider_model(provider_id, payload)
                self._send_json(200 if ok else 400, body)
                return

            if len(parts) == 3 and parts[0] == "providers" and parts[2] == "test":
                provider_id = parts[1]
                ok, body = self.runtime.test_provider(provider_id, payload)
                self._send_json(200 if ok else 400, body)
                return

            if len(parts) == 3 and parts[0] == "providers" and parts[1] == "ollama" and parts[2] == "pull":
                if not self._request_is_loopback():
                    self._send_json(
                        403,
                        {
                            "ok": False,
                            "error": "forbidden",
                            "error_kind": "forbidden",
                            "message": "This endpoint is restricted to loopback requests.",
                        },
                    )
                    return
                ok, body = self.runtime.pull_ollama_model(payload)
                self._send_json(200 if ok else 400, body)
                return

            if len(parts) == 4 and parts[0] == "providers" and parts[2] == "models" and parts[3] == "refresh":
                provider_id = parts[1]
                ok, body = self.runtime.refresh_models({"provider": provider_id, **payload})
                self._send_json(200 if ok else 400, body)
                return
            if (
                len(parts) == 4
                and parts[0] == "model_scout"
                and parts[1] == "suggestions"
                and parts[3] == "dismiss"
            ):
                suggestion_id = parts[2]
                ok, body = self.runtime.dismiss_model_scout_suggestion(suggestion_id)
                self._send_json(200 if ok else 400, body)
                return
            if (
                len(parts) == 4
                and parts[0] == "model_scout"
                and parts[1] == "suggestions"
                and parts[3] == "mark_installed"
            ):
                suggestion_id = parts[2]
                ok, body = self.runtime.mark_model_scout_installed(suggestion_id)
                self._send_json(200 if ok else 400, body)
                return

            self._send_json(404, {"ok": False, "error": "not_found"})
        except Exception as exc:
            self._handle_internal_error("POST", exc)

    def do_PUT(self) -> None:  # noqa: N802
        try:
            path, parts = self._path_parts()
            payload = self._read_json()

            if path == "/config":
                ok, body = self.runtime.update_config(payload)
                self._send_json(200 if ok else 400, body)
                return

            if path == "/defaults":
                ok, body = self.runtime.update_defaults(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/permissions":
                ok, body = self.runtime.update_permissions(payload)
                self._send_json(200 if ok else 400, body)
                return

            if len(parts) == 2 and parts[0] == "providers":
                provider_id = parts[1]
                ok, body = self.runtime.update_provider(provider_id, payload)
                self._send_json(200 if ok else 400, body)
                return

            self._send_json(404, {"ok": False, "error": "not_found"})
        except Exception as exc:
            self._handle_internal_error("PUT", exc)

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            path, parts = self._path_parts()
            if len(parts) == 2 and parts[0] == "providers":
                provider_id = parts[1]
                ok, body = self.runtime.delete_provider(provider_id)
                self._send_json(200 if ok else 400, body)
                return
            self._send_json(404, {"ok": False, "error": "not_found"})
        except Exception as exc:
            self._handle_internal_error("DELETE", exc)


def build_runtime(config: Config | None = None, *, defer_bootstrap_warmup: bool = False) -> AgentRuntime:
    build_started = time.monotonic()
    if config is None:
        _startup_stdout_event("runtime.build.config_load.start")
        loaded = load_config(require_telegram_token=False)
        _startup_stdout_event(
            "runtime.build.config_load.done",
            elapsed_ms=int((time.monotonic() - build_started) * 1000),
        )
    else:
        loaded = config
        _startup_stdout_event("runtime.build.config_load.done", elapsed_ms=0, source="provided")
    _startup_stdout_event("runtime.build.init.start")
    runtime = AgentRuntime(loaded, defer_bootstrap_warmup=defer_bootstrap_warmup)
    _startup_stdout_event(
        "runtime.build.init.done",
        elapsed_ms=int((time.monotonic() - build_started) * 1000),
    )
    return runtime


def run_server(host: str, port: int) -> None:
    server_started = time.monotonic()
    _startup_stdout_event("server.start.begin", host=host, port=int(port))
    runtime = build_runtime(defer_bootstrap_warmup=True)
    resolved_git_commit = str(runtime.git_commit or "").strip() or runtime._read_git_commit() or "unknown"
    _startup_stdout_event(
        "server.start.code_identity",
        module_path=str(Path(__file__).resolve()),
        git_commit=resolved_git_commit,
    )
    _startup_stdout_event(
        "server.start.runtime_built",
        elapsed_ms=int((time.monotonic() - server_started) * 1000),
    )
    runtime_config = getattr(runtime, "config", None)
    startup_report = (
        run_startup_checks(service="api", config=runtime_config)
        if runtime_config is not None
        else {
            "trace_id": "startup-api-skipped",
            "component": "api.startup",
            "status": "PASS",
            "checks": [],
            "failure_code": None,
            "next_action": None,
        }
    )
    _startup_stdout_event(
        "server.start.startup_checks",
        status=str(startup_report.get("status") or "UNKNOWN"),
        trace_id=str(startup_report.get("trace_id") or ""),
    )
    for row in (startup_report.get("checks") if isinstance(startup_report.get("checks"), list) else []):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip().upper()
        if status not in {"WARN", "FAIL"}:
            continue
        _startup_stdout_event(
            "server.start.startup_check",
            check_id=str(row.get("check_id") or ""),
            status=status,
            failure_code=str(row.get("failure_code") or ""),
            next_action=str(row.get("next_action") or ""),
        )
    if str(startup_report.get("status") or "").strip().upper() == "FAIL":
        _startup_stdout_event(
            "server.start.startup_failed",
            trace_id=str(startup_report.get("trace_id") or ""),
            component=str(startup_report.get("component") or "api.startup"),
            failure_code=str(startup_report.get("failure_code") or "startup_check_failed"),
            next_action=str(startup_report.get("next_action") or ""),
        )
        print(
            deterministic_error_message(
                title="❌ Startup checks failed",
                trace_id=str(startup_report.get("trace_id") or "startup-api-unknown"),
                component=str(startup_report.get("component") or "api.startup"),
                failure_code=str(startup_report.get("failure_code") or "startup_check_failed"),
                next_action=str(startup_report.get("next_action") or "Run: python -m agent doctor"),
            ),
            file=sys.stderr,
            flush=True,
        )
        runtime.close()
        raise SystemExit(1)
    runtime.set_listening(host, port)
    _startup_stdout_event("server.start.route_wiring.start")

    class _Handler(APIServerHandler):
        pass

    _Handler.runtime = runtime
    _startup_stdout_event("server.start.route_wiring.done")

    try:
        _startup_stdout_event("server.start.bind.start", listening=runtime.listening_url)
        server = ThreadingHTTPServer((host, port), _Handler)
    except OSError as exc:
        print(
            f"Failed to bind Personal Agent API on {runtime.listening_url}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc
    _startup_stdout_event(
        "server.start.bind.done",
        listening=runtime.listening_url,
        elapsed_ms=int((time.monotonic() - server_started) * 1000),
    )
    runtime.mark_server_listening()

    _startup_stdout_event("server.start.telegram.start")
    runtime.start_embedded_telegram()
    _startup_stdout_event("server.start.telegram.done")

    print(
        f"Personal Agent API started pid={runtime.pid} listening={runtime.listening_url} "
        f"registry_path={runtime.registry_store.path} version={runtime.version} "
        f"git_commit={runtime.git_commit or 'unknown'}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Personal Agent HTTP API")
    parser.add_argument("--host", default=os.getenv("AGENT_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENT_API_PORT", "8765")))
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

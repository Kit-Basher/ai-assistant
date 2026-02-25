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
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from agent.config import Config, load_config
from agent.logging_utils import log_event
from agent.model_watch import (
    ModelWatchStore,
    latest_model_watch_batch,
    model_watch_last_run_at,
    normalize_model_watch_state,
    summarize_model_watch_batch,
)
from agent.model_watch_catalog import (
    build_openrouter_snapshot,
    load_latest_snapshot as load_model_watch_catalog_snapshot,
    snapshot_path_for_config as model_watch_catalog_path_for_config,
    write_snapshot_atomic as write_model_watch_catalog_snapshot,
)
from agent.model_watch_skill import run_watch_once_for_config
from agent.model_scout import build_model_scout
from agent.audit_log import AuditLog, redact as redact_audit_value
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
from agent.llm.health import HealthProbeSettings, HealthStateStore, LLMHealthMonitor
from agent.llm.hygiene import apply_hygiene_plan, build_hygiene_plan
from agent.llm.notifications import (
    NotificationStore,
    build_notification_from_diff,
    build_notification_from_state_diff,
    sanitize_notification_text,
    should_send,
)
from agent.llm.notify_delivery import DeliveryResult, LocalTarget, TelegramTarget
from agent.llm.probes import probe_model, probe_provider
from agent.llm.provider_validation import validate_provider_call_format
from agent.llm.registry_txn import RegistrySnapshotStore, apply_with_rollback
from agent.llm.self_heal import (
    apply_self_heal_plan,
    build_drift_report,
    build_self_heal_plan,
)
from agent.llm.support import (
    build_model_diagnosis,
    build_provider_diagnosis,
    build_support_remediation_plan,
    sanitize_support_payload,
)
from agent.llm.registry import RegistryStore
from agent.llm.router import LLMRouter
from agent.llm.types import LLMError, Message, Request
from agent.modelops import ModelOpsExecutor, ModelOpsPlanner, SafeRunner
from agent.orchestrator import classify_authoritative_domain, has_local_observations_block
from agent.perception import analyze_snapshot, collect_snapshot, summarize_inventory
from agent.permissions import PermissionPolicy, PermissionRequest, PermissionStore
from agent.secret_store import SecretStore
from memory.db import MemoryDB


_PROVIDER_ID_RE = re.compile(r"^[a-z0-9_-]{2,64}$")
_TELEGRAM_BOT_TOKEN_SECRET_KEY = "telegram:bot_token"
_AUTOPILOT_APPLY_ACTIONS = {
    "llm.autoconfig.apply",
    "llm.hygiene.apply",
    "llm.cleanup.apply",
    "llm.self_heal.apply",
    "llm.capabilities.reconcile.apply",
    "llm.autopilot.bootstrap.apply",
}


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


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

    def __init__(self, config: Config) -> None:
        self.config = config
        self._registry_lock = threading.RLock()
        self.secret_store = SecretStore(path=os.getenv("AGENT_SECRET_STORE_PATH", "").strip() or None)
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
        self.permission_store = PermissionStore(path=os.getenv("AGENT_PERMISSIONS_PATH", "").strip() or None)
        self.permission_policy = PermissionPolicy()
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
        model_watch_state_path = self._runtime_state_path(
            config,
            self.config.model_watch_state_path,
            "model_watch_state.json",
        )
        self._model_watch_store = ModelWatchStore(path=model_watch_state_path)
        self._model_watch_catalog_path = model_watch_catalog_path_for_config(self.config)
        installer_script = self._repo_root / "agent" / "modelops" / "install_ollama.sh"
        self.modelops_planner = ModelOpsPlanner(installer_script_path=str(installer_script))
        self.modelops_executor = ModelOpsExecutor(
            safe_runner=SafeRunner(str(installer_script)),
            apply_defaults=self._modelops_apply_defaults,
            toggle_enabled=self._modelops_toggle_enabled,
        )

        self._scheduler_stop = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_next_run: dict[str, float] = {}
        self._health_monitor = LLMHealthMonitor(
            HealthProbeSettings(
                interval_seconds=max(1, int(self.config.llm_health_interval_seconds)),
                max_probes_per_run=max(1, int(self.config.llm_health_max_probes_per_run)),
                probe_timeout_seconds=max(0.1, float(self.config.llm_health_probe_timeout_seconds)),
            ),
            store=HealthStateStore(path=self.config.llm_health_state_path),
            probe_fn=self._probe_llm_candidate,
        )
        self._catalog_store = CatalogStore(path=self.config.llm_catalog_path)
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
        self._reload_router()
        self._router.set_external_health_state(self._health_monitor.state)
        self._start_background_scheduler_if_enabled()

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

    def close(self) -> None:
        self._scheduler_stop.set()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=2.0)
        self.model_scout.close()

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

    def _scheduler_loop(self) -> None:
        latest_inventory: dict[str, Any] = {}
        while not self._scheduler_stop.wait(1.0):
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
                    self._scheduler_next_run["refresh"] = now + max(30.0, float(self.config.llm_health_interval_seconds))
            except Exception:
                self._scheduler_next_run["refresh"] = now + max(60.0, float(self.config.llm_health_interval_seconds))

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
                        86400.0, float(self.config.llm_self_heal_interval_seconds)
                    )
            except Exception:
                self._scheduler_next_run["bootstrap"] = now + max(
                    86400.0, float(self.config.llm_self_heal_interval_seconds)
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
                        60.0, float(self.config.llm_catalog_refresh_interval_seconds)
                    )
            except Exception:
                self._scheduler_next_run["catalog"] = now + max(
                    300.0, float(self.config.llm_catalog_refresh_interval_seconds)
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
                        30.0, float(self.config.llm_health_interval_seconds)
                    )
            except Exception:
                self._scheduler_next_run["capabilities_reconcile"] = now + max(
                    120.0, float(self.config.llm_health_interval_seconds)
                )

            try:
                if now >= float(self._scheduler_next_run.get("health", 0.0)):
                    _start_cycle()
                    self.run_llm_health(trigger="scheduler")
                    self._scheduler_next_run["health"] = now + max(1.0, float(self.config.llm_health_interval_seconds))
            except Exception:
                self._scheduler_next_run["health"] = now + max(30.0, float(self.config.llm_health_interval_seconds))

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
                        300.0, float(self.config.llm_hygiene_interval_seconds)
                    )
            except Exception:
                self._scheduler_next_run["hygiene"] = now + max(
                    600.0, float(self.config.llm_hygiene_interval_seconds)
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
                        300.0, float(self.config.llm_hygiene_interval_seconds)
                    )
            except Exception:
                self._scheduler_next_run["cleanup"] = now + max(
                    600.0, float(self.config.llm_hygiene_interval_seconds)
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
                        300.0, float(self.config.llm_self_heal_interval_seconds)
                    )
            except Exception:
                self._scheduler_next_run["self_heal"] = now + max(
                    600.0, float(self.config.llm_self_heal_interval_seconds)
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
                        300.0, float(self.config.llm_autoconfig_interval_seconds)
                    )
            except Exception:
                self._scheduler_next_run["autoconfig"] = now + max(
                    600.0, float(self.config.llm_autoconfig_interval_seconds)
                )

            try:
                if now >= float(self._scheduler_next_run.get("model_scout", 0.0)):
                    self.run_model_scout()
                    self._scheduler_next_run["model_scout"] = now + max(
                        60.0, float(self.config.llm_model_scout_interval_seconds)
                    )
            except Exception:
                self._scheduler_next_run["model_scout"] = now + max(
                    300.0, float(self.config.llm_model_scout_interval_seconds)
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
                            log_event(
                                self.config.log_path,
                                "model_watch_scheduler_error",
                                {"error": str(watch_body.get("error") or "run_failed")},
                            )
                except Exception as exc:
                    log_event(
                        self.config.log_path,
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
                    extra_changes=sorted({str(item).strip() for item in cycle_extra_changes if str(item).strip()}),
                    trigger="scheduler",
                )

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
        return probe_provider(
            provider_cfg,
            timeout_seconds=float(timeout_seconds),
            http_get_json=self._http_get_json,
        )

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
                "error_kind": provider_error or "provider_error",
                "status_code": provider_probe.get("status_code"),
                "message": str(provider_probe.get("detail") or ""),
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
            "error_kind": model_error or "provider_error",
            "status_code": model_probe.get("status_code"),
            "message": str(model_probe.get("detail") or ""),
        }

    def _save_registry_document(self, document: dict[str, Any]) -> None:
        with self._registry_lock:
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
        with self._registry_lock:
            result = apply_with_rollback(
                registry_path=self.registry_store.path,
                snapshot_store=self._registry_snapshot_store,
                plan_apply_fn=plan_apply_fn,
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
            if blocked_lines:
                dedup_blocked = sorted({str(item).strip() for item in blocked_lines if str(item).strip()})
                paused_detail = [f"{line} (paused: churn_detected)" for line in dedup_blocked]
                self._safe_mode_last_blocked_reason = paused_detail[0]
                return filtered, paused_detail
            blocked_reason = f"{action}: blocked because safe mode is paused (churn_detected)"
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
        defaults.setdefault("routing_mode", "auto")
        defaults.setdefault("default_provider", None)
        defaults.setdefault("default_model", None)
        defaults.setdefault("allow_remote_fallback", True)
        defaults.setdefault("fallback_chain", [])
        document["defaults"] = defaults
        return defaults

    def health(self) -> dict[str, Any]:
        snapshot = self._router.doctor_snapshot()
        return {
            "ok": True,
            "service": "personal-agent-api",
            "time": datetime.now(timezone.utc).isoformat(),
            "routing_mode": snapshot.get("routing_mode"),
            "configured_providers": [item.get("id") for item in snapshot.get("providers") or []],
            "registry_path": self.registry_store.path,
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

    def telegram_status(self) -> dict[str, Any]:
        token = (self.secret_store.get_secret(_TELEGRAM_BOT_TOKEN_SECRET_KEY) or "").strip()
        return {
            "ok": True,
            "configured": bool(token),
            "token_source": "secret_store" if token else "none",
        }

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
            response = {
                "ok": False,
                "provider": provider_key,
                "model": model_override or None,
                "error": str(models_probe.get("error") or "bad_request"),
                "message": str(models_probe.get("message") or self._provider_test_message(str(models_probe.get("error") or ""))),
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
                "message": "Provider is not available in router.",
                "models_probe": models_probe,
            }
            self._log_request(f"/providers/{provider_key}/test", False, response)
            return False, response

        if key_override and hasattr(impl, "set_api_key"):
            getattr(impl, "set_api_key")(key_override)

        start = time.monotonic()
        try:
            response_obj = impl.chat(
                Request(
                    messages=(Message(role="user", content="ping"),),
                    purpose="diagnostics",
                    task_type="diagnostics",
                ),
                model=model_override,
                timeout_seconds=timeout_seconds,
            )
        except LLMError as exc:
            error_kind = self._normalize_provider_test_error(exc.kind, exc.status_code)
            response = {
                "ok": False,
                "provider": provider_key,
                "model": model_override,
                "error": error_kind,
                "status_code": exc.status_code,
                "message": str(exc.message or self._provider_test_message(error_kind)),
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

    def get_defaults(self) -> dict[str, Any]:
        defaults = self._ensure_defaults(self.registry_document)
        return {
            "routing_mode": defaults.get("routing_mode") or self._router.policy.mode,
            "default_provider": defaults.get("default_provider"),
            "default_model": defaults.get("default_model"),
            "allow_remote_fallback": bool(defaults.get("allow_remote_fallback", True)),
        }

    def update_defaults(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        valid_modes = {
            "auto",
            "prefer_cheap",
            "prefer_best",
            "prefer_local_lowest_cost_capable",
        }

        document = self.registry_document
        defaults = self._ensure_defaults(document)
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

        if "default_model" in payload:
            model = str(payload.get("default_model") or "").strip() or None
            if model is None:
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
                defaults["default_model"] = canonical_model
        elif (
            "default_provider" in payload
            and defaults.get("default_model")
            and str(defaults.get("default_provider") or "").strip()
        ):
            model_id = str(defaults.get("default_model") or "").strip()
            if model_id and model_id in models:
                existing_provider = str((models.get(model_id) or {}).get("provider") or "").strip().lower()
                selected_provider = str(defaults.get("default_provider") or "").strip().lower()
                if existing_provider and selected_provider and existing_provider != selected_provider:
                    defaults["default_model"] = None

        if "allow_remote_fallback" in payload:
            defaults["allow_remote_fallback"] = bool(payload.get("allow_remote_fallback"))

        document["defaults"] = defaults
        saved, error = self._persist_registry_document(document)
        if not saved:
            assert error is not None
            return False, error
        return True, {"ok": True, **self.get_defaults()}

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

    def chat(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        request_started_epoch = int(time.time())
        messages = self._normalize_messages(payload)
        if not messages:
            return False, {"ok": False, "error": "messages must be a non-empty list"}

        defaults = self.get_defaults()
        model_override = str(payload.get("model") or "").strip() or defaults.get("default_model")
        provider_override = str(payload.get("provider") or "").strip().lower() or defaults.get("default_provider")
        explicit_require_tools = "require_tools" in payload
        require_tools = bool(payload.get("require_tools"))
        routed_messages = list(messages)

        last_user_text = ""
        for item in reversed(messages):
            if item.get("role") == "user":
                last_user_text = str(item.get("content") or "")
                break
        if not last_user_text:
            last_user_text = str((messages[-1] or {}).get("content") or "")

        if not explicit_require_tools:
            domains = classify_authoritative_domain(last_user_text)
            if domains and not has_local_observations_block(last_user_text):
                try:
                    local_observations = self._collect_authoritative_observations(domains)
                except Exception as exc:
                    text = self._authoritative_tool_failure_text(domains, exc)
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
                "fallback_used": bool(result.get("fallback_used")),
                "attempts": result.get("attempts") or [],
                "duration_ms": int(result.get("duration_ms") or 0),
                "error": result.get("error_class"),
                "autopilot": self._chat_autopilot_meta(request_started_epoch),
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
                try:
                    parsed = self._http_get_json(base_url + "/api/tags", headers=request_headers)
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

        notification_line = (
            "Autopilot paused applies after churn detection; use /llm/autopilot/undo or "
            "adjust safe mode policy before re-enabling."
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
        return build_drift_report(
            self.registry_document,
            summary,
            router_snapshot=snapshot,
        )

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

    def _deliver_autopilot_notification(
        self,
        *,
        message: str,
        allow_remote: bool,
    ) -> DeliveryResult:
        token, chat_id = self._resolve_telegram_target()
        telegram_target = TelegramTarget(
            token=token,
            chat_id=chat_id,
            send_fn=self._send_telegram_message,
            enabled=bool(allow_remote),
        )
        local_target = LocalTarget(enabled=True)
        payload = {"message": str(message or "").strip()}
        telegram_descriptor = telegram_target.target
        local_descriptor = local_target.target

        if telegram_descriptor.enabled and telegram_descriptor.configured:
            result = telegram_target.deliver(payload)
            if result.ok:
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
        diff = build_notification_from_state_diff(before_state, after_state, reasons, extra_changes=extra_changes)
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
        plan = build_support_remediation_plan(
            target=target,
            intent=intent,
            diagnosis=diagnosis,
            drift_report=self._current_drift_report(),
            safe_mode_enabled=bool(self._effective_safe_mode()),
        )
        return True, {"ok": True, "plan": plan}

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
        status = self.model_scout.status()
        return {
            "ok": True,
            "enabled": bool(self.config.model_scout_enabled),
            "status": status,
        }

    def model_scout_suggestions(self) -> dict[str, Any]:
        suggestions = self.model_scout.list_suggestions(limit=200)
        return {
            "ok": True,
            "suggestions": suggestions,
        }

    def run_model_scout(self) -> tuple[bool, dict[str, Any]]:
        result = self.model_scout.run(
            registry_document=self.registry_document,
            router_snapshot=self._router.doctor_snapshot(),
            usage_stats_snapshot=self._router.usage_stats_snapshot(),
            notify_sender=None,
        )
        self._log_request(
            "/model_scout/run",
            bool(result.get("ok")),
            {
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
                "suggestions": len(result.get("suggestions") or []),
                "new": len(result.get("new_suggestions") or []),
            },
        )
        return bool(result.get("ok")), {"ok": bool(result.get("ok")), **result}

    def dismiss_model_scout_suggestion(self, suggestion_id: str) -> tuple[bool, dict[str, Any]]:
        target = urllib.parse.unquote(str(suggestion_id or "")).strip()
        if not target:
            return False, {"ok": False, "error": "suggestion id is required"}
        if not self.model_scout.dismiss(target):
            return False, {"ok": False, "error": "suggestion not found"}
        return True, {"ok": True, "id": target, "status": "dismissed"}

    def mark_model_scout_installed(self, suggestion_id: str) -> tuple[bool, dict[str, Any]]:
        target = urllib.parse.unquote(str(suggestion_id or "")).strip()
        if not target:
            return False, {"ok": False, "error": "suggestion id is required"}
        if not self.model_scout.mark_installed(target):
            return False, {"ok": False, "error": "suggestion not found"}
        return True, {"ok": True, "id": target, "status": "installed"}

    def run_model_watch_once(self, *, trigger: str = "manual") -> tuple[bool, dict[str, Any]]:
        now = int(time.time())
        interval_seconds = max(1, int(self.config.model_watch_interval_seconds))
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
                    log_event(
                        self.config.log_path,
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
            log_event(
                self.config.log_path,
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

        body = {
            "ok": bool(result.get("ok")),
            **result,
            "trigger": trigger,
            "next_check_after_seconds": interval_seconds,
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
            },
        )
        log_event(
            self.config.log_path,
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
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                return parsed
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return {}

    def _handle_internal_error(self, method: str, exc: Exception) -> None:
        print(
            f"api_server internal_error method={method} path={getattr(self, 'path', '')} "
            f"error={exc.__class__.__name__}",
            file=sys.stderr,
            flush=True,
        )
        try:
            self._send_json(500, {"ok": False, "error": "internal_error"})
        except OSError:
            pass

    def _path_parts(self) -> tuple[str, list[str]]:
        parsed = urllib.parse.urlparse(self.path)
        clean_path = parsed.path or "/"
        parts = [part for part in clean_path.split("/") if part]
        return clean_path, parts

    def do_OPTIONS(self) -> None:  # noqa: N802
        try:
            self._send_json(200, {"ok": True})
        except Exception as exc:
            self._handle_internal_error("OPTIONS", exc)

    def do_GET(self) -> None:  # noqa: N802
        try:
            path, parts = self._path_parts()
            if path == "/health":
                self._send_json(200, self.runtime.health())
                return
            if path == "/version":
                self._send_json(200, self.runtime.version_info())
                return
            if path == "/telegram/status":
                self._send_json(200, self.runtime.telegram_status())
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
            if path == "/llm/health":
                self._send_json(200, self.runtime.llm_health_summary())
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
            payload = self._read_json()

            if path == "/chat":
                ok, body = self.runtime.chat(payload)
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
            if path == "/model_watch/run":
                ok, body = self.runtime.run_model_watch_once(trigger="manual")
                self._send_json(200 if ok else 400, body)
                return
            if path == "/model_watch/refresh":
                ok, body = self.runtime.model_watch_refresh(payload)
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
            if path == "/llm/registry/rollback":
                ok, body = self.runtime.llm_registry_rollback(payload)
                self._send_json(200 if ok else 400, body)
                return
            if path == "/llm/autopilot/undo":
                ok, body = self.runtime.llm_autopilot_undo(payload)
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


def build_runtime(config: Config | None = None) -> AgentRuntime:
    loaded = config or load_config(require_telegram_token=False)
    return AgentRuntime(loaded)


def run_server(host: str, port: int) -> None:
    runtime = build_runtime()
    runtime.set_listening(host, port)

    class _Handler(APIServerHandler):
        pass

    _Handler.runtime = runtime

    try:
        server = ThreadingHTTPServer((host, port), _Handler)
    except OSError as exc:
        print(
            f"Failed to bind Personal Agent API on {runtime.listening_url}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc

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

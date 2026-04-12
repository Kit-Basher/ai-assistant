from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Callable

from agent.error_response_ux import compose_actionable_message
from agent.llm.model_inventory import build_model_inventory
from agent.model_watch_hf import deterministic_ollama_model_name, hf_snapshot_download


MODEL_LIFECYCLE_STATES = (
    "not_installed",
    "queued",
    "downloading",
    "installed",
    "installed_not_ready",
    "ready",
    "failed",
)


InventoryBuilder = Callable[..., list[dict[str, Any]]]
InstallPlannerFn = Callable[..., dict[str, Any]]
InstallExecutorFn = Callable[..., dict[str, Any]]
SnapshotDownloadFn = Callable[..., str]
SubprocessRunFn = Callable[..., subprocess.CompletedProcess[str]]


def default_model_manager_state_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "targets": {},
    }


def model_manager_state_path_for_runtime(runtime: Any) -> Path:
    runtime_path = str(getattr(runtime, "_model_manager_state_path", "") or "").strip()
    if runtime_path:
        return Path(runtime_path).expanduser().resolve()
    explicit = os.getenv("AGENT_MODEL_MANAGER_STATE_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    runtime_state_path = getattr(runtime, "_runtime_state_path", None)
    config = getattr(runtime, "config", None)
    if callable(runtime_state_path) and config is not None:
        candidate = runtime_state_path(config, None, "model_manager_state.json")
        if candidate:
            return Path(candidate).expanduser().resolve()
    return (Path.home() / ".local" / "share" / "personal-agent" / "model_manager_state.json").resolve()


def _normalize_state_row(row: dict[str, Any]) -> dict[str, Any] | None:
    payload = row if isinstance(row, dict) else {}
    target_key = str(payload.get("target_key") or "").strip()
    if not target_key:
        return None
    state = str(payload.get("state") or "").strip().lower() or "not_installed"
    if state not in MODEL_LIFECYCLE_STATES:
        state = "not_installed"
    try:
        updated_ts = max(0, int(payload.get("updated_ts") or 0))
    except (TypeError, ValueError):
        updated_ts = 0
    try:
        last_success_ts = max(0, int(payload.get("last_success_ts") or 0)) or None
    except (TypeError, ValueError):
        last_success_ts = None
    try:
        last_failure_ts = max(0, int(payload.get("last_failure_ts") or 0)) or None
    except (TypeError, ValueError):
        last_failure_ts = None
    return {
        "target_key": target_key,
        "target_type": str(payload.get("target_type") or "model").strip().lower() or "model",
        "provider_id": str(payload.get("provider_id") or "").strip().lower() or None,
        "model_id": str(payload.get("model_id") or "").strip() or None,
        "artifact_id": str(payload.get("artifact_id") or "").strip() or None,
        "repo_id": str(payload.get("repo_id") or "").strip() or None,
        "revision": str(payload.get("revision") or "").strip() or None,
        "state": state,
        "message": str(payload.get("message") or "").strip() or None,
        "error_kind": str(payload.get("error_kind") or "").strip() or None,
        "source": str(payload.get("source") or "").strip() or None,
        "updated_ts": updated_ts,
        "last_success_ts": last_success_ts,
        "last_failure_ts": last_failure_ts,
    }


def load_model_manager_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return default_model_manager_state_document()
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default_model_manager_state_document()
    if not isinstance(parsed, dict):
        return default_model_manager_state_document()
    targets_raw = parsed.get("targets") if isinstance(parsed.get("targets"), dict) else {}
    targets: dict[str, dict[str, Any]] = {}
    for key, value in sorted(targets_raw.items()):
        normalized = _normalize_state_row({"target_key": str(key or "").strip(), **(value if isinstance(value, dict) else {})})
        if normalized is None:
            continue
        targets[normalized["target_key"]] = normalized
    return {
        "schema_version": 1,
        "targets": targets,
    }


def save_model_manager_state(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    normalized = load_model_manager_state(path) if not isinstance(state, dict) else {
        "schema_version": 1,
        "targets": {
            key: value
            for key, value in (
                (
                    str(key or "").strip(),
                    _normalize_state_row({"target_key": str(key or "").strip(), **(value if isinstance(value, dict) else {})}),
                )
                for key, value in (
                    (state.get("targets") if isinstance(state.get("targets"), dict) else {}).items()
                )
            )
            if key and isinstance(value, dict)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
    return normalized


def _canonical_ollama_model_id(model_ref: str | None) -> str | None:
    normalized = str(model_ref or "").strip()
    if not normalized:
        return None
    if normalized.startswith("ollama:"):
        return normalized
    return f"ollama:{normalized}"


def _hf_artifact_id(repo_id: str, revision: str | None) -> str:
    normalized_repo = str(repo_id or "").strip()
    normalized_revision = str(revision or "").strip() or "latest"
    return f"hf:{normalized_repo}@{normalized_revision}"


def _tail(text: str | None, *, limit: int = 400) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[-limit:]


def _verification_row(inventory: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    row = next(
        (
            item
            for item in inventory
            if isinstance(item, dict) and str(item.get("id") or "").strip() == str(model_id or "").strip()
        ),
        None,
    )
    if not isinstance(row, dict):
        return {
            "found": False,
            "installed": False,
            "available": False,
            "healthy": False,
            "verification_status": "degraded",
        }
    installed = bool(row.get("installed", False))
    available = bool(row.get("available", False))
    healthy = bool(row.get("healthy", False))
    return {
        "found": True,
        "installed": installed,
        "available": available,
        "healthy": healthy,
        "reason": str(row.get("reason") or "").strip() or None,
        "health_reason": str(row.get("health_reason") or row.get("health_failure_kind") or row.get("reason") or "").strip() or None,
        "capability_source": str(row.get("capability_source") or "").strip() or None,
        "verification_status": "ok" if (installed and available and healthy) else "degraded",
        "provider": str(row.get("provider") or "").strip() or None,
        "capabilities": list(row.get("capabilities") or []) if isinstance(row.get("capabilities"), list) else [],
    }


def build_model_manager_request_from_hf_plan_rows(plan_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [dict(row) for row in plan_rows if isinstance(row, dict)]
    if not rows:
        return None
    actions = {str(row.get("action") or "").strip() for row in rows}
    if not actions or not all(action.startswith("hf.") for action in actions):
        return None
    download_step = next((row for row in rows if str(row.get("action") or "").strip() == "hf.snapshot_download"), None)
    if not isinstance(download_step, dict):
        return None
    download_params = download_step.get("params") if isinstance(download_step.get("params"), dict) else {}
    repo_id = str(download_params.get("repo_id") or "").strip()
    target_dir = str(download_params.get("target_dir") or "").strip()
    if not repo_id or not target_dir:
        return None
    revision = str(download_params.get("revision") or "").strip() or "main"
    allow_patterns = (
        [str(item).strip() for item in download_params.get("allow_patterns", []) if str(item).strip()]
        if isinstance(download_params.get("allow_patterns"), list)
        else []
    )
    generate_step = next((row for row in rows if str(row.get("action") or "").strip() == "hf.generate_modelfile"), None)
    create_step = next((row for row in rows if str(row.get("action") or "").strip() == "hf.ollama_create"), None)
    mark_only_step = next((row for row in rows if str(row.get("action") or "").strip() == "hf.mark_download_only"), None)
    generate_params = generate_step.get("params") if isinstance(generate_step, dict) and isinstance(generate_step.get("params"), dict) else {}
    create_params = create_step.get("params") if isinstance(create_step, dict) and isinstance(create_step.get("params"), dict) else {}
    selected_gguf = str(generate_params.get("selected_gguf") or "").strip() or None
    modelfile_path = str(
        create_params.get("modelfile_path")
        or generate_params.get("modelfile_path")
        or ""
    ).strip() or None
    ollama_model_name = str(
        create_params.get("ollama_model_name")
        or generate_params.get("ollama_model_name")
        or ""
    ).strip() or None
    download_only = isinstance(mark_only_step, dict) and not ollama_model_name
    if not download_only and not ollama_model_name:
        ollama_model_name = deterministic_ollama_model_name(
            repo_id=repo_id,
            selected_gguf=selected_gguf,
            revision=revision,
        )
    return {
        "kind": "hf_local_download",
        "repo_id": repo_id,
        "revision": revision,
        "target_dir": target_dir,
        "allow_patterns": allow_patterns,
        "selected_gguf": selected_gguf,
        "modelfile_path": modelfile_path,
        "ollama_model_name": ollama_model_name,
        "download_only": download_only,
    }


def build_model_lifecycle_rows(
    *,
    inventory_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]] | None,
    manager_state: dict[str, Any],
) -> list[dict[str, Any]]:
    readiness_lookup = {
        str(row.get("model_id") or "").strip(): dict(row)
        for row in (readiness_rows or [])
        if isinstance(row, dict) and str(row.get("model_id") or "").strip()
    }
    state_targets = manager_state.get("targets") if isinstance(manager_state.get("targets"), dict) else {}
    state_by_model_id = {
        str(row.get("model_id") or "").strip(): dict(row)
        for row in state_targets.values()
        if isinstance(row, dict) and str(row.get("model_id") or "").strip()
    }
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for inventory_row in [dict(row) for row in inventory_rows if isinstance(row, dict)]:
        model_id = str(inventory_row.get("model_id") or "").strip()
        if not model_id:
            continue
        readiness_row = readiness_lookup.get(model_id, {})
        state_row = state_by_model_id.get(model_id, {})
        transient_state = str(state_row.get("state") or "").strip().lower()
        installed_local = bool(
            inventory_row.get("installed_local", inventory_row.get("installed", False))
        )
        usable_now = bool(readiness_row.get("usable_now", False))
        if usable_now:
            lifecycle_state = "ready"
        elif transient_state in {"queued", "downloading"}:
            lifecycle_state = transient_state
        elif installed_local and readiness_row:
            lifecycle_state = "installed_not_ready"
        elif installed_local:
            lifecycle_state = "installed"
        elif transient_state == "failed":
            lifecycle_state = "failed"
        else:
            lifecycle_state = "not_installed"
        rows.append(
            {
                "target_key": model_id,
                "target_type": "model",
                "provider_id": str(inventory_row.get("provider_id") or "").strip().lower() or None,
                "model_id": model_id,
                "artifact_id": None,
                "repo_id": state_row.get("repo_id"),
                "revision": state_row.get("revision"),
                "lifecycle_state": lifecycle_state,
                "message": state_row.get("message"),
                "error_kind": state_row.get("error_kind"),
                "updated_ts": state_row.get("updated_ts"),
                "last_success_ts": state_row.get("last_success_ts"),
                "last_failure_ts": state_row.get("last_failure_ts"),
                "installed": installed_local,
                "ready": usable_now,
                "availability_state": str(readiness_row.get("availability_state") or "").strip() or None,
            }
        )
        seen_keys.add(model_id)

    for state_row in [dict(row) for row in state_targets.values() if isinstance(row, dict)]:
        target_key = str(state_row.get("target_key") or "").strip()
        if not target_key or target_key in seen_keys:
            continue
        rows.append(
            {
                "target_key": target_key,
                "target_type": str(state_row.get("target_type") or "model").strip().lower() or "model",
                "provider_id": str(state_row.get("provider_id") or "").strip().lower() or None,
                "model_id": str(state_row.get("model_id") or "").strip() or None,
                "artifact_id": str(state_row.get("artifact_id") or "").strip() or None,
                "repo_id": str(state_row.get("repo_id") or "").strip() or None,
                "revision": str(state_row.get("revision") or "").strip() or None,
                "lifecycle_state": str(state_row.get("state") or "not_installed").strip().lower() or "not_installed",
                "message": state_row.get("message"),
                "error_kind": state_row.get("error_kind"),
                "updated_ts": state_row.get("updated_ts"),
                "last_success_ts": state_row.get("last_success_ts"),
                "last_failure_ts": state_row.get("last_failure_ts"),
                "installed": False,
                "ready": False,
                "availability_state": None,
            }
        )
    rows.sort(
        key=lambda row: (
            0 if str(row.get("lifecycle_state") or "") in {"downloading", "queued"} else 1,
            0 if bool(row.get("ready", False)) else 1,
            str(row.get("model_id") or row.get("artifact_id") or row.get("target_key") or ""),
        )
    )
    return rows


class CanonicalModelManager:
    def __init__(
        self,
        runtime: Any,
        *,
        install_planner_fn: InstallPlannerFn,
        install_executor_fn: InstallExecutorFn,
        inventory_builder: InventoryBuilder = build_model_inventory,
        hf_snapshot_download_fn: SnapshotDownloadFn = hf_snapshot_download,
        subprocess_run_fn: SubprocessRunFn = subprocess.run,
    ) -> None:
        self.runtime = runtime
        self._install_planner_fn = install_planner_fn
        self._install_executor_fn = install_executor_fn
        self._inventory_builder = inventory_builder
        self._hf_snapshot_download_fn = hf_snapshot_download_fn
        self._subprocess_run_fn = subprocess_run_fn
        self._state_path = model_manager_state_path_for_runtime(runtime)

    def _load_state(self) -> dict[str, Any]:
        return load_model_manager_state(self._state_path)

    def _save_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return save_model_manager_state(self._state_path, state)

    def _update_target_state(
        self,
        *,
        target_key: str,
        target_type: str,
        provider_id: str | None,
        model_id: str | None,
        artifact_id: str | None,
        repo_id: str | None,
        revision: str | None,
        state: str,
        message: str | None,
        error_kind: str | None,
        source: str | None,
    ) -> dict[str, Any]:
        current = self._load_state()
        targets = current.get("targets") if isinstance(current.get("targets"), dict) else {}
        existing = targets.get(target_key) if isinstance(targets.get(target_key), dict) else {}
        now_epoch = int(time.time())
        row = {
            "target_key": target_key,
            "target_type": target_type,
            "provider_id": provider_id,
            "model_id": model_id,
            "artifact_id": artifact_id,
            "repo_id": repo_id,
            "revision": revision,
            "state": state,
            "message": message,
            "error_kind": error_kind,
            "source": source,
            "updated_ts": now_epoch,
            "last_success_ts": existing.get("last_success_ts"),
            "last_failure_ts": existing.get("last_failure_ts"),
        }
        if state in {"installed", "installed_not_ready", "ready"}:
            row["last_success_ts"] = now_epoch
            row["last_failure_ts"] = existing.get("last_failure_ts")
        elif state == "failed":
            row["last_failure_ts"] = now_epoch
            row["last_success_ts"] = existing.get("last_success_ts")
        targets[target_key] = row
        current["targets"] = targets
        self._save_state(current)
        return row

    def _guard_install_allowed(self, *, source: str) -> tuple[bool, dict[str, Any] | None]:
        policy_reader = getattr(self.runtime, "_chat_control_policy", None)
        policy: dict[str, Any] = {}
        if callable(policy_reader):
            try:
                policy_candidate = policy_reader()
            except Exception:
                policy_candidate = {}
            policy = dict(policy_candidate) if isinstance(policy_candidate, dict) else {}
        if not policy:
            truth_getter = getattr(self.runtime, "runtime_truth_service", None)
            if callable(truth_getter):
                try:
                    policy_obj = truth_getter()
                    policy_candidate = (
                        policy_obj.model_controller_policy_status()
                        if callable(getattr(policy_obj, "model_controller_policy_status", None))
                        else {}
                    )
                    policy = dict(policy_candidate) if isinstance(policy_candidate, dict) else {}
                except Exception:
                    policy = {}
        safe_mode = bool(policy.get("safe_mode", False))
        if safe_mode or policy.get("allow_install_pull") is False:
            why = "The current mode does not allow model installs or downloads."
            next_action = "Stay on the current model, or switch to Controlled Mode explicitly before retrying."
            return False, {
                "ok": False,
                "executed": False,
                "error_kind": "safe_mode_blocked",
                "message": compose_actionable_message(
                    what_happened="I did not start downloading or installing the model.",
                    why=why,
                    next_action=next_action,
                ),
                "why": why,
                "next_action": next_action,
                "source": source,
            }
        return True, None

    def _inventory(self) -> list[dict[str, Any]]:
        return self._inventory_builder(
            config=self.runtime.config,
            registry=self.runtime._router.registry,
            router_snapshot=self.runtime._router.doctor_snapshot(),
        )

    def _refresh_ollama(self) -> tuple[bool, dict[str, Any]]:
        refresher = getattr(self.runtime, "refresh_models", None)
        if not callable(refresher):
            return False, {"ok": False, "error": "refresh_unavailable"}
        ok, body = refresher({"provider": "ollama"})
        return bool(ok), dict(body) if isinstance(body, dict) else {"ok": bool(ok)}

    @staticmethod
    def _result_state_from_verification(verification: dict[str, Any]) -> str:
        if bool(verification.get("found", False)) and bool(verification.get("installed", False)):
            if bool(verification.get("available", False)) and bool(verification.get("healthy", False)):
                return "ready"
            return "installed_not_ready"
        return "installed"

    def execute_request(
        self,
        request: dict[str, Any],
        *,
        approve: bool,
        trace_id: str | None = None,
        timeout_seconds: float | None = None,
        source: str,
    ) -> dict[str, Any]:
        kind = str(request.get("kind") or "").strip().lower()
        if kind == "approved_ollama_pull":
            return self._execute_approved_ollama_pull(
                request=request,
                approve=approve,
                trace_id=trace_id,
                timeout_seconds=timeout_seconds,
                source=source,
            )
        if kind == "hf_local_download":
            return self._execute_hf_local_download(
                request=request,
                approve=approve,
                trace_id=trace_id,
                source=source,
            )
        if kind == "ollama_import_gguf":
            return self._execute_ollama_import_gguf(
                request=request,
                approve=approve,
                trace_id=trace_id,
                source=source,
            )
        return {
            "ok": False,
            "executed": False,
            "error_kind": "unsupported_install_request",
            "message": "Unsupported model manager request.",
            "source": source,
        }

    def _execute_approved_ollama_pull(
        self,
        *,
        request: dict[str, Any],
        approve: bool,
        trace_id: str | None,
        timeout_seconds: float | None,
        source: str,
    ) -> dict[str, Any]:
        model_ref = _canonical_ollama_model_id(str(request.get("model_ref") or request.get("model") or "").strip())
        if not model_ref:
            return {
                "ok": False,
                "executed": False,
                "error_kind": "model_required",
                "message": "model is required",
                "source": source,
            }
        plan = self._install_planner_fn(inventory=self._inventory(), model_ref=model_ref)
        candidate = (
            plan.get("candidates")[0]
            if isinstance(plan.get("candidates"), list) and plan.get("candidates")
            and isinstance(plan.get("candidates")[0], dict)
            else {}
        )
        model_id = str((candidate or {}).get("model_id") or model_ref).strip() or model_ref
        install_name = str((candidate or {}).get("install_name") or "").strip() or None
        allow, blocked = self._guard_install_allowed(source=source)
        if not allow:
            response = {
                **dict(blocked or {}),
                "model_id": model_id,
                "install_name": install_name,
                "install_plan": plan,
                "trace_id": str(trace_id or ""),
                "verification": {},
            }
            return response
        if not approve and bool(plan.get("needed", False)) and bool(plan.get("approved", False)):
            self._update_target_state(
                target_key=model_id,
                target_type="model",
                provider_id="ollama",
                model_id=model_id,
                artifact_id=None,
                repo_id=None,
                revision=None,
                state="queued",
                message="Awaiting explicit approval for local install.",
                error_kind=None,
                source=source,
            )
        if approve:
            self._update_target_state(
                target_key=model_id,
                target_type="model",
                provider_id="ollama",
                model_id=model_id,
                artifact_id=None,
                repo_id=None,
                revision=None,
                state="downloading",
                message=f"Installing {model_id}.",
                error_kind=None,
                source=source,
            )
        result = self._install_executor_fn(
            config=self.runtime.config,
            registry=self.runtime._router.registry,
            plan=plan,
            approve=approve,
            trace_id=trace_id,
            timeout_seconds=float(timeout_seconds or 1800.0),
        )
        payload = dict(result) if isinstance(result, dict) else {"ok": False, "executed": False}
        payload["install_plan"] = plan
        verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
        if bool(payload.get("ok", False)):
            refresh_ok, refresh_body = self._refresh_ollama()
            payload["refresh_ok"] = bool(refresh_ok)
            payload["refresh_result"] = refresh_body
            verification = _verification_row(self._inventory(), model_id)
            payload["verification"] = verification
            lifecycle_state = self._result_state_from_verification(verification)
            self._update_target_state(
                target_key=model_id,
                target_type="model",
                provider_id="ollama",
                model_id=model_id,
                artifact_id=None,
                repo_id=None,
                revision=None,
                state=lifecycle_state,
                message=str(payload.get("message") or "").strip() or None,
                error_kind=None,
                source=source,
            )
        else:
            error_kind = str(payload.get("error_kind") or "install_failed").strip() or "install_failed"
            if error_kind == "approval_required":
                lifecycle_state = "queued"
            else:
                lifecycle_state = "failed"
            self._update_target_state(
                target_key=model_id,
                target_type="model",
                provider_id="ollama",
                model_id=model_id,
                artifact_id=None,
                repo_id=None,
                revision=None,
                state=lifecycle_state,
                message=str(payload.get("message") or "").strip() or None,
                error_kind=None if lifecycle_state == "queued" else error_kind,
                source=source,
            )
        return payload

    def _execute_hf_local_download(
        self,
        *,
        request: dict[str, Any],
        approve: bool,
        trace_id: str | None,
        source: str,
    ) -> dict[str, Any]:
        repo_id = str(request.get("repo_id") or "").strip()
        revision = str(request.get("revision") or "").strip() or "main"
        target_dir = str(request.get("target_dir") or "").strip()
        allow_patterns = (
            [str(item).strip() for item in request.get("allow_patterns", []) if str(item).strip()]
            if isinstance(request.get("allow_patterns"), list)
            else []
        )
        selected_gguf = str(request.get("selected_gguf") or "").strip() or None
        modelfile_path = str(request.get("modelfile_path") or "").strip() or None
        ollama_model_name = str(request.get("ollama_model_name") or "").strip() or None
        download_only = bool(request.get("download_only", False))
        if not repo_id or not target_dir:
            return {
                "ok": False,
                "executed": False,
                "error_kind": "repo_id_and_target_dir_required",
                "message": "repo_id and target_dir are required",
                "source": source,
            }
        model_id = _canonical_ollama_model_id(ollama_model_name) if ollama_model_name else None
        artifact_id = _hf_artifact_id(repo_id, revision)
        target_key = model_id or artifact_id
        allow, blocked = self._guard_install_allowed(source=source)
        if not allow:
            return {
                **dict(blocked or {}),
                "trace_id": str(trace_id or ""),
                "model_id": model_id,
                "artifact_id": artifact_id,
                "verification": {},
            }
        if not approve:
            self._update_target_state(
                target_key=target_key,
                target_type="model" if model_id else "artifact",
                provider_id="ollama" if model_id else "huggingface",
                model_id=model_id,
                artifact_id=None if model_id else artifact_id,
                repo_id=repo_id,
                revision=revision,
                state="queued",
                message="Awaiting explicit approval for model download.",
                error_kind=None,
                source=source,
            )
            return {
                "ok": False,
                "executed": False,
                "error_kind": "approval_required",
                "message": "Explicit approval is required before executing this download.",
                "trace_id": str(trace_id or ""),
                "model_id": model_id,
                "artifact_id": artifact_id,
                "verification": {},
                "source": source,
            }
        self._update_target_state(
            target_key=target_key,
            target_type="model" if model_id else "artifact",
            provider_id="ollama" if model_id else "huggingface",
            model_id=model_id,
            artifact_id=None if model_id else artifact_id,
            repo_id=repo_id,
            revision=revision,
            state="downloading",
            message=f"Downloading {repo_id}.",
            error_kind=None,
            source=source,
        )
        try:
            downloaded_path = self._hf_snapshot_download_fn(
                repo_id=repo_id,
                revision=revision,
                target_dir=target_dir,
                allow_patterns=allow_patterns,
            )
        except Exception as exc:
            error_kind = f"snapshot_download_failed:{exc.__class__.__name__}"
            self._update_target_state(
                target_key=target_key,
                target_type="model" if model_id else "artifact",
                provider_id="ollama" if model_id else "huggingface",
                model_id=model_id,
                artifact_id=None if model_id else artifact_id,
                repo_id=repo_id,
                revision=revision,
                state="failed",
                message="Snapshot download failed.",
                error_kind=error_kind,
                source=source,
            )
            return {
                "ok": False,
                "executed": True,
                "error_kind": error_kind,
                "message": "Snapshot download failed.",
                "trace_id": str(trace_id or ""),
                "model_id": model_id,
                "artifact_id": artifact_id,
                "verification": {},
                "source": source,
            }

        verification: dict[str, Any] = {}
        refresh_ok = False
        refresh_result: dict[str, Any] | None = None
        if not download_only and model_id and selected_gguf:
            modelfile_target = Path(
                modelfile_path
                or str((Path(target_dir).expanduser().resolve() / "Modelfile.personal-agent").resolve())
            ).expanduser().resolve()
            gguf_path = (Path(target_dir).expanduser().resolve() / selected_gguf).resolve()
            if not gguf_path.is_file():
                error_kind = "selected_gguf_missing"
                self._update_target_state(
                    target_key=target_key,
                    target_type="model",
                    provider_id="ollama",
                    model_id=model_id,
                    artifact_id=None,
                    repo_id=repo_id,
                    revision=revision,
                    state="failed",
                    message="Selected GGUF file is missing after download.",
                    error_kind=error_kind,
                    source=source,
                )
                return {
                    "ok": False,
                    "executed": True,
                    "error_kind": error_kind,
                    "message": "Selected GGUF file is missing after download.",
                    "trace_id": str(trace_id or ""),
                    "model_id": model_id,
                    "artifact_id": None,
                    "verification": {},
                    "source": source,
                }
            modelfile_target.parent.mkdir(parents=True, exist_ok=True)
            fd = -1
            tmp_path = ""
            try:
                fd, tmp_path = tempfile.mkstemp(
                    prefix=f".{modelfile_target.name}.",
                    suffix=".tmp",
                    dir=str(modelfile_target.parent),
                )
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(f"FROM {str(gguf_path)}\n")
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
                error_kind = f"modelfile_write_failed:{exc.__class__.__name__}"
                self._update_target_state(
                    target_key=target_key,
                    target_type="model",
                    provider_id="ollama",
                    model_id=model_id,
                    artifact_id=None,
                    repo_id=repo_id,
                    revision=revision,
                    state="failed",
                    message="Modelfile generation failed.",
                    error_kind=error_kind,
                    source=source,
                )
                return {
                    "ok": False,
                    "executed": True,
                    "error_kind": error_kind,
                    "message": "Modelfile generation failed.",
                    "trace_id": str(trace_id or ""),
                    "model_id": model_id,
                    "artifact_id": None,
                    "verification": {},
                    "source": source,
                }

            try:
                completed = self._subprocess_run_fn(
                    ["ollama", "create", str(model_id.split(":", 1)[1]), "-f", str(modelfile_target)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
            except Exception as exc:
                error_kind = f"ollama_create_failed:{exc.__class__.__name__}"
                self._update_target_state(
                    target_key=target_key,
                    target_type="model",
                    provider_id="ollama",
                    model_id=model_id,
                    artifact_id=None,
                    repo_id=repo_id,
                    revision=revision,
                    state="failed",
                    message="Ollama create failed.",
                    error_kind=error_kind,
                    source=source,
                )
                return {
                    "ok": False,
                    "executed": True,
                    "error_kind": error_kind,
                    "message": "Ollama create failed.",
                    "trace_id": str(trace_id or ""),
                    "model_id": model_id,
                    "artifact_id": None,
                    "verification": {},
                    "source": source,
                }
            if int(completed.returncode) != 0:
                error_kind = "ollama_create_failed"
                detail = _tail(str(completed.stderr or "").strip() or "ollama_create_failed", limit=240)
                self._update_target_state(
                    target_key=target_key,
                    target_type="model",
                    provider_id="ollama",
                    model_id=model_id,
                    artifact_id=None,
                    repo_id=repo_id,
                    revision=revision,
                    state="failed",
                    message="Ollama create failed.",
                    error_kind=error_kind,
                    source=source,
                )
                return {
                    "ok": False,
                    "executed": True,
                    "error_kind": error_kind,
                    "message": detail,
                    "trace_id": str(trace_id or ""),
                    "model_id": model_id,
                    "artifact_id": None,
                    "verification": {},
                    "stdout_tail": _tail(completed.stdout),
                    "stderr_tail": _tail(completed.stderr),
                    "source": source,
                }
            refresh_ok, refresh_result = self._refresh_ollama()
            verification = _verification_row(self._inventory(), model_id)
            lifecycle_state = self._result_state_from_verification(verification)
            self._update_target_state(
                target_key=target_key,
                target_type="model",
                provider_id="ollama",
                model_id=model_id,
                artifact_id=None,
                repo_id=repo_id,
                revision=revision,
                state=lifecycle_state,
                message="Downloaded and imported model into Ollama.",
                error_kind=None,
                source=source,
            )
            return {
                "ok": True,
                "executed": True,
                "error_kind": None,
                "message": "Downloaded and imported model into Ollama.",
                "trace_id": str(trace_id or ""),
                "model_id": model_id,
                "artifact_id": None,
                "download_path": str(downloaded_path),
                "modelfile_path": str(modelfile_target),
                "refresh_ok": bool(refresh_ok),
                "refresh_result": refresh_result,
                "verification": verification,
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
                "source": source,
            }

        marker_path = (Path(target_dir).expanduser().resolve() / ".personal-agent-download-only.json").resolve()
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_payload = {
            "repo_id": repo_id,
            "revision": revision,
            "status": "download_only",
        }
        try:
            marker_path.write_text(json.dumps(marker_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as exc:
            error_kind = f"download_marker_failed:{exc.__class__.__name__}"
            self._update_target_state(
                target_key=target_key,
                target_type="artifact",
                provider_id="huggingface",
                model_id=None,
                artifact_id=artifact_id,
                repo_id=repo_id,
                revision=revision,
                state="failed",
                message="Download marker write failed.",
                error_kind=error_kind,
                source=source,
            )
            return {
                "ok": False,
                "executed": True,
                "error_kind": error_kind,
                "message": "Download marker write failed.",
                "trace_id": str(trace_id or ""),
                "model_id": None,
                "artifact_id": artifact_id,
                "verification": {},
                "source": source,
            }
        self._update_target_state(
            target_key=target_key,
            target_type="artifact",
            provider_id="huggingface",
            model_id=None,
            artifact_id=artifact_id,
            repo_id=repo_id,
            revision=revision,
            state="installed",
            message="Downloaded snapshot for offline inspection.",
            error_kind=None,
            source=source,
        )
        return {
            "ok": True,
            "executed": True,
            "error_kind": None,
            "message": "Downloaded snapshot for offline inspection.",
            "trace_id": str(trace_id or ""),
            "model_id": None,
            "artifact_id": artifact_id,
            "download_path": str(downloaded_path),
            "verification": {},
            "source": source,
        }

    def _execute_ollama_import_gguf(
        self,
        *,
        request: dict[str, Any],
        approve: bool,
        trace_id: str | None,
        source: str,
    ) -> dict[str, Any]:
        model_name = str(request.get("model_name") or "").strip()
        modelfile_path = str(request.get("modelfile_path") or "").strip()
        model_id = _canonical_ollama_model_id(model_name)
        if not model_id or not modelfile_path:
            return {
                "ok": False,
                "executed": False,
                "error_kind": "model_name_and_modelfile_path_required",
                "message": "model_name and modelfile_path are required",
                "source": source,
            }
        allow, blocked = self._guard_install_allowed(source=source)
        if not allow:
            return {
                **dict(blocked or {}),
                "trace_id": str(trace_id or ""),
                "model_id": model_id,
                "verification": {},
            }
        if not approve:
            self._update_target_state(
                target_key=model_id,
                target_type="model",
                provider_id="ollama",
                model_id=model_id,
                artifact_id=None,
                repo_id=None,
                revision=None,
                state="queued",
                message="Awaiting explicit approval for model import.",
                error_kind=None,
                source=source,
            )
            return {
                "ok": False,
                "executed": False,
                "error_kind": "approval_required",
                "message": "Explicit approval is required before importing this model.",
                "trace_id": str(trace_id or ""),
                "model_id": model_id,
                "verification": {},
                "source": source,
            }
        self._update_target_state(
            target_key=model_id,
            target_type="model",
            provider_id="ollama",
            model_id=model_id,
            artifact_id=None,
            repo_id=None,
            revision=None,
            state="downloading",
            message=f"Importing {model_id} into Ollama.",
            error_kind=None,
            source=source,
        )
        try:
            completed = self._subprocess_run_fn(
                ["ollama", "create", str(model_id.split(":", 1)[1]), "-f", modelfile_path],
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
            )
        except Exception as exc:
            error_kind = f"ollama_create_failed:{exc.__class__.__name__}"
            self._update_target_state(
                target_key=model_id,
                target_type="model",
                provider_id="ollama",
                model_id=model_id,
                artifact_id=None,
                repo_id=None,
                revision=None,
                state="failed",
                message="Ollama create failed.",
                error_kind=error_kind,
                source=source,
            )
            return {
                "ok": False,
                "executed": True,
                "error_kind": error_kind,
                "message": "Ollama create failed.",
                "trace_id": str(trace_id or ""),
                "model_id": model_id,
                "verification": {},
                "source": source,
            }
        if int(completed.returncode) != 0:
            error_kind = "ollama_create_failed"
            detail = _tail(str(completed.stderr or "").strip() or "ollama_create_failed", limit=240)
            self._update_target_state(
                target_key=model_id,
                target_type="model",
                provider_id="ollama",
                model_id=model_id,
                artifact_id=None,
                repo_id=None,
                revision=None,
                state="failed",
                message="Ollama create failed.",
                error_kind=error_kind,
                source=source,
            )
            return {
                "ok": False,
                "executed": True,
                "error_kind": error_kind,
                "message": detail,
                "trace_id": str(trace_id or ""),
                "model_id": model_id,
                "verification": {},
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
                "source": source,
            }
        refresh_ok, refresh_result = self._refresh_ollama()
        verification = _verification_row(self._inventory(), model_id)
        lifecycle_state = self._result_state_from_verification(verification)
        self._update_target_state(
            target_key=model_id,
            target_type="model",
            provider_id="ollama",
            model_id=model_id,
            artifact_id=None,
            repo_id=None,
            revision=None,
            state=lifecycle_state,
            message="Imported model into Ollama.",
            error_kind=None,
            source=source,
        )
        return {
            "ok": True,
            "executed": True,
            "error_kind": None,
            "message": "Imported model into Ollama.",
            "trace_id": str(trace_id or ""),
            "model_id": model_id,
            "refresh_ok": bool(refresh_ok),
            "refresh_result": refresh_result,
            "verification": verification,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
            "source": source,
        }


__all__ = [
    "CanonicalModelManager",
    "MODEL_LIFECYCLE_STATES",
    "build_model_lifecycle_rows",
    "build_model_manager_request_from_hf_plan_rows",
    "default_model_manager_state_document",
    "load_model_manager_state",
    "model_manager_state_path_for_runtime",
    "save_model_manager_state",
]

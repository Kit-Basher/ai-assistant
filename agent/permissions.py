from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import tempfile
from typing import Any


MODEL_OPS_ACTIONS = (
    "modelops.install_ollama",
    "modelops.pull_ollama_model",
    "modelops.import_gguf_to_ollama",
    "modelops.set_default_model",
    "modelops.enable_disable_provider_or_model",
)

_PERMISSION_MODES = {"manual_confirm", "auto"}


def default_permissions_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "manual_confirm",
        "actions": {action: False for action in MODEL_OPS_ACTIONS},
        "constraints": {
            "max_download_bytes": 5 * 1024 * 1024 * 1024,
            "allow_install_ollama": False,
            "allow_remote_models": True,
            "allowed_providers": ["ollama", "openrouter"],
            "allowed_model_patterns": [],
        },
    }


def _normalize_document(document: dict[str, Any]) -> dict[str, Any]:
    merged = default_permissions_document()

    mode = str(document.get("mode") or merged["mode"]).strip().lower()
    merged["mode"] = mode if mode in _PERMISSION_MODES else merged["mode"]

    raw_actions = document.get("actions") if isinstance(document.get("actions"), dict) else {}
    actions = dict(merged["actions"])
    for action in MODEL_OPS_ACTIONS:
        if action in raw_actions:
            actions[action] = bool(raw_actions.get(action))
    merged["actions"] = actions

    raw_constraints = document.get("constraints") if isinstance(document.get("constraints"), dict) else {}
    constraints = dict(merged["constraints"])

    max_download_bytes = raw_constraints.get("max_download_bytes")
    max_download_gb = raw_constraints.get("max_download_gb")
    if max_download_bytes is None and max_download_gb is not None:
        try:
            max_download_bytes = int(float(max_download_gb) * 1024 * 1024 * 1024)
        except Exception:
            max_download_bytes = constraints["max_download_bytes"]
    try:
        constraints["max_download_bytes"] = max(0, int(max_download_bytes))
    except Exception:
        pass

    if "allow_install_ollama" in raw_constraints:
        constraints["allow_install_ollama"] = bool(raw_constraints.get("allow_install_ollama"))
    if "allow_remote_models" in raw_constraints:
        constraints["allow_remote_models"] = bool(raw_constraints.get("allow_remote_models"))

    allowed_providers = raw_constraints.get("allowed_providers")
    if isinstance(allowed_providers, list):
        normalized = sorted({str(item).strip().lower() for item in allowed_providers if str(item).strip()})
        constraints["allowed_providers"] = normalized

    patterns = raw_constraints.get("allowed_model_patterns")
    if isinstance(patterns, list):
        constraints["allowed_model_patterns"] = [str(item).strip() for item in patterns if str(item).strip()]

    constraints["max_download_gb"] = round(float(constraints["max_download_bytes"]) / (1024 * 1024 * 1024), 2)
    merged["constraints"] = constraints

    return merged


@dataclass(frozen=True)
class PermissionRequest:
    action: str
    params: dict[str, Any]
    estimated_cost: float | None = None
    estimated_bytes: int | None = None
    risk_level: str | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class PermissionDecision:
    allow: bool
    reason: str
    requires_confirmation: bool


class PermissionStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()

    @staticmethod
    def default_path() -> str:
        env_value = os.getenv("AGENT_PERMISSIONS_PATH", "").strip()
        if env_value:
            return env_value
        return str(Path.home() / ".config" / "personal-agent" / "permissions.json")

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return default_permissions_document()
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return default_permissions_document()
        if not isinstance(parsed, dict):
            return default_permissions_document()
        return _normalize_document(parsed)

    def save(self, document: dict[str, Any]) -> dict[str, Any]:
        normalized = _normalize_document(document)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(normalized, ensure_ascii=True, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
        return normalized

    def update(self, partial: dict[str, Any]) -> dict[str, Any]:
        current = self.load()
        merged = {
            **current,
            **(partial if isinstance(partial, dict) else {}),
        }
        if isinstance(current.get("actions"), dict) and isinstance(partial.get("actions") if isinstance(partial, dict) else None, dict):
            merged["actions"] = {
                **current["actions"],
                **partial["actions"],
            }
        if isinstance(current.get("constraints"), dict) and isinstance(
            partial.get("constraints") if isinstance(partial, dict) else None,
            dict,
        ):
            merged["constraints"] = {
                **current["constraints"],
                **partial["constraints"],
            }
        return self.save(merged)


class PermissionPolicy:
    def evaluate(self, request: PermissionRequest, permissions_document: dict[str, Any]) -> PermissionDecision:
        permissions = _normalize_document(permissions_document)
        action = str(request.action or "").strip()

        if action not in MODEL_OPS_ACTIONS:
            return PermissionDecision(False, "unknown_action", True)

        actions = permissions.get("actions") if isinstance(permissions.get("actions"), dict) else {}
        if not bool(actions.get(action, False)):
            return PermissionDecision(False, "action_not_permitted", True)

        constraints = permissions.get("constraints") if isinstance(permissions.get("constraints"), dict) else {}

        if action == "modelops.install_ollama" and not bool(constraints.get("allow_install_ollama", False)):
            return PermissionDecision(False, "install_not_allowed", True)

        max_download_bytes = int(constraints.get("max_download_bytes", 0) or 0)
        estimated_bytes = int(request.estimated_bytes or 0)
        if estimated_bytes > max_download_bytes:
            return PermissionDecision(False, "download_limit_exceeded", True)

        provider = self._provider_for_request(request)
        allowed_providers = {
            str(item).strip().lower()
            for item in (constraints.get("allowed_providers") or [])
            if str(item).strip()
        }
        if provider and allowed_providers and provider not in allowed_providers:
            return PermissionDecision(False, "provider_not_allowed", True)

        if provider and provider != "ollama" and not bool(constraints.get("allow_remote_models", True)):
            return PermissionDecision(False, "remote_models_not_allowed", True)

        model_id = self._model_for_request(request)
        patterns = [
            str(item).strip()
            for item in (constraints.get("allowed_model_patterns") or [])
            if str(item).strip()
        ]
        if model_id and patterns and not any(fnmatchcase(model_id, pattern) for pattern in patterns):
            return PermissionDecision(False, "model_pattern_not_allowed", True)

        mode = str(permissions.get("mode") or "manual_confirm").strip().lower()
        requires_confirmation = mode != "auto"
        return PermissionDecision(True, "allowed", requires_confirmation)

    @staticmethod
    def _provider_for_request(request: PermissionRequest) -> str | None:
        action = request.action
        params = request.params if isinstance(request.params, dict) else {}

        if action in {"modelops.install_ollama", "modelops.pull_ollama_model", "modelops.import_gguf_to_ollama"}:
            return "ollama"

        if action == "modelops.set_default_model":
            provider = str(params.get("default_provider") or "").strip().lower()
            if provider:
                return provider
            default_model = str(params.get("default_model") or "").strip()
            if ":" in default_model:
                return default_model.split(":", 1)[0].strip().lower() or None
            return None

        if action == "modelops.enable_disable_provider_or_model":
            target_type = str(params.get("target_type") or "").strip().lower()
            target_id = str(params.get("id") or "").strip()
            if target_type == "provider":
                return target_id.lower() or None
            if target_type == "model" and ":" in target_id:
                return target_id.split(":", 1)[0].strip().lower() or None
            return None

        return None

    @staticmethod
    def _model_for_request(request: PermissionRequest) -> str | None:
        action = request.action
        params = request.params if isinstance(request.params, dict) else {}

        if action == "modelops.pull_ollama_model":
            return str(params.get("model") or "").strip() or None

        if action == "modelops.import_gguf_to_ollama":
            return str(params.get("model_name") or "").strip() or None

        if action == "modelops.set_default_model":
            return str(params.get("default_model") or "").strip() or None

        if action == "modelops.enable_disable_provider_or_model":
            target_type = str(params.get("target_type") or "").strip().lower()
            if target_type == "model":
                return str(params.get("id") or "").strip() or None

        return None

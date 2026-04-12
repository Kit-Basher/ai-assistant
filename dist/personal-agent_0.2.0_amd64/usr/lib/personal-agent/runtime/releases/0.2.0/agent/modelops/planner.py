from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.permissions import MODEL_OPS_ACTIONS


class ModelOpsPlanner:
    def __init__(self, installer_script_path: str) -> None:
        self.installer_script_path = str(Path(installer_script_path).resolve())

    def plan(self, action: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized_action = str(action or "").strip()
        normalized_params = self._normalize_params(params or {})
        if normalized_action not in MODEL_OPS_ACTIONS:
            raise ValueError("unsupported action")

        if normalized_action == "modelops.install_ollama":
            estimated_download = self._estimated_download_bytes(normalized_params, default_bytes=250 * 1024 * 1024)
            return {
                "action": normalized_action,
                "params": normalized_params,
                "risk_level": "high",
                "estimated_download_bytes": estimated_download,
                "estimated_duration_seconds": 120,
                "steps": [
                    {
                        "id": "check_ollama_binary",
                        "type": "command",
                        "description": "Check whether ollama is already installed.",
                        "command": ["ollama", "--version"],
                        "timeout_seconds": 10,
                    },
                    {
                        "id": "install_ollama",
                        "type": "command",
                        "description": "Install ollama using the bundled official installer script.",
                        "command": ["sh", self.installer_script_path],
                        "timeout_seconds": 600,
                    },
                ],
            }

        if normalized_action == "modelops.pull_ollama_model":
            model = str(normalized_params.get("model") or "").strip()
            if not model:
                raise ValueError("model is required")
            estimated_download = self._estimated_download_bytes(normalized_params, default_bytes=3 * 1024 * 1024 * 1024)
            return {
                "action": normalized_action,
                "params": {"model": model, **{k: v for k, v in normalized_params.items() if k != "model"}},
                "risk_level": "medium",
                "estimated_download_bytes": estimated_download,
                "estimated_duration_seconds": 300,
                "steps": [
                    {
                        "id": "pull_model",
                        "type": "command",
                        "description": f"Pull ollama model {model}.",
                        "command": ["ollama", "pull", model],
                        "timeout_seconds": 1800,
                    }
                ],
            }

        if normalized_action == "modelops.import_gguf_to_ollama":
            model_name = str(normalized_params.get("model_name") or "").strip()
            modelfile_path = str(normalized_params.get("modelfile_path") or "").strip()
            if not model_name:
                raise ValueError("model_name is required")
            if not modelfile_path:
                raise ValueError("modelfile_path is required")
            return {
                "action": normalized_action,
                "params": {
                    "model_name": model_name,
                    "modelfile_path": modelfile_path,
                },
                "risk_level": "medium",
                "estimated_download_bytes": self._estimated_download_bytes(normalized_params, default_bytes=0),
                "estimated_duration_seconds": 120,
                "steps": [
                    {
                        "id": "create_model",
                        "type": "command",
                        "description": f"Create ollama model {model_name} from Modelfile.",
                        "command": ["ollama", "create", model_name, "-f", modelfile_path],
                        "timeout_seconds": 900,
                    }
                ],
            }

        if normalized_action == "modelops.set_default_model":
            default_provider = str(normalized_params.get("default_provider") or "").strip().lower()
            default_model = str(normalized_params.get("default_model") or "").strip()
            if not default_provider:
                raise ValueError("default_provider is required")
            if not default_model:
                raise ValueError("default_model is required")
            return {
                "action": normalized_action,
                "params": {
                    "default_provider": default_provider,
                    "default_model": default_model,
                },
                "risk_level": "low",
                "estimated_download_bytes": 0,
                "estimated_duration_seconds": 1,
                "steps": [
                    {
                        "id": "set_defaults",
                        "type": "set_default_model",
                        "description": "Update registry defaults for provider/model.",
                        "payload": {
                            "default_provider": default_provider,
                            "default_model": default_model,
                        },
                    }
                ],
            }

        if normalized_action == "modelops.enable_disable_provider_or_model":
            target_type = str(normalized_params.get("target_type") or "").strip().lower()
            target_id = str(normalized_params.get("id") or "").strip()
            enabled = bool(normalized_params.get("enabled"))
            if target_type not in {"provider", "model"}:
                raise ValueError("target_type must be provider or model")
            if not target_id:
                raise ValueError("id is required")
            return {
                "action": normalized_action,
                "params": {
                    "target_type": target_type,
                    "id": target_id,
                    "enabled": enabled,
                },
                "risk_level": "low",
                "estimated_download_bytes": 0,
                "estimated_duration_seconds": 1,
                "steps": [
                    {
                        "id": "toggle_enabled",
                        "type": "toggle_enabled",
                        "description": f"Set {target_type} {target_id} enabled={enabled}.",
                        "payload": {
                            "target_type": target_type,
                            "id": target_id,
                            "enabled": enabled,
                        },
                    }
                ],
            }

        raise ValueError("unsupported action")

    @staticmethod
    def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key in sorted(params.keys()):
            value = params[key]
            if isinstance(value, str):
                normalized[key] = value.strip()
            else:
                normalized[key] = value
        return normalized

    @staticmethod
    def _estimated_download_bytes(params: dict[str, Any], *, default_bytes: int) -> int:
        if "estimated_download_bytes" in params:
            try:
                return max(0, int(params.get("estimated_download_bytes") or 0))
            except Exception:
                return max(0, int(default_bytes))
        if "estimated_download_gb" in params:
            try:
                gb = float(params.get("estimated_download_gb") or 0)
                return max(0, int(gb * 1024 * 1024 * 1024))
            except Exception:
                return max(0, int(default_bytes))
        return max(0, int(default_bytes))

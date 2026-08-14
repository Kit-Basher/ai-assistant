from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable
from pathlib import Path

from agent.llm.capabilities import (
    capability_list_from_inference,
    infer_capabilities_from_catalog,
    is_embedding_model_name,
    is_vision_model_name,
)
from agent.llm.ollama_endpoints import normalize_ollama_base_urls


MODEL_TRUTH_CONTRACT = "personal-agent.model-runtime-truth.v1"
MODEL_TRUTH_STALE_AFTER_SECONDS = 300
MODEL_TRUTH_MAX_METADATA_CHARS = 512


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded(value: Any, limit: int = MODEL_TRUTH_MAX_METADATA_CHARS) -> str | None:
    text = str(value or "").strip()
    return text[:limit] if text else None


def canonical_native_name(value: str | None) -> str:
    name = str(value or "").strip()
    if name.lower().startswith("ollama:"):
        name = name.split(":", 1)[1]
    if not name:
        return ""
    if ":" not in name:
        name = f"{name}:latest"
    return name.lower()


def canonical_model_ref(provider: str | None, model: str | None) -> str:
    provider_key = str(provider or "").strip().lower()
    if provider_key == "ollama":
        native = canonical_native_name(model)
        return f"ollama:{native}" if native else ""
    raw = str(model or "").strip()
    if raw.lower().startswith(f"{provider_key}:"):
        return f"{provider_key}:{raw.split(':', 1)[1]}"
    return f"{provider_key}:{raw}" if provider_key and raw else raw


def _roles(name: str, capabilities: Iterable[str]) -> list[str]:
    caps = {str(item).strip().lower() for item in capabilities if str(item).strip()}
    normalized = name.lower()
    roles: set[str] = set()
    if "embedding" in caps or is_embedding_model_name(normalized):
        roles.add("embedding")
    if "vision" in caps or is_vision_model_name(normalized):
        roles.add("vision")
    if "chat" in caps:
        roles.add("general_chat")
    if any(marker in normalized for marker in ("coder", "code", "starcoder")):
        roles.add("coding")
    if any(marker in normalized for marker in ("reason", "deepseek-r1", "qwen3")):
        roles.add("reasoning")
    return sorted(roles)


class ModelRuntimeTruth:
    """Canonical, side-effect-free model observation and reconciliation.

    A provider timeout never means "not installed". The last successful physical
    observation remains available with explicit age/staleness until refresh works.
    """

    def __init__(self, runtime: Any, *, timeout_seconds: float = 1.0) -> None:
        self.runtime = runtime
        self.timeout_seconds = max(0.1, min(float(timeout_seconds), 3.0))
        self._last_success: dict[str, Any] | None = None

    def _ollama_base(self) -> str:
        config = getattr(self.runtime, "config", None)
        configured = str(
            getattr(config, "ollama_base_url", None)
            or getattr(config, "ollama_host", None)
            or "http://127.0.0.1:11434"
        ).strip()
        return str(normalize_ollama_base_urls(configured).get("native_base") or configured).rstrip("/")

    def _ollama_tags(self) -> tuple[list[dict[str, Any]], str | None]:
        request = urllib.request.Request(
            f"{self._ollama_base()}/api/tags",
            method="GET",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                return [], "provider_response_too_large"
            payload = json.loads(raw.decode("utf-8"))
            rows = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                return [], "provider_response_malformed"
            return [dict(row) for row in rows if isinstance(row, dict)][:512], None
        except (urllib.error.URLError, TimeoutError):
            return [], "provider_unavailable"
        except (UnicodeError, json.JSONDecodeError, OSError, ValueError):
            return [], "provider_response_malformed"

    def _registry_rows(self) -> list[dict[str, Any]]:
        truth = getattr(self.runtime, "runtime_truth_service", lambda: None)()
        inventory = getattr(truth, "_runtime_inventory_rows", lambda: [])()
        return [dict(row) for row in inventory if isinstance(row, dict)][:2048]

    def _manager_history(self) -> dict[str, Any]:
        truth = getattr(self.runtime, "runtime_truth_service", lambda: None)()
        reader = getattr(truth, "_model_manager_state", None)
        state = reader() if callable(reader) else {}
        targets = state.get("targets") if isinstance(state, dict) and isinstance(state.get("targets"), dict) else {}
        rows = [dict(row) for row in targets.values() if isinstance(row, dict)][:512]
        active_states = {"queued", "downloading"}
        active = [row for row in rows if str(row.get("state") or "").strip().lower() in active_states]
        history = [row for row in rows if row not in active]
        return {
            "schema_version": int(state.get("schema_version") or 1) if isinstance(state, dict) else 1,
            "active": active,
            "history": history,
            "counts": {"active": len(active), "history": len(history), "total": len(rows)},
            "source": "canonical_model_manager_state",
        }

    def _selection(self) -> dict[str, Any]:
        defaults = getattr(self.runtime, "get_defaults", lambda: {})()
        defaults = dict(defaults) if isinstance(defaults, dict) else {}
        configured = str(defaults.get("resolved_default_model") or defaults.get("default_model") or "").strip()
        provider = str(defaults.get("default_provider") or "").strip().lower()
        effective = configured
        override = None
        override_reader = getattr(self.runtime, "_temporary_model_override_status", None)
        if callable(override_reader):
            try:
                raw_override = override_reader()
            except Exception:
                raw_override = None
            if isinstance(raw_override, dict):
                override = str(raw_override.get("model_id") or raw_override.get("model") or "").strip() or None
                if override:
                    effective = override
        return {
            "provider": provider or (configured.split(":", 1)[0].lower() if ":" in configured else None),
            "default_model": configured or None,
            "temporary_override": override,
            "effective_model": effective or None,
            "remote_fallback": bool(defaults.get("allow_remote_fallback", True)),
        }

    def _evaluation_evidence(self) -> dict[str, Any]:
        repo_root = Path(str(getattr(self.runtime, "_repo_root", "") or Path(__file__).resolve().parents[2]))
        path = repo_root / "agent" / "data" / "model_runtime_evidence.json"
        try:
            raw = path.read_bytes()
            if len(raw) > 2 * 1024 * 1024:
                return {"status": "unavailable", "reason": "evidence_too_large"}
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return {"status": "unavailable", "reason": "no_current_evaluation"}
        if not isinstance(payload, dict) or payload.get("contract") != "personal-agent.model-evaluation.v1":
            return {"status": "unavailable", "reason": "invalid_evidence_contract"}
        observed_at = str(payload.get("observed_at") or "").strip()
        age_seconds = None
        try:
            observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
            age_seconds = max(0.0, (datetime.now(timezone.utc) - observed).total_seconds())
        except (TypeError, ValueError):
            pass
        return {
            **payload,
            "status": "current" if age_seconds is not None and age_seconds <= 604800 else "stale",
            "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
            "code_commit": str(getattr(self.runtime, "git_commit", "") or "").strip() or None,
        }

    @staticmethod
    def _installed_row(raw: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any] | None:
        native_name = _bounded(raw.get("name") or raw.get("model"), 200)
        if not native_name:
            return None
        details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
        inferred = infer_capabilities_from_catalog("ollama", {"model": native_name})
        capabilities = capability_list_from_inference(inferred)
        canonical_id = canonical_model_ref("ollama", native_name)
        default_id = canonical_model_ref("ollama", selection.get("default_model"))
        effective_id = canonical_model_ref("ollama", selection.get("effective_model"))
        override_id = canonical_model_ref("ollama", selection.get("temporary_override"))
        return {
            "canonical_id": canonical_id,
            "provider": "ollama",
            "provider_native_id": native_name,
            "aliases": sorted({native_name, canonical_id}),
            "digest": _bounded(raw.get("digest"), 160),
            "size_bytes": int(raw.get("size")) if isinstance(raw.get("size"), int) else None,
            "parameter_size": _bounded(details.get("parameter_size"), 80),
            "quantization": _bounded(details.get("quantization_level"), 80),
            "family": _bounded(details.get("family"), 80),
            "format": _bounded(details.get("format"), 80),
            "context_window": None,
            "capabilities": capabilities,
            "roles": _roles(native_name, capabilities),
            "local": True,
            "physically_installed": True,
            "physical_observation": "observed",
            "registered": False,
            "ready": "chat" in capabilities,
            "routable": "chat" in capabilities,
            "chat_eligible": "chat" in capabilities,
            "eligibility_reason": "installed_chat_model" if "chat" in capabilities else "non_chat_model",
            "default": canonical_id == default_id,
            "selected": canonical_id == default_id,
            "effective": canonical_id == effective_id,
            "temporary_override": bool(override_id and canonical_id == override_id),
            "benchmark": {"status": "not_evaluated", "observed_at": None},
        }

    def refresh(self) -> dict[str, Any]:
        started = time.monotonic()
        observed_at = _utc_now()
        tags, error = self._ollama_tags()
        if error:
            if isinstance(self._last_success, dict):
                stale = deepcopy(self._last_success)
                stale["observation"] = {
                    **dict(stale.get("observation") or {}),
                    "status": "unknown",
                    "last_refresh_attempt_at": observed_at,
                    "last_refresh_error": error,
                    "stale": True,
                }
                return stale
            return {
                "contract": MODEL_TRUTH_CONTRACT,
                "ok": False,
                "observation": {
                    "status": "unknown",
                    "observed_at": observed_at,
                    "last_refresh_error": error,
                    "stale": True,
                },
                "selection": self._selection(),
                "installed": [],
                "registered_not_observed": [],
                "remote_registered": [],
                "history_only": [],
                "counts": {"physically_installed": 0, "installed_chat_eligible": 0},
            }

        selection = self._selection()
        manager = self._manager_history()
        evaluation = self._evaluation_evidence()
        evaluated_by_name = {
            canonical_native_name(str(row.get("model") or "")): row
            for row in (
                evaluation.get("evaluated_models")
                if isinstance(evaluation.get("evaluated_models"), list)
                else []
            )
            if isinstance(row, dict)
        }
        installed_by_id: dict[str, dict[str, Any]] = {}
        digest_owner: dict[str, str] = {}
        for raw in tags:
            row = self._installed_row(raw, selection)
            if not row:
                continue
            canonical_id = str(row["canonical_id"])
            digest = str(row.get("digest") or "")
            if digest and digest in digest_owner:
                owner = installed_by_id[digest_owner[digest]]
                owner["aliases"] = sorted({*owner.get("aliases", []), *row.get("aliases", [])})
                continue
            installed_by_id[canonical_id] = row
            if digest:
                digest_owner[digest] = canonical_id
        for canonical_id, row in installed_by_id.items():
            benchmark = evaluated_by_name.get(canonical_native_name(str(row.get("provider_native_id") or "")))
            if isinstance(benchmark, dict):
                row["benchmark"] = {
                    "status": str(evaluation.get("status") or "unknown"),
                    "observed_at": evaluation.get("observed_at"),
                    "score": dict(benchmark.get("score") or {}),
                    "latency": dict(benchmark.get("latency") or {}),
                    "provider_metrics": dict(benchmark.get("provider_metrics") or {}),
                }
                latency = benchmark.get("latency") if isinstance(benchmark.get("latency"), dict) else {}
                stability = benchmark.get("stability") if isinstance(benchmark.get("stability"), dict) else {}
                if int(latency.get("samples") or 0) == 0 and int(stability.get("errors") or 0) > 0:
                    row["ready"] = False
                    row["routable"] = False
                    row["eligibility_reason"] = "benchmark_provider_unusable"

        registered_not_observed: list[dict[str, Any]] = []
        remote_registered: list[dict[str, Any]] = []
        for registry_row in self._registry_rows():
            provider = str(registry_row.get("provider") or "").strip().lower()
            raw_id = str(registry_row.get("id") or "").strip()
            canonical_id = canonical_model_ref(provider, raw_id)
            if provider == "ollama" and canonical_id in installed_by_id:
                row = installed_by_id[canonical_id]
                row["registered"] = True
                row["routable"] = bool(registry_row.get("routable", row.get("routable")))
                row["ready"] = bool(registry_row.get("available", row.get("ready")))
                row["aliases"] = sorted({*row.get("aliases", []), raw_id})
                continue
            normalized = {
                "canonical_id": canonical_id or raw_id,
                "provider": provider,
                "provider_native_id": str(registry_row.get("model_name") or raw_id).strip(),
                "registered": True,
                "physically_installed": False if provider == "ollama" else None,
                "physical_observation": "not_observed" if provider == "ollama" else "not_applicable",
                "ready": bool(registry_row.get("available", False)),
                "routable": bool(registry_row.get("routable", False)),
                "health_status": str(registry_row.get("health_status") or "unknown"),
                "history_reason": str(registry_row.get("health_reason") or "registered_not_observed"),
            }
            (registered_not_observed if provider == "ollama" else remote_registered).append(normalized)

        for row in installed_by_id.values():
            benchmark = row.get("benchmark") if isinstance(row.get("benchmark"), dict) else {}
            benchmark_row = evaluated_by_name.get(canonical_native_name(str(row.get("provider_native_id") or "")))
            latency = benchmark.get("latency") if isinstance(benchmark.get("latency"), dict) else {}
            stability = benchmark_row.get("stability") if isinstance(benchmark_row, dict) and isinstance(benchmark_row.get("stability"), dict) else {}
            if int(latency.get("samples") or 0) == 0 and int(stability.get("errors") or 0) > 0:
                row["ready"] = False
                row["routable"] = False
                row["eligibility_reason"] = "benchmark_provider_unusable"

        installed = sorted(
            installed_by_id.values(),
            key=lambda row: (not bool(row.get("effective")), not bool(row.get("chat_eligible")), str(row["canonical_id"])),
        )
        payload = {
            "contract": MODEL_TRUTH_CONTRACT,
            "ok": True,
            "observation": {
                "status": "current",
                "source": "ollama:/api/tags",
                "observed_at": observed_at,
                "last_refresh_attempt_at": observed_at,
                "last_refresh_error": None,
                "stale": False,
                "stale_after_seconds": MODEL_TRUTH_STALE_AFTER_SECONDS,
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
            },
            "selection": selection,
            "installed": installed,
            "registered_not_observed": sorted(registered_not_observed, key=lambda row: str(row["canonical_id"])),
            "remote_registered": sorted(remote_registered, key=lambda row: str(row["canonical_id"])),
            "history_only": list(manager.get("history") or []),
            "manager": manager,
            "scout": {
                "status": "completed" if evaluation.get("status") in {"current", "stale"} else "never_completed",
                "last_run_at": evaluation.get("observed_at"),
                "result": dict(evaluation.get("recommendation") or {}),
                "age_seconds": evaluation.get("age_seconds"),
                "error": None if evaluation.get("status") in {"current", "stale"} else evaluation.get("reason"),
                "next_action": "Refresh installed model evaluation when evidence is stale.",
                "source": "installed_model_evaluation",
            },
            "evaluation": evaluation,
            "recommendation": dict(evaluation.get("recommendation") or {}),
            "counts": {
                "physically_installed": len(installed),
                "installed_chat_eligible": sum(bool(row.get("chat_eligible")) for row in installed),
                "installed_non_chat": sum(not bool(row.get("chat_eligible")) for row in installed),
                "registered_not_observed": len(registered_not_observed),
                "remote_registered": len(remote_registered),
                "duplicate_aliases_collapsed": sum(max(0, len(row.get("aliases", [])) - 2) for row in installed),
            },
        }
        self._last_success = deepcopy(payload)
        return payload


__all__ = [
    "MODEL_TRUTH_CONTRACT",
    "MODEL_TRUTH_STALE_AFTER_SECONDS",
    "ModelRuntimeTruth",
    "canonical_model_ref",
    "canonical_native_name",
]

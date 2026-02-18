from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable


_HEALTH_SCHEMA_VERSION = 1
_STATUS_ORDER = {"ok": 0, "degraded": 1, "down": 2, "unknown": 3}
_NOT_APPLICABLE_ERROR = "not_applicable"


def _now_epoch() -> int:
    return int(time.time())


def _iso_from_epoch(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    if epoch <= 0:
        return None
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class HealthProbeSettings:
    interval_seconds: int = 900
    max_probes_per_run: int = 6
    probe_timeout_seconds: float = 6.0
    initial_backoff_seconds: int = 60
    max_backoff_seconds: int = 3600
    models_per_provider: int = 2


class HealthStateStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()

    @staticmethod
    def default_path() -> str:
        env_path = os.getenv("LLM_HEALTH_STATE_PATH", "").strip()
        if env_path:
            return env_path
        return str(Path.home() / ".local" / "share" / "personal-agent" / "llm_health_state.json")

    @staticmethod
    def empty_state() -> dict[str, Any]:
        return {
            "schema_version": _HEALTH_SCHEMA_VERSION,
            "last_run_at": None,
            "providers": {},
            "models": {},
        }

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self.empty_state()
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return self.empty_state()
        if not isinstance(parsed, dict):
            return self.empty_state()
        return self._normalize(parsed)

    def save(self, state: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(state if isinstance(state, dict) else {})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(normalized, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
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

    def _normalize(self, state: dict[str, Any]) -> dict[str, Any]:
        output = self.empty_state()
        output["schema_version"] = _HEALTH_SCHEMA_VERSION
        output["last_run_at"] = _safe_int(state.get("last_run_at"), 0) or None

        providers_raw = state.get("providers") if isinstance(state.get("providers"), dict) else {}
        providers: dict[str, dict[str, Any]] = {}
        for provider_id, payload in sorted(providers_raw.items()):
            if not isinstance(payload, dict):
                continue
            pid = str(provider_id).strip().lower()
            if not pid:
                continue
            providers[pid] = {
                "status": _normalize_status(payload.get("status")),
                "last_error_kind": _normalize_error_kind(payload.get("last_error_kind")),
                "status_code": _safe_int(payload.get("status_code"), 0) or None,
                "last_checked_at": _safe_int(payload.get("last_checked_at"), 0) or None,
                "cooldown_until": _safe_int(payload.get("cooldown_until"), 0) or None,
                "down_since": _safe_int(payload.get("down_since"), 0) or None,
                "failure_streak": max(0, _safe_int(payload.get("failure_streak"), 0)),
                "next_probe_at": _safe_int(payload.get("next_probe_at"), 0) or None,
            }
        output["providers"] = providers

        models_raw = state.get("models") if isinstance(state.get("models"), dict) else {}
        models: dict[str, dict[str, Any]] = {}
        for model_id, payload in sorted(models_raw.items()):
            if not isinstance(payload, dict):
                continue
            mid = str(model_id).strip()
            if not mid:
                continue
            provider_id = str(payload.get("provider_id") or "").strip().lower()
            if not provider_id and ":" in mid:
                provider_id = mid.split(":", 1)[0].strip().lower()
            models[mid] = {
                "provider_id": provider_id,
                "status": _normalize_status(payload.get("status")),
                "last_error_kind": _normalize_error_kind(payload.get("last_error_kind")),
                "status_code": _safe_int(payload.get("status_code"), 0) or None,
                "last_checked_at": _safe_int(payload.get("last_checked_at"), 0) or None,
                "cooldown_until": _safe_int(payload.get("cooldown_until"), 0) or None,
                "down_since": _safe_int(payload.get("down_since"), 0) or None,
                "failure_streak": max(0, _safe_int(payload.get("failure_streak"), 0)),
                "next_probe_at": _safe_int(payload.get("next_probe_at"), 0) or None,
            }
        output["models"] = models
        return output


def _normalize_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"ok", "degraded", "down"}:
        return normalized
    return "unknown"


def _normalize_error_kind(value: Any) -> str | None:
    kind = str(value or "").strip().lower()
    return kind or None


class LLMHealthMonitor:
    def __init__(
        self,
        settings: HealthProbeSettings,
        *,
        store: HealthStateStore,
        probe_fn: Callable[[str, str, float], dict[str, Any]],
        now_fn: Callable[[], int] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self._probe_fn = probe_fn
        self._now_fn = now_fn or _now_epoch
        self.state = self.store.load()

    def run_once(self, registry_document: dict[str, Any]) -> dict[str, Any]:
        now = int(self._now_fn())
        candidates = self._candidate_models(registry_document)
        probed: list[dict[str, Any]] = []
        skipped = 0
        not_applicable = 0

        for provider_id, model_id in candidates:
            if len(probed) >= int(self.settings.max_probes_per_run):
                break
            model_state = self._get_model_state(model_id, provider_id)
            next_probe_at = _safe_int(model_state.get("next_probe_at"), 0)
            if next_probe_at and now < next_probe_at:
                skipped += 1
                continue

            probe = self._probe_fn(provider_id, model_id, float(self.settings.probe_timeout_seconds))
            if _is_not_applicable_probe(probe):
                not_applicable += 1
                model_state["status"] = "unknown"
                model_state["last_error_kind"] = _NOT_APPLICABLE_ERROR
                model_state["status_code"] = None
                model_state["last_checked_at"] = now
                model_state["cooldown_until"] = None
                model_state["down_since"] = None
                model_state["failure_streak"] = 0
                model_state["next_probe_at"] = now + max(1, int(self.settings.interval_seconds))
                probed.append(
                    {
                        "provider_id": provider_id,
                        "model_id": model_id,
                        "status": "not_applicable",
                        "error_kind": _NOT_APPLICABLE_ERROR,
                        "status_code": None,
                    }
                )
                continue

            updated = self._apply_model_probe_result(
                model_id=model_id,
                provider_id=provider_id,
                result=probe,
                now=now,
            )
            probed.append(
                {
                    "provider_id": provider_id,
                    "model_id": model_id,
                    "status": updated["status"],
                    "error_kind": updated.get("last_error_kind"),
                    "status_code": updated.get("status_code"),
                }
            )

        self._refresh_provider_states(registry_document, now=now)
        self.state["last_run_at"] = now
        self.state = self.store.save(self.state)
        summary = self.summary(registry_document)
        summary["probed"] = probed
        summary["skipped"] = int(skipped)
        summary["not_applicable"] = int(not_applicable)
        summary["max_probes_per_run"] = int(self.settings.max_probes_per_run)
        return summary

    def summary(self, registry_document: dict[str, Any]) -> dict[str, Any]:
        providers = self.state.get("providers") if isinstance(self.state.get("providers"), dict) else {}
        models = self.state.get("models") if isinstance(self.state.get("models"), dict) else {}
        provider_docs = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
        model_docs = registry_document.get("models") if isinstance(registry_document.get("models"), dict) else {}

        provider_rows: list[dict[str, Any]] = []
        for provider_id in sorted(provider_docs.keys()):
            state = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
            provider_rows.append(
                {
                    "id": provider_id,
                    "status": _normalize_status(state.get("status")),
                    "last_error_kind": state.get("last_error_kind"),
                    "status_code": state.get("status_code"),
                    "failure_streak": max(0, _safe_int(state.get("failure_streak"), 0)),
                    "last_checked_at": state.get("last_checked_at"),
                    "last_checked_at_iso": _iso_from_epoch(state.get("last_checked_at")),
                    "cooldown_until": state.get("cooldown_until"),
                    "cooldown_until_iso": _iso_from_epoch(state.get("cooldown_until")),
                }
            )

        model_rows: list[dict[str, Any]] = []
        for model_id in sorted(model_docs.keys()):
            state = models.get(model_id) if isinstance(models.get(model_id), dict) else {}
            model_rows.append(
                {
                    "id": model_id,
                    "provider_id": str((state.get("provider_id") or model_docs.get(model_id, {}).get("provider") or "")).strip().lower(),
                    "status": _normalize_status(state.get("status")),
                    "last_error_kind": state.get("last_error_kind"),
                    "status_code": state.get("status_code"),
                    "failure_streak": max(0, _safe_int(state.get("failure_streak"), 0)),
                    "last_checked_at": state.get("last_checked_at"),
                    "last_checked_at_iso": _iso_from_epoch(state.get("last_checked_at")),
                    "cooldown_until": state.get("cooldown_until"),
                    "cooldown_until_iso": _iso_from_epoch(state.get("cooldown_until")),
                    "down_since": state.get("down_since"),
                    "down_since_iso": _iso_from_epoch(state.get("down_since")),
                }
            )

        status_counts = {"ok": 0, "degraded": 0, "down": 0, "unknown": 0, "not_applicable": 0}
        for row in model_rows:
            if str(row.get("last_error_kind") or "").strip().lower() == _NOT_APPLICABLE_ERROR:
                status_counts["not_applicable"] = int(status_counts.get("not_applicable", 0)) + 1
                continue
            status = _normalize_status(row.get("status"))
            status_counts[status] = int(status_counts.get(status, 0)) + 1

        return {
            "ok": True,
            "last_run_at": self.state.get("last_run_at"),
            "last_run_at_iso": _iso_from_epoch(self.state.get("last_run_at")),
            "providers": provider_rows,
            "models": model_rows,
            "counts": status_counts,
        }

    def _candidate_models(self, registry_document: dict[str, Any]) -> list[tuple[str, str]]:
        providers = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
        models = registry_document.get("models") if isinstance(registry_document.get("models"), dict) else {}
        candidates: list[tuple[str, str]] = []

        for provider_id, provider_payload in sorted(providers.items()):
            if not isinstance(provider_payload, dict):
                continue
            if not bool(provider_payload.get("enabled", True)):
                continue
            provider_models: list[str] = []
            for model_id, model_payload in sorted(models.items()):
                if not isinstance(model_payload, dict):
                    continue
                if str(model_payload.get("provider") or "").strip().lower() != str(provider_id).strip().lower():
                    continue
                if not bool(model_payload.get("enabled", True)) or not bool(model_payload.get("available", True)):
                    continue
                capabilities = {
                    str(item).strip().lower()
                    for item in (model_payload.get("capabilities") or [])
                    if str(item).strip()
                }
                if "chat" not in capabilities:
                    continue
                provider_models.append(str(model_id))

            for model_id in provider_models[: max(1, int(self.settings.models_per_provider))]:
                candidates.append((str(provider_id).strip().lower(), model_id))
        return candidates

    def _get_model_state(self, model_id: str, provider_id: str) -> dict[str, Any]:
        models = self.state.setdefault("models", {})
        if not isinstance(models, dict):
            models = {}
            self.state["models"] = models
        model_state = models.get(model_id)
        if not isinstance(model_state, dict):
            model_state = {
                "provider_id": provider_id,
                "status": "unknown",
                "last_error_kind": None,
                "status_code": None,
                "last_checked_at": None,
                "cooldown_until": None,
                "down_since": None,
                "failure_streak": 0,
                "next_probe_at": None,
            }
            models[model_id] = model_state
        model_state["provider_id"] = provider_id
        return model_state

    def _get_provider_state(self, provider_id: str) -> dict[str, Any]:
        providers = self.state.setdefault("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            self.state["providers"] = providers
        provider_state = providers.get(provider_id)
        if not isinstance(provider_state, dict):
            provider_state = {
                "status": "unknown",
                "last_error_kind": None,
                "status_code": None,
                "last_checked_at": None,
                "cooldown_until": None,
                "down_since": None,
                "failure_streak": 0,
                "next_probe_at": None,
            }
            providers[provider_id] = provider_state
        return provider_state

    def _apply_model_probe_result(
        self,
        *,
        model_id: str,
        provider_id: str,
        result: dict[str, Any],
        now: int,
    ) -> dict[str, Any]:
        state = self._get_model_state(model_id, provider_id)
        ok = bool(result.get("ok"))
        state["last_checked_at"] = now

        if ok:
            state["status"] = "ok"
            state["last_error_kind"] = None
            state["status_code"] = None
            state["failure_streak"] = 0
            state["cooldown_until"] = None
            state["next_probe_at"] = now + max(1, int(self.settings.interval_seconds))
            state["down_since"] = None
            return state

        error_kind = _normalize_error_kind(result.get("error_kind") or result.get("error"))
        status_code = _safe_int(result.get("status_code"), 0) or None
        status = _status_for_error(error_kind, status_code)
        state["status"] = status
        state["last_error_kind"] = error_kind
        state["status_code"] = status_code
        state["failure_streak"] = max(1, _safe_int(state.get("failure_streak"), 0) + 1)
        cooldown = _cooldown_seconds(
            status=status,
            failure_streak=int(state["failure_streak"]),
            base_seconds=max(1, int(self.settings.initial_backoff_seconds)),
            max_seconds=max(1, int(self.settings.max_backoff_seconds)),
        )
        state["cooldown_until"] = now + cooldown
        state["next_probe_at"] = now + cooldown
        if status == "down":
            state["down_since"] = _safe_int(state.get("down_since"), 0) or now
        else:
            state["down_since"] = None
        return state

    def _refresh_provider_states(self, registry_document: dict[str, Any], *, now: int) -> None:
        providers_doc = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
        models_state = self.state.get("models") if isinstance(self.state.get("models"), dict) else {}

        for provider_id, provider_payload in sorted(providers_doc.items()):
            state = self._get_provider_state(provider_id)
            if not bool(isinstance(provider_payload, dict) and provider_payload.get("enabled", True)):
                state["status"] = "down"
                state["last_error_kind"] = "provider_disabled"
                state["last_checked_at"] = now
                continue

            provider_models = [
                payload
                for payload in models_state.values()
                if isinstance(payload, dict) and str(payload.get("provider_id") or "").strip().lower() == provider_id
            ]
            applicable_models = [
                payload
                for payload in provider_models
                if str(payload.get("last_error_kind") or "").strip().lower() != _NOT_APPLICABLE_ERROR
            ]
            if not applicable_models:
                state["status"] = "unknown"
                state["last_error_kind"] = None
                state["status_code"] = None
                state["last_checked_at"] = now
                state["cooldown_until"] = None
                state["down_since"] = None
                state["failure_streak"] = 0
                state["next_probe_at"] = now + max(1, int(self.settings.interval_seconds))
                continue

            applicable_models.sort(
                key=lambda item: (
                    _STATUS_ORDER.get(_normalize_status(item.get("status")), 3),
                    -_safe_int(item.get("last_checked_at"), 0),
                )
            )
            worst_status = "ok"
            if any(_normalize_status(item.get("status")) == "down" for item in applicable_models):
                worst_status = "down"
            elif any(_normalize_status(item.get("status")) == "degraded" for item in applicable_models):
                worst_status = "degraded"

            latest = max(applicable_models, key=lambda item: _safe_int(item.get("last_checked_at"), 0))
            state["status"] = worst_status
            state["last_error_kind"] = latest.get("last_error_kind")
            state["status_code"] = latest.get("status_code")
            state["last_checked_at"] = latest.get("last_checked_at") or now
            state["cooldown_until"] = max(_safe_int(item.get("cooldown_until"), 0) for item in applicable_models) or None
            state["next_probe_at"] = min(
                (
                    _safe_int(item.get("next_probe_at"), 0)
                    for item in applicable_models
                    if _safe_int(item.get("next_probe_at"), 0) > 0
                ),
                default=now + max(1, int(self.settings.interval_seconds)),
            )
            down_since_values = [
                _safe_int(item.get("down_since"), 0)
                for item in applicable_models
                if _safe_int(item.get("down_since"), 0) > 0
            ]
            state["down_since"] = min(down_since_values) if down_since_values else None
            state["failure_streak"] = max(_safe_int(item.get("failure_streak"), 0) for item in applicable_models)


def _is_not_applicable_probe(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    return str(result.get("error_kind") or "").strip().lower() == _NOT_APPLICABLE_ERROR


def _status_for_error(error_kind: str | None, status_code: int | None) -> str:
    kind = str(error_kind or "").strip().lower()
    code = int(status_code) if status_code is not None else None
    if kind in {"auth_error", "provider_unavailable", "provider_not_implemented"}:
        return "down"
    if code in {401, 403}:
        return "down"
    if kind in {"rate_limit", "server_error", "timeout", "provider_error"}:
        return "degraded"
    if code is not None and 500 <= code <= 599:
        return "degraded"
    return "degraded"


def _cooldown_seconds(*, status: str, failure_streak: int, base_seconds: int, max_seconds: int) -> int:
    streak = max(1, int(failure_streak))
    if status == "down":
        scaled = base_seconds * (2 ** min(streak, 5))
    else:
        scaled = base_seconds * (2 ** min(streak - 1, 4))
    return max(1, min(int(scaled), int(max_seconds)))

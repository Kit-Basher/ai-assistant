from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


_SCHEMA_VERSION = 1
_DEFAULT_MAX_RECENT_APPLY_IDS = 80
_DEFAULT_CHURN_WINDOW_SECONDS = 1800
_DEFAULT_CHURN_MIN_APPLIES = 4
_MODEL_ID_RE = re.compile(r"\b([a-z0-9_-]+:[A-Za-z0-9._/\-]+)\b")

_AUTOPILOT_APPLY_ACTIONS = {
    "llm.autoconfig.apply",
    "llm.hygiene.apply",
    "llm.cleanup.apply",
    "llm.self_heal.apply",
    "llm.capabilities.reconcile.apply",
    "llm.autopilot.bootstrap.apply",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iso_from_epoch(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    if int(epoch) <= 0:
        return None
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


def _normalize_apply_ids(raw_ids: list[Any], *, max_items: int) -> list[str]:
    ordered = [str(item).strip() for item in raw_ids if str(item).strip()]
    if not ordered:
        return []
    deduped_desc: list[str] = []
    seen: set[str] = set()
    for value in reversed(ordered):
        if value in seen:
            continue
        seen.add(value)
        deduped_desc.append(value)
        if len(deduped_desc) >= max(1, int(max_items)):
            break
    return list(reversed(deduped_desc))


class AutopilotSafetyStateStore:
    def __init__(self, path: str | None = None, *, max_recent_apply_ids: int = _DEFAULT_MAX_RECENT_APPLY_IDS) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()
        self.max_recent_apply_ids = max(1, int(max_recent_apply_ids))
        self.state = self.load()

    @staticmethod
    def default_path() -> str:
        env_value = os.getenv("LLM_AUTOPILOT_STATE_PATH", "").strip()
        if env_value:
            return env_value
        return str(Path.home() / ".local" / "share" / "personal-agent" / "autopilot_state.json")

    @staticmethod
    def empty_state() -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "safe_mode_override": None,
            "safe_mode_reason": None,
            "safe_mode_entered_ts": None,
            "last_churn_event_ts": None,
            "last_churn_reason": None,
            "recent_apply_ids": [],
        }

    def _normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = payload if isinstance(payload, dict) else {}
        output = self.empty_state()
        override = source.get("safe_mode_override")
        output["safe_mode_override"] = bool(override) if isinstance(override, bool) else None
        output["safe_mode_reason"] = str(source.get("safe_mode_reason") or "").strip() or None
        output["safe_mode_entered_ts"] = _safe_int(source.get("safe_mode_entered_ts"), 0) or None
        output["last_churn_event_ts"] = _safe_int(source.get("last_churn_event_ts"), 0) or None
        output["last_churn_reason"] = str(source.get("last_churn_reason") or "").strip() or None
        raw_ids = source.get("recent_apply_ids") if isinstance(source.get("recent_apply_ids"), list) else []
        output["recent_apply_ids"] = _normalize_apply_ids(raw_ids, max_items=self.max_recent_apply_ids)
        return output

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self.empty_state()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return self.empty_state()
        if not isinstance(raw, dict):
            return self.empty_state()
        normalized = self._normalize(raw)
        if json.dumps(raw, ensure_ascii=True, sort_keys=True) != json.dumps(normalized, ensure_ascii=True, sort_keys=True):
            try:
                _write_json_atomic(self.path, normalized)
            except OSError:
                pass
        return normalized

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(payload if isinstance(payload, dict) else {})
        try:
            _write_json_atomic(self.path, normalized)
        except OSError:
            # Keep runtime behavior deterministic even when the configured path is not writable.
            pass
        self.state = normalized
        return normalized

    def status(self) -> dict[str, Any]:
        state = self.state if isinstance(self.state, dict) else self.empty_state()
        safe_mode_entered_ts = _safe_int(state.get("safe_mode_entered_ts"), 0) or None
        last_churn_event_ts = _safe_int(state.get("last_churn_event_ts"), 0) or None
        return {
            "safe_mode_override": state.get("safe_mode_override"),
            "safe_mode_reason": state.get("safe_mode_reason"),
            "safe_mode_entered_ts": safe_mode_entered_ts,
            "safe_mode_entered_ts_iso": _iso_from_epoch(safe_mode_entered_ts),
            "last_churn_event_ts": last_churn_event_ts,
            "last_churn_event_ts_iso": _iso_from_epoch(last_churn_event_ts),
            "last_churn_reason": state.get("last_churn_reason"),
            "recent_apply_ids": list(state.get("recent_apply_ids") or []),
        }

    def effective_safe_mode(self, config_safe_mode: bool) -> bool:
        state = self.state if isinstance(self.state, dict) else self.empty_state()
        override = state.get("safe_mode_override")
        return bool(config_safe_mode) or bool(override is True)

    def apply_pause_enabled(self) -> bool:
        state = self.state if isinstance(self.state, dict) else self.empty_state()
        return bool(state.get("safe_mode_override") is True)

    def update_recent_apply_ids(self, apply_ids: list[str]) -> None:
        current = dict(self.state if isinstance(self.state, dict) else self.empty_state())
        current["recent_apply_ids"] = _normalize_apply_ids(apply_ids, max_items=self.max_recent_apply_ids)
        self.save(current)

    def enter_safe_mode(self, *, reason: str, now_epoch: int) -> dict[str, Any]:
        current = dict(self.state if isinstance(self.state, dict) else self.empty_state())
        current["safe_mode_override"] = True
        current["safe_mode_reason"] = str(reason or "churn_detected").strip() or "churn_detected"
        current["safe_mode_entered_ts"] = int(now_epoch)
        current["last_churn_event_ts"] = int(now_epoch)
        current["last_churn_reason"] = str(reason or "churn_detected").strip() or "churn_detected"
        return self.save(current)


def _apply_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in entries:
        if not isinstance(row, dict):
            continue
        action = str(row.get("action") or "").strip()
        if action not in _AUTOPILOT_APPLY_ACTIONS:
            continue
        if str(row.get("outcome") or "").strip() != "success":
            continue
        ts = _safe_int(row.get("ts"), 0)
        if ts <= 0:
            continue
        rows.append(
            {
                "id": str(row.get("id") or "").strip(),
                "ts": ts,
                "action": action,
                "reason": str(row.get("reason") or "").strip(),
                "changed_ids": sorted(
                    {
                        str(item).strip()
                        for item in (row.get("changed_ids") or [])
                        if str(item).strip()
                    }
                ),
            }
        )
    rows.sort(key=lambda item: (int(item.get("ts") or 0), str(item.get("id") or "")))
    return rows


def _extract_model_id(reason: str) -> str | None:
    value = str(reason or "").strip()
    if not value:
        return None
    match = _MODEL_ID_RE.search(value)
    if not match:
        return None
    candidate = str(match.group(1) or "").strip()
    if ":" not in candidate:
        return None
    provider, model = candidate.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        return None
    return f"{provider}:{model}"


def _flip_flop_sequence(window_rows: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in window_rows:
        changed_ids = {
            str(item).strip()
            for item in (row.get("changed_ids") or [])
            if str(item).strip()
        }
        if "defaults:default_model" not in changed_ids:
            continue
        candidate = _extract_model_id(str(row.get("reason") or ""))
        if not candidate:
            continue
        if values and values[-1] == candidate:
            continue
        values.append(candidate)
    return values


def detect_autopilot_churn(
    entries: list[dict[str, Any]],
    *,
    now_epoch: int,
    window_seconds: int = _DEFAULT_CHURN_WINDOW_SECONDS,
    min_applies: int = _DEFAULT_CHURN_MIN_APPLIES,
) -> dict[str, Any]:
    rows = _apply_rows(entries if isinstance(entries, list) else [])
    now_value = int(now_epoch)
    window = max(60, int(window_seconds))
    min_count = max(2, int(min_applies))
    cutoff = now_value - window
    in_window = [row for row in rows if int(row.get("ts") or 0) >= cutoff]
    apply_count = len(in_window)
    flip_values = _flip_flop_sequence(in_window)

    flip_flop = False
    for index in range(0, max(0, len(flip_values) - 2)):
        a = flip_values[index]
        b = flip_values[index + 1]
        c = flip_values[index + 2]
        if a and b and c and a != b and a == c:
            flip_flop = True
            break

    if flip_flop:
        reason = "flip_flop_default_model"
    elif apply_count >= min_count:
        reason = "apply_churn_threshold"
    else:
        reason = "stable"
    return {
        "triggered": bool(flip_flop or apply_count >= min_count),
        "reason": reason,
        "apply_count_window": apply_count,
        "window_seconds": window,
        "min_applies": min_count,
        "flip_flop_models": flip_values,
        "recent_apply_ids": [str(row.get("id") or "").strip() for row in rows if str(row.get("id") or "").strip()],
    }


__all__ = [
    "AutopilotSafetyStateStore",
    "detect_autopilot_churn",
]

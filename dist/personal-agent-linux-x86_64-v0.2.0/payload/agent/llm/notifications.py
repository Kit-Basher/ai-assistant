from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from zoneinfo import ZoneInfo


_SCHEMA_VERSION = 3
_DEFAULT_MAX_RECENT = 50
_DEFAULT_MAX_ITEMS = 200
_DEFAULT_MAX_AGE_DAYS = 30
_DEFAULT_COMPACT_KEEP_RECENT = 3

_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}\b")
_QUERY_SECRET_RE = re.compile(r"(?i)([?&](?:api[-_]?key|token|access[-_]?token|auth|authorization|key)=)[^&\s]+")
_HEADER_SECRET_RE = re.compile(r"(?i)\b(x-api-key|authorization)\s*[:=]\s*([^\s,;]+)")


def _iso_from_epoch(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    if int(epoch) <= 0:
        return None
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_optional_int(value: Any) -> int | None:
    parsed = _safe_int(value, 0)
    return parsed or None


def sanitize_notification_text(raw: Any) -> str:
    text = str(raw or "")
    if not text:
        return ""
    text = _OPENAI_KEY_RE.sub("[REDACTED]", text)
    text = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", text)
    text = _QUERY_SECRET_RE.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
    text = _HEADER_SECRET_RE.sub(lambda match: f"{match.group(1)}: [REDACTED]", text)
    return text


def _stable_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        redacted = sanitize_notification_text(value)
        return redacted or "null"
    return sanitize_notification_text(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def _message_title(message: str) -> str:
    first = str(message or "").splitlines()[0:1]
    return (first[0] if first else "").strip() or "LLM Autopilot updated configuration"


def _routeable(model_payload: dict[str, Any], provider_payload: dict[str, Any]) -> bool:
    if not bool(provider_payload.get("enabled", True)):
        return False
    if not bool(model_payload.get("enabled", True)):
        return False
    if not bool(model_payload.get("available", True)):
        return False
    capabilities = {
        str(item).strip().lower()
        for item in (model_payload.get("capabilities") or [])
        if str(item).strip()
    }
    return "chat" in capabilities


def _normalize_reasons(reasons: list[Any] | None) -> list[str]:
    normalized = sorted(
        {
            str(item).strip()
            for item in (reasons or [])
            if str(item).strip()
        }
    )
    return normalized


def _normalize_notify_state(raw: dict[str, Any] | None) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    defaults_raw = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
    providers_raw = payload.get("providers") if isinstance(payload.get("providers"), dict) else {}
    models_raw = payload.get("models") if isinstance(payload.get("models"), dict) else {}

    defaults = {
        "routing_mode": defaults_raw.get("routing_mode"),
        "default_provider": defaults_raw.get("default_provider"),
        "default_model": defaults_raw.get("default_model"),
        "allow_remote_fallback": defaults_raw.get("allow_remote_fallback"),
    }

    providers_input: dict[str, dict[str, Any]] = {}
    for provider_id_raw, row in providers_raw.items():
        provider_id = str(provider_id_raw).strip().lower()
        if not provider_id or not isinstance(row, dict):
            continue
        providers_input[provider_id] = row

    providers: dict[str, dict[str, Any]] = {}
    for provider_id in sorted(providers_input.keys()):
        row = providers_input.get(provider_id) if isinstance(providers_input.get(provider_id), dict) else {}
        health = row.get("health") if isinstance(row.get("health"), dict) else {}
        providers[provider_id] = {
            "enabled": bool(row.get("enabled", True)),
            "available": bool(row.get("available", False)),
            "health_status": str(health.get("status") or "unknown").strip().lower() or "unknown",
            "cooldown_until": _safe_optional_int(health.get("cooldown_until")),
            "down_since": _safe_optional_int(health.get("down_since")),
            "failure_streak": max(0, _safe_int(health.get("failure_streak"), 0)),
        }

    models_input: dict[str, dict[str, Any]] = {}
    for model_id_raw, row in models_raw.items():
        model_id = str(model_id_raw).strip()
        if not model_id or not isinstance(row, dict):
            continue
        models_input[model_id] = row

    models: dict[str, dict[str, Any]] = {}
    for model_id in sorted(models_input.keys()):
        row = models_input.get(model_id) if isinstance(models_input.get(model_id), dict) else {}
        health = row.get("health") if isinstance(row.get("health"), dict) else {}
        models[model_id] = {
            "enabled": bool(row.get("enabled", True)),
            "available": bool(row.get("available", False)),
            "routable": bool(row.get("routable", False)),
            "health_status": str(health.get("status") or "unknown").strip().lower() or "unknown",
            "cooldown_until": _safe_optional_int(health.get("cooldown_until")),
            "down_since": _safe_optional_int(health.get("down_since")),
            "failure_streak": max(0, _safe_int(health.get("failure_streak"), 0)),
        }

    return {
        "defaults": defaults,
        "providers": providers,
        "models": models,
    }


class NotificationStore:
    def __init__(
        self,
        path: str | None = None,
        *,
        max_recent: int = _DEFAULT_MAX_RECENT,
        max_items: int = _DEFAULT_MAX_ITEMS,
        max_age_days: int = _DEFAULT_MAX_AGE_DAYS,
        compact: bool = True,
        compact_keep_recent: int = _DEFAULT_COMPACT_KEEP_RECENT,
    ) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()
        self.max_recent = max(1, int(max_recent))
        self.max_items = max(1, int(max_items))
        self.max_age_days = max(0, int(max_age_days))
        self.compact = bool(compact)
        self.compact_keep_recent = max(1, int(compact_keep_recent))
        self.state = self.load()

    @staticmethod
    def default_path() -> str:
        env_path = os.getenv("AUTOPILOT_NOTIFY_STORE_PATH", "").strip()
        if env_path:
            return env_path
        return str(Path.home() / ".local" / "share" / "personal-agent" / "llm_notifications.json")

    @staticmethod
    def empty_state() -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "last_sent_ts": None,
            "last_sent_hash": None,
            "last_seen_hash": None,
            "last_read_hash": None,
            "pruned_count_last_run": 0,
            "pruned_count_total": 0,
            "last_prune_at": None,
            "last_prune_stats": {
                "removed_age": 0,
                "removed_compact": 0,
                "removed_items": 0,
                "removed_total": 0,
                "kept": 0,
            },
            "notifications": [],
        }

    @staticmethod
    def _notifications_sort_key(row: dict[str, Any]) -> tuple[int, str]:
        return int(row.get("ts") or 0), str(row.get("dedupe_hash") or "")

    @classmethod
    def _sorted_notifications(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted([dict(row) for row in rows if isinstance(row, dict)], key=cls._notifications_sort_key)

    @staticmethod
    def _prune_stats_from_raw(raw: dict[str, Any]) -> dict[str, int]:
        stats_raw = raw.get("last_prune_stats") if isinstance(raw.get("last_prune_stats"), dict) else {}
        return {
            "removed_age": max(0, _safe_int(stats_raw.get("removed_age"), 0)),
            "removed_compact": max(0, _safe_int(stats_raw.get("removed_compact"), 0)),
            "removed_items": max(0, _safe_int(stats_raw.get("removed_items"), 0)),
            "removed_total": max(0, _safe_int(stats_raw.get("removed_total"), 0)),
            "kept": max(0, _safe_int(stats_raw.get("kept"), 0)),
        }

    @staticmethod
    def _compaction_key(row: dict[str, Any]) -> tuple[str, str] | None:
        reason = str(row.get("reason") or "").strip().lower()
        if reason == "no_changes":
            return ("reason", "no_changes")
        dedupe_hash = str(row.get("dedupe_hash") or "").strip()
        if dedupe_hash:
            return ("hash", dedupe_hash)
        return None

    def _prune_rows(self, rows: list[dict[str, Any]], *, now_epoch: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
        ordered_rows = self._sorted_notifications(rows)

        removed_age = 0
        if self.max_age_days > 0:
            cutoff_epoch = int(now_epoch) - (int(self.max_age_days) * 86400)
            age_filtered = [row for row in ordered_rows if int(row.get("ts") or 0) >= cutoff_epoch]
            removed_age = max(0, len(ordered_rows) - len(age_filtered))
        else:
            age_filtered = ordered_rows

        removed_compact = 0
        if self.compact:
            groups_kept: dict[tuple[str, str], int] = {}
            kept_desc: list[dict[str, Any]] = []
            for row in reversed(age_filtered):
                key = self._compaction_key(row)
                if key is None:
                    kept_desc.append(row)
                    continue
                already_kept = groups_kept.get(key, 0)
                if already_kept >= self.compact_keep_recent:
                    removed_compact += 1
                    continue
                groups_kept[key] = already_kept + 1
                kept_desc.append(row)
            compacted_rows = list(reversed(kept_desc))
        else:
            compacted_rows = age_filtered

        removed_items = 0
        if self.max_items > 0 and len(compacted_rows) > self.max_items:
            removed_items = len(compacted_rows) - self.max_items
            compacted_rows = compacted_rows[-self.max_items :]

        final_rows = self._sorted_notifications(compacted_rows)
        stats = {
            "removed_age": int(removed_age),
            "removed_compact": int(removed_compact),
            "removed_items": int(removed_items),
            "removed_total": int(removed_age + removed_compact + removed_items),
            "kept": len(final_rows),
        }
        return final_rows, stats

    @staticmethod
    def _unread_count_for_rows(rows_asc: list[dict[str, Any]], last_read_hash: str | None) -> int:
        if not rows_asc:
            return 0
        read_hash = str(last_read_hash or "").strip()
        if not read_hash:
            return len(rows_asc)
        read_index = -1
        for idx, row in enumerate(rows_asc):
            if str(row.get("dedupe_hash") or "").strip() == read_hash:
                read_index = idx
        if read_index < 0:
            return len(rows_asc)
        return max(0, len(rows_asc) - read_index - 1)

    @staticmethod
    def _diff_summary_from_modified_ids(modified_ids: list[Any]) -> dict[str, list[str]]:
        defaults_changed: set[str] = set()
        providers_changed: set[str] = set()
        models_changed: set[str] = set()
        for raw in modified_ids:
            item = str(raw or "").strip()
            if not item or ":" not in item:
                continue
            prefix, suffix = item.split(":", 1)
            prefix = prefix.strip().lower()
            suffix = suffix.strip()
            if not suffix:
                continue
            if prefix == "defaults":
                defaults_changed.add(suffix)
            elif prefix == "provider":
                providers_changed.add(suffix)
            elif prefix == "model":
                models_changed.add(suffix)
        return {
            "defaults_changed": sorted(defaults_changed),
            "providers_changed": sorted(providers_changed),
            "models_changed": sorted(models_changed),
        }

    @staticmethod
    def _suggested_steps(diff_summary: dict[str, Any]) -> list[str]:
        defaults_changed = diff_summary.get("defaults_changed") if isinstance(diff_summary, dict) else []
        providers_changed = diff_summary.get("providers_changed") if isinstance(diff_summary, dict) else []
        models_changed = diff_summary.get("models_changed") if isinstance(diff_summary, dict) else []
        steps: list[str] = []
        if defaults_changed:
            steps.append("Review defaults in the Setup tab to confirm the active provider/model.")
        if providers_changed:
            steps.append("Open Providers and verify connectivity, auth, and enabled states.")
        if models_changed:
            steps.append("Run /llm/health/run to verify model health and routability.")
        if not steps:
            steps.append("Run /llm/health/run to collect a fresh health snapshot.")
        return steps

    @staticmethod
    def _row_summary(row: dict[str, Any] | None) -> dict[str, Any]:
        source = row if isinstance(row, dict) else {}
        message = sanitize_notification_text(str(source.get("message") or "").strip())
        return {
            "hash": str(source.get("dedupe_hash") or "").strip() or None,
            "ts": _safe_int(source.get("ts"), 0) or None,
            "ts_iso": str(source.get("ts_iso") or _iso_from_epoch(_safe_int(source.get("ts"), 0) or None) or "").strip() or None,
            "title": _message_title(message),
            "outcome": str(source.get("outcome") or "").strip() or None,
            "reason": sanitize_notification_text(str(source.get("reason") or "").strip()) or None,
            "delivered_to": str(source.get("delivered_to") or "").strip() or None,
        }

    def _normalize(
        self,
        raw: dict[str, Any],
        *,
        now_epoch: int | None = None,
        force_prune_meta: bool = False,
    ) -> dict[str, Any]:
        output = self.empty_state()
        output["schema_version"] = _SCHEMA_VERSION
        output["last_sent_ts"] = _safe_int(raw.get("last_sent_ts"), 0) or None
        last_hash = str(raw.get("last_sent_hash") or "").strip()
        output["last_sent_hash"] = last_hash or None
        output["last_seen_hash"] = str(raw.get("last_seen_hash") or "").strip() or None
        output["last_read_hash"] = str(raw.get("last_read_hash") or "").strip() or None

        notifications_raw = raw.get("notifications") if isinstance(raw.get("notifications"), list) else []
        normalized_rows: list[dict[str, Any]] = []
        for row in notifications_raw:
            if not isinstance(row, dict):
                continue
            ts = _safe_int(row.get("ts"), 0)
            dedupe_hash = str(row.get("dedupe_hash") or "").strip()
            message = sanitize_notification_text(str(row.get("message") or "").strip())
            if not ts or not dedupe_hash or not message:
                continue
            normalized_rows.append(
                {
                    "ts": ts,
                    "ts_iso": str(row.get("ts_iso") or _iso_from_epoch(ts) or ""),
                    "message": message,
                    "dedupe_hash": dedupe_hash,
                    "delivered_to": str(row.get("delivered_to") or "none"),
                    "deferred": bool(row.get("deferred", False)),
                    "outcome": str(row.get("outcome") or "skipped"),
                    "reason": sanitize_notification_text(str(row.get("reason") or "")),
                    "modified_ids": sorted(
                        {
                            str(item).strip()
                            for item in (row.get("modified_ids") or [])
                            if str(item).strip()
                        }
                    ),
                }
            )
        prune_now = int(time.time()) if now_epoch is None else int(now_epoch)
        pruned_rows, prune_stats = self._prune_rows(normalized_rows, now_epoch=prune_now)
        output["notifications"] = pruned_rows

        previous_pruned_total = max(0, _safe_int(raw.get("pruned_count_total"), 0))
        removed_total = int(prune_stats.get("removed_total") or 0)
        if removed_total > 0 or bool(force_prune_meta):
            output["pruned_count_last_run"] = removed_total
            output["pruned_count_total"] = previous_pruned_total + removed_total
            output["last_prune_at"] = prune_now
            output["last_prune_stats"] = prune_stats
        else:
            output["pruned_count_last_run"] = max(0, _safe_int(raw.get("pruned_count_last_run"), 0))
            output["pruned_count_total"] = previous_pruned_total
            output["last_prune_at"] = _safe_int(raw.get("last_prune_at"), 0) or None
            output["last_prune_stats"] = self._prune_stats_from_raw(raw)

        latest_hash = ""
        if pruned_rows:
            latest_hash = str(pruned_rows[-1].get("dedupe_hash") or "").strip()
        output["last_seen_hash"] = latest_hash or output["last_seen_hash"] or None
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
        raw_canonical = json.dumps(raw, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        normalized_canonical = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if raw_canonical != normalized_canonical:
            try:
                self._write(normalized)
            except OSError:
                pass
        return normalized

    def _write(self, normalized: dict[str, Any]) -> None:
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

    def save(self, state: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(state if isinstance(state, dict) else {})
        self._write(normalized)
        self.state = normalized
        return normalized

    def append(
        self,
        *,
        ts: int,
        message: str,
        dedupe_hash: str,
        delivered_to: str,
        deferred: bool,
        outcome: str,
        reason: str,
        modified_ids: list[str] | None = None,
        mark_sent: bool = False,
    ) -> dict[str, Any]:
        current = dict(self.state if isinstance(self.state, dict) else self.empty_state())
        rows = current.get("notifications") if isinstance(current.get("notifications"), list) else []
        next_row = {
            "ts": int(ts),
            "ts_iso": _iso_from_epoch(int(ts)) or "",
            "message": sanitize_notification_text(str(message or "").strip()),
            "dedupe_hash": str(dedupe_hash or "").strip(),
            "delivered_to": str(delivered_to or "none"),
            "deferred": bool(deferred),
            "outcome": str(outcome or "skipped"),
            "reason": sanitize_notification_text(str(reason or "")),
            "modified_ids": sorted(
                {
                    str(item).strip()
                    for item in (modified_ids or [])
                    if str(item).strip()
                }
            ),
        }
        rows = [row for row in rows if isinstance(row, dict)]
        rows.append(next_row)
        current["notifications"] = rows
        if mark_sent:
            current["last_sent_ts"] = int(ts)
            current["last_sent_hash"] = str(dedupe_hash or "").strip() or None
        return self.save(current)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.state.get("notifications") if isinstance(self.state.get("notifications"), list) else []
        newest_first = list(reversed([dict(row) for row in rows if isinstance(row, dict)]))
        max_limit = min(self.max_recent, max(1, int(limit)))
        return newest_first[:max_limit]

    def count_newer_than(self, epoch: int) -> int:
        ts = int(epoch or 0)
        rows = self.state.get("notifications") if isinstance(self.state.get("notifications"), list) else []
        return sum(1 for row in rows if isinstance(row, dict) and int(row.get("ts") or 0) > ts)

    def latest_notification_summary(self) -> dict[str, Any]:
        rows = self.state.get("notifications") if isinstance(self.state.get("notifications"), list) else []
        latest = rows[-1] if rows else None
        return self._row_summary(latest)

    def mark_read(self, read_hash: str) -> dict[str, Any]:
        target_hash = str(read_hash or "").strip()
        if not target_hash:
            return {"ok": False, "error": "hash_required"}
        rows = self.state.get("notifications") if isinstance(self.state.get("notifications"), list) else []
        known = any(str(row.get("dedupe_hash") or "").strip() == target_hash for row in rows if isinstance(row, dict))
        if not known:
            return {"ok": False, "error": "hash_not_found"}
        current = dict(self.state if isinstance(self.state, dict) else self.empty_state())
        current["last_read_hash"] = target_hash
        saved = self.save(current)
        saved_rows = saved.get("notifications") if isinstance(saved.get("notifications"), list) else []
        unread_count = self._unread_count_for_rows(
            [dict(row) for row in saved_rows if isinstance(row, dict)],
            str(saved.get("last_read_hash") or "").strip() or None,
        )
        return {"ok": True, "last_read_hash": target_hash, "unread_count": unread_count}

    def last_change(self) -> dict[str, Any] | None:
        skip_reasons = {
            "no_changes",
            "skipped_rate_limit",
            "skipped_dedupe",
            "skipped_quiet_hours",
            "rate_limited",
            "dedupe_hash_match",
            "quiet_hours",
            "quiet_hours_deferred",
        }
        for row in self.recent(limit=self.max_recent):
            if not isinstance(row, dict):
                continue
            reason = str(row.get("reason") or "").strip().lower()
            if reason in skip_reasons:
                continue
            summary = self._row_summary(row)
            body = sanitize_notification_text(str(row.get("message") or "").strip())
            diff_summary = self._diff_summary_from_modified_ids(row.get("modified_ids") or [])
            return {
                "hash": summary.get("hash"),
                "ts": summary.get("ts"),
                "ts_iso": summary.get("ts_iso"),
                "title": summary.get("title"),
                "body": body,
                "outcome": summary.get("outcome"),
                "reason": summary.get("reason"),
                "delivered_to": summary.get("delivered_to"),
                "diff_summary": diff_summary,
                "suggested_next_steps": self._suggested_steps(diff_summary),
            }
        return None

    def status(self) -> dict[str, Any]:
        rows = self.state.get("notifications") if isinstance(self.state.get("notifications"), list) else []
        rows_asc = [dict(row) for row in rows if isinstance(row, dict)]
        last_read_hash = str(self.state.get("last_read_hash") or "").strip() or None
        last_prune_at = _safe_int(self.state.get("last_prune_at"), 0) or None
        return {
            "stored_count": len(rows_asc),
            "last_seen_hash": str(self.state.get("last_seen_hash") or "").strip() or None,
            "last_read_hash": last_read_hash,
            "unread_count": self._unread_count_for_rows(rows_asc, last_read_hash),
            "pruned_count_last_run": max(0, _safe_int(self.state.get("pruned_count_last_run"), 0)),
            "pruned_count_total": max(0, _safe_int(self.state.get("pruned_count_total"), 0)),
            "last_prune_at": last_prune_at,
            "last_prune_at_iso": _iso_from_epoch(last_prune_at),
            "retention": {
                "max_items": int(self.max_items),
                "max_age_days": int(self.max_age_days),
                "compact": bool(self.compact),
                "compact_keep_recent": int(self.compact_keep_recent),
            },
        }

    def prune_now(self, *, now_epoch: int | None = None) -> dict[str, Any]:
        current_state = dict(self.state if isinstance(self.state, dict) else self.empty_state())
        normalized = self._normalize(
            current_state,
            now_epoch=int(time.time()) if now_epoch is None else int(now_epoch),
            force_prune_meta=True,
        )
        self._write(normalized)
        self.state = normalized
        stats = self._prune_stats_from_raw(normalized)
        return {
            "removed_total": int(stats.get("removed_total") or 0),
            "removed_age": int(stats.get("removed_age") or 0),
            "removed_compact": int(stats.get("removed_compact") or 0),
            "removed_items": int(stats.get("removed_items") or 0),
            "kept": int(stats.get("kept") or 0),
            "stored_count": len(normalized.get("notifications") if isinstance(normalized.get("notifications"), list) else []),
            "last_seen_hash": str(normalized.get("last_seen_hash") or "").strip() or None,
            "last_prune_at": _safe_int(normalized.get("last_prune_at"), 0) or None,
            "last_prune_at_iso": _iso_from_epoch(_safe_int(normalized.get("last_prune_at"), 0) or None),
        }


def build_notification_from_diff(
    before: dict[str, Any],
    after: dict[str, Any],
    reasons: list[str] | None,
    modified_ids: list[str] | None,
) -> tuple[str, str, list[str]]:
    before_doc = before if isinstance(before, dict) else {}
    after_doc = after if isinstance(after, dict) else {}
    change_lines: list[str] = []

    before_defaults = before_doc.get("defaults") if isinstance(before_doc.get("defaults"), dict) else {}
    after_defaults = after_doc.get("defaults") if isinstance(after_doc.get("defaults"), dict) else {}
    for field in ("routing_mode", "default_provider", "default_model", "allow_remote_fallback"):
        before_value = before_defaults.get(field)
        after_value = after_defaults.get(field)
        if before_value == after_value:
            continue
        change_lines.append(
            f"Defaults: {field} -> {_stable_value(after_value)} (was {_stable_value(before_value)})"
        )

    before_providers = before_doc.get("providers") if isinstance(before_doc.get("providers"), dict) else {}
    after_providers = after_doc.get("providers") if isinstance(after_doc.get("providers"), dict) else {}
    for provider_id in sorted(set(before_providers.keys()) | set(after_providers.keys())):
        before_provider = before_providers.get(provider_id) if isinstance(before_providers.get(provider_id), dict) else {}
        after_provider = after_providers.get(provider_id) if isinstance(after_providers.get(provider_id), dict) else {}
        for field in ("enabled", "local", "base_url", "chat_path", "default_headers", "default_query_params"):
            before_value = before_provider.get(field)
            after_value = after_provider.get(field)
            if before_value == after_value:
                continue
            if field == "enabled":
                if bool(after_value):
                    change_lines.append(f"Provider {provider_id}: enabled (was disabled)")
                else:
                    change_lines.append(f"Provider {provider_id}: disabled (was enabled)")
            else:
                change_lines.append(
                    f"Provider {provider_id}: {field} -> {_stable_value(after_value)} (was {_stable_value(before_value)})"
                )

    for model_id in sorted(
        set(
            (after_doc.get("models") if isinstance(after_doc.get("models"), dict) else {}).keys()
        )
        | set((before_doc.get("models") if isinstance(before_doc.get("models"), dict) else {}).keys())
    ):
        before_models = before_doc.get("models") if isinstance(before_doc.get("models"), dict) else {}
        after_models = after_doc.get("models") if isinstance(after_doc.get("models"), dict) else {}
        before_model = before_models.get(model_id) if isinstance(before_models.get(model_id), dict) else {}
        after_model = after_models.get(model_id) if isinstance(after_models.get(model_id), dict) else {}
        for field in ("enabled", "available"):
            before_value = before_model.get(field)
            after_value = after_model.get(field)
            if before_value == after_value:
                continue
            if field == "available" and (before_value is True and after_value is False):
                change_lines.append(f"Model {model_id}: marked unroutable (available=false; was true)")
            else:
                change_lines.append(
                    f"Model {model_id}: {field} -> {_stable_value(after_value)} (was {_stable_value(before_value)})"
                )

        before_provider_id = str(before_model.get("provider") or "").strip().lower()
        after_provider_id = str(after_model.get("provider") or "").strip().lower()
        before_provider = (
            before_providers.get(before_provider_id)
            if isinstance(before_providers.get(before_provider_id), dict)
            else {}
        )
        after_provider = (
            after_providers.get(after_provider_id)
            if isinstance(after_providers.get(after_provider_id), dict)
            else {}
        )
        before_routable = _routeable(before_model, before_provider)
        after_routable = _routeable(after_model, after_provider)
        if before_routable != after_routable:
            change_lines.append(
                f"Model {model_id}: routable -> {_stable_value(after_routable)} (was {_stable_value(before_routable)})"
            )

    if not change_lines:
        return "", "", []

    normalized_reasons = _normalize_reasons(reasons)
    reason_line = normalized_reasons[0] if normalized_reasons else "reconciled deterministic configuration changes"
    reason_line = sanitize_notification_text(reason_line)
    modified = sorted(
        {
            str(item).strip()
            for item in (modified_ids or [])
            if str(item).strip()
        }
    )
    if not modified:
        modified = sorted(change_lines)
    safe_changes = [sanitize_notification_text(line) for line in sorted(change_lines)]
    message_lines = ["LLM Autopilot updated configuration"]
    message_lines.extend([f"- {line}" for line in safe_changes])
    message_lines.append(f"Reason: {reason_line}")
    message = sanitize_notification_text("\n".join(message_lines))

    payload = {
        "title": "LLM Autopilot updated configuration",
        "changes": safe_changes,
        "reason": reason_line,
        "modified_ids": modified,
    }
    dedupe_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return message, dedupe_hash, safe_changes


def build_notification_from_state_diff(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    reasons: list[str] | None,
    *,
    extra_changes: list[str] | None = None,
) -> dict[str, Any]:
    before_state = _normalize_notify_state(before)
    after_state = _normalize_notify_state(after)
    change_lines: list[str] = []
    changed_defaults: set[str] = set()
    changed_providers: set[str] = set()
    changed_models: set[str] = set()
    modified_ids: set[str] = set()

    before_defaults = before_state.get("defaults") if isinstance(before_state.get("defaults"), dict) else {}
    after_defaults = after_state.get("defaults") if isinstance(after_state.get("defaults"), dict) else {}
    for field in ("routing_mode", "default_provider", "default_model", "allow_remote_fallback"):
        before_value = before_defaults.get(field)
        after_value = after_defaults.get(field)
        if before_value == after_value:
            continue
        changed_defaults.add(field)
        modified_ids.add(f"defaults:{field}")
        change_lines.append(
            f"Defaults: {field} -> {_stable_value(after_value)} (was {_stable_value(before_value)})"
        )

    provider_fields = (
        ("enabled", "enabled"),
        ("available", "available"),
        ("health_status", "health.status"),
        ("cooldown_until", "health.cooldown_until"),
        ("down_since", "health.down_since"),
        ("failure_streak", "health.failure_streak"),
    )
    before_providers = before_state.get("providers") if isinstance(before_state.get("providers"), dict) else {}
    after_providers = after_state.get("providers") if isinstance(after_state.get("providers"), dict) else {}
    for provider_id in sorted(set(before_providers.keys()) | set(after_providers.keys())):
        before_row = before_providers.get(provider_id) if isinstance(before_providers.get(provider_id), dict) else {}
        after_row = after_providers.get(provider_id) if isinstance(after_providers.get(provider_id), dict) else {}
        provider_changed = False
        for field_key, field_label in provider_fields:
            before_value = before_row.get(field_key)
            after_value = after_row.get(field_key)
            if before_value == after_value:
                continue
            provider_changed = True
            change_lines.append(
                f"Provider {provider_id}: {field_label} -> {_stable_value(after_value)} (was {_stable_value(before_value)})"
            )
        if provider_changed:
            changed_providers.add(provider_id)
            modified_ids.add(f"provider:{provider_id}")

    model_fields = (
        ("enabled", "enabled"),
        ("available", "available"),
        ("routable", "routable"),
        ("health_status", "health.status"),
        ("cooldown_until", "health.cooldown_until"),
        ("down_since", "health.down_since"),
        ("failure_streak", "health.failure_streak"),
    )
    before_models = before_state.get("models") if isinstance(before_state.get("models"), dict) else {}
    after_models = after_state.get("models") if isinstance(after_state.get("models"), dict) else {}
    for model_id in sorted(set(before_models.keys()) | set(after_models.keys())):
        before_row = before_models.get(model_id) if isinstance(before_models.get(model_id), dict) else {}
        after_row = after_models.get(model_id) if isinstance(after_models.get(model_id), dict) else {}
        model_changed = False
        for field_key, field_label in model_fields:
            before_value = before_row.get(field_key)
            after_value = after_row.get(field_key)
            if before_value == after_value:
                continue
            model_changed = True
            change_lines.append(
                f"Model {model_id}: {field_label} -> {_stable_value(after_value)} (was {_stable_value(before_value)})"
            )
        if model_changed:
            changed_models.add(model_id)
            modified_ids.add(f"model:{model_id}")

    if extra_changes:
        for item in sorted({str(raw).strip() for raw in extra_changes if str(raw).strip()}):
            change_lines.append(sanitize_notification_text(item))

    if not change_lines:
        return {
            "message": "",
            "dedupe_hash": "",
            "changes": [],
            "counts": {"defaults": 0, "providers": 0, "models": 0},
            "modified_ids": [],
            "reason": "",
            "title": "LLM Autopilot updated configuration",
        }

    normalized_reasons = _normalize_reasons(reasons)
    reason_line = normalized_reasons[0] if normalized_reasons else "reconciled deterministic configuration changes"
    reason_line = sanitize_notification_text(reason_line)
    sorted_changes = [sanitize_notification_text(line) for line in sorted(change_lines)]
    sorted_modified_ids = sorted(modified_ids)
    message_lines = ["LLM Autopilot updated configuration"]
    message_lines.extend([f"- {line}" for line in sorted_changes])
    message_lines.append(f"Reason: {reason_line}")
    message = sanitize_notification_text("\n".join(message_lines))
    counts = {
        "defaults": len(changed_defaults),
        "providers": len(changed_providers),
        "models": len(changed_models),
    }
    payload = {
        "title": "LLM Autopilot updated configuration",
        "changes": sorted_changes,
        "reason": reason_line,
        "modified_ids": sorted_modified_ids,
        "counts": counts,
    }
    dedupe_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "message": message,
        "dedupe_hash": dedupe_hash,
        "changes": sorted_changes,
        "counts": counts,
        "modified_ids": sorted_modified_ids,
        "reason": reason_line,
        "title": "LLM Autopilot updated configuration",
    }


def should_send(
    *,
    now_epoch: int,
    last_sent_ts: int | None,
    last_sent_hash: str | None,
    message_hash: str,
    enabled: bool,
    rate_limit_seconds: int,
    dedupe_window_seconds: int,
    quiet_start_hour: int | None,
    quiet_end_hour: int | None,
    timezone_name: str,
) -> dict[str, Any]:
    now_value = int(now_epoch)
    if not bool(enabled):
        return {"send": False, "deferred": False, "reason": "disabled"}

    last_ts = int(last_sent_ts or 0)
    last_hash = str(last_sent_hash or "").strip()
    if (
        dedupe_window_seconds > 0
        and last_ts > 0
        and last_hash
        and last_hash == str(message_hash or "").strip()
        and (now_value - last_ts) < int(dedupe_window_seconds)
    ):
        return {"send": False, "deferred": False, "reason": "dedupe_hash_match"}

    if rate_limit_seconds > 0 and last_ts > 0 and (now_value - last_ts) < int(rate_limit_seconds):
        return {"send": False, "deferred": False, "reason": "rate_limited"}

    if _is_in_quiet_hours(now_value, quiet_start_hour, quiet_end_hour, timezone_name):
        return {"send": False, "deferred": True, "reason": "quiet_hours"}

    return {"send": True, "deferred": False, "reason": "send"}


def _is_in_quiet_hours(
    now_epoch: int,
    quiet_start_hour: int | None,
    quiet_end_hour: int | None,
    timezone_name: str,
) -> bool:
    if quiet_start_hour is None or quiet_end_hour is None:
        return False
    if quiet_start_hour == quiet_end_hour:
        return False

    try:
        tz = ZoneInfo(str(timezone_name or "UTC"))
    except Exception:
        tz = timezone.utc
    local_hour = datetime.fromtimestamp(int(now_epoch), tz=tz).hour

    start = int(quiet_start_hour)
    end = int(quiet_end_hour)
    if start < end:
        return start <= local_hour < end
    return local_hour >= start or local_hour < end

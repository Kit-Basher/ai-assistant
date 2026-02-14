from __future__ import annotations

from collections import OrderedDict
from typing import Any


PREF_DEFAULTS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("show_next_action", "on"),
        ("show_summary", "on"),
        ("terse_mode", "off"),
        ("commands_in_codeblock", "off"),
    ]
)

ALLOWED_PREF_KEYS = tuple(PREF_DEFAULTS.keys())
_TRUE_VALUES = {"1", "true", "on", "yes"}
_FALSE_VALUES = {"0", "false", "off", "no"}


def _normalize_pref_value(value: str) -> str:
    lowered = (value or "").strip().lower()
    if lowered in _TRUE_VALUES:
        return "on"
    if lowered in _FALSE_VALUES:
        return "off"
    return lowered


def get_pref(db: Any, key: str) -> str | None:
    if key not in PREF_DEFAULTS:
        return None
    fn = getattr(db, "get_user_pref", None)
    if callable(fn):
        value = fn(key)
        if isinstance(value, str) and value.strip():
            normalized = _normalize_pref_value(value)
            if normalized in {"on", "off"}:
                return normalized
    return PREF_DEFAULTS[key]


def set_pref(db: Any, key: str, value: str) -> None:
    if key not in PREF_DEFAULTS:
        raise ValueError(f"Unsupported preference key: {key}")
    normalized = _normalize_pref_value(value)
    if normalized not in {"on", "off"}:
        raise ValueError(f"Unsupported preference value: {value}")
    fn = getattr(db, "set_user_pref", None)
    if not callable(fn):
        raise RuntimeError("DB does not support user preferences")
    fn(key, normalized)


def list_prefs(db: Any) -> dict[str, str]:
    output: "OrderedDict[str, str]" = OrderedDict()
    for key in ALLOWED_PREF_KEYS:
        value = get_pref(db, key)
        output[key] = value if value in {"on", "off"} else PREF_DEFAULTS[key]
    return dict(output)


def get_bool_pref(db: Any, key: str, default: bool) -> bool:
    value = get_pref(db, key)
    if value is None:
        return bool(default)
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return bool(default)


def _get_thread_pref(db: Any, thread_id: str, key: str) -> str | None:
    fn = getattr(db, "get_thread_pref", None)
    if not callable(fn):
        return None
    value = fn(thread_id, key)
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = _normalize_pref_value(value)
    if normalized not in {"on", "off"}:
        return None
    return normalized


def get_pref_effective(db: Any, thread_id: str | None, key: str, default: Any) -> Any:
    if key not in PREF_DEFAULTS:
        return default
    if isinstance(thread_id, str) and thread_id.strip():
        thread_value = _get_thread_pref(db, thread_id.strip(), key)
        if thread_value is not None:
            if isinstance(default, bool):
                return thread_value == "on"
            return thread_value
    global_value = get_pref(db, key)
    if global_value is None:
        return default
    if isinstance(default, bool):
        return global_value == "on"
    return global_value


def get_pref_effective_with_source(db: Any, thread_id: str | None, key: str) -> tuple[str, str]:
    if key not in PREF_DEFAULTS:
        return "off", "default"
    if isinstance(thread_id, str) and thread_id.strip():
        thread_value = _get_thread_pref(db, thread_id.strip(), key)
        if thread_value is not None:
            return thread_value, "thread"
    fn = getattr(db, "get_user_pref", None)
    if callable(fn):
        raw = fn(key)
        if isinstance(raw, str) and raw.strip():
            normalized = _normalize_pref_value(raw)
            if normalized in {"on", "off"}:
                return normalized, "global"
    return PREF_DEFAULTS[key], "default"


def set_thread_pref(db: Any, thread_id: str, key: str, value: str) -> None:
    if key not in PREF_DEFAULTS:
        raise ValueError(f"Unsupported preference key: {key}")
    normalized = _normalize_pref_value(value)
    if normalized not in {"on", "off"}:
        raise ValueError(f"Unsupported preference value: {value}")
    fn = getattr(db, "set_thread_pref", None)
    if not callable(fn):
        raise RuntimeError("DB does not support thread preferences")
    fn(thread_id, key, normalized)


def list_thread_prefs(db: Any, thread_id: str) -> dict[str, str]:
    output: "OrderedDict[str, str]" = OrderedDict()
    fn = getattr(db, "list_thread_prefs", None)
    raw: dict[str, str] = fn(thread_id) if callable(fn) else {}
    for key in ALLOWED_PREF_KEYS:
        value = raw.get(key) if isinstance(raw, dict) else None
        normalized = _normalize_pref_value(str(value)) if isinstance(value, str) else None
        if normalized in {"on", "off"}:
            output[key] = normalized
    return dict(output)


def reset_prefs(db: Any) -> None:
    fn = getattr(db, "clear_user_prefs", None)
    if callable(fn):
        fn()


def reset_thread_prefs(db: Any, thread_id: str) -> None:
    fn = getattr(db, "clear_thread_prefs", None)
    if callable(fn):
        fn(thread_id)

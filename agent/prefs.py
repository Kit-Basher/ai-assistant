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


def reset_prefs(db: Any) -> None:
    fn = getattr(db, "clear_user_prefs", None)
    if callable(fn):
        fn()


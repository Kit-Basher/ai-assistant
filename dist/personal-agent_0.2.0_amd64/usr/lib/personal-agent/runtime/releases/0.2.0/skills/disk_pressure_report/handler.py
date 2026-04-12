from __future__ import annotations

import os
from typing import Any

from agent.cards import normalize_card


def _bytes_to_human(num_bytes: int) -> str:
    if num_bytes < 0:
        return "0B"
    units = ["B", "K", "M", "G", "T", "P", "E"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}B"
            formatted = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{formatted}{unit}"
        value /= 1024
    return f"{int(value)}B"


def _largest_files(paths: list[str], limit: int = 5, scan_limit: int = 2000) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    seen = 0
    for base in paths:
        if not os.path.exists(base):
            continue
        for root, _dirs, files in os.walk(base):
            for name in files:
                path = os.path.join(root, name)
                try:
                    size = int(os.path.getsize(path))
                except OSError:
                    continue
                found.append((path, size))
                seen += 1
                if seen >= scan_limit:
                    break
            if seen >= scan_limit:
                break
        if seen >= scan_limit:
            break
    found.sort(key=lambda item: item[1], reverse=True)
    return found[:limit]


def disk_pressure_report(context: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    db = context.get("db") if context else None
    if not db:
        text = "Database not available."
        return {"status": "blocked", "text": text, "message": text}

    root_latest = db.get_latest_dir_size_samples("root_top") or {}
    home_latest = db.get_latest_dir_size_samples("home_top") or {}
    root_prev = db.get_previous_dir_size_samples("root_top", root_latest.get("taken_at")) if root_latest else {}
    home_prev = db.get_previous_dir_size_samples("home_top", home_latest.get("taken_at")) if home_latest else {}
    growth: list[tuple[str, int]] = []
    for current, prev in ((root_latest, root_prev), (home_latest, home_prev)):
        prev_map = {p: b for p, b in (prev or {}).get("samples", [])}
        for path, bytes_val in (current or {}).get("samples", []):
            if path in prev_map:
                delta = int(bytes_val) - int(prev_map[path])
                if delta > 0:
                    growth.append((path, delta))
    growth.sort(key=lambda item: item[1], reverse=True)

    preferred_paths_raw = db.get_preference("important_paths")
    scan_paths = [os.path.expanduser("~")]
    if preferred_paths_raw:
        try:
            import json
            parsed = json.loads(preferred_paths_raw)
            if isinstance(parsed, list):
                scan_paths = [str(p) for p in parsed if str(p).strip()][:3] or scan_paths
        except Exception:
            pass
    largest = _largest_files(scan_paths)

    top_dirs = (root_latest.get("samples", [])[:3] if root_latest else []) + (home_latest.get("samples", [])[:3] if home_latest else [])
    cards = [
        normalize_card(
            {
                "title": "Disk pressure summary",
                "lines": [f"{path}: {_bytes_to_human(int(size))}" for path, size in top_dirs] or ["No directory samples found."],
                "severity": "ok",
            },
            0,
        ),
        normalize_card(
            {
                "title": "Top growing paths",
                "lines": [f"{path}: +{_bytes_to_human(delta)}" for path, delta in growth[:5]] or ["No growth data available."],
                "severity": "warn" if growth else "ok",
            },
            1,
        ),
        normalize_card(
            {
                "title": "Largest files (bounded scan)",
                "lines": [f"{path}: {_bytes_to_human(size)}" for path, size in largest] or ["No files scanned."],
                "severity": "ok",
            },
            2,
        ),
    ]

    return {
        "status": "ok",
        "text": "Disk pressure report ready.",
        "payload": {"top_dirs": top_dirs, "growth": growth[:5], "largest_files": largest},
        "cards_payload": {"cards": cards, "raw_available": True},
    }

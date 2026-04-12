from __future__ import annotations

import os
import shutil
import socket
from datetime import datetime
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from memory.db import MemoryDB

TOP_N_DEFAULT = 10
DEFAULT_MOUNTPOINTS = ["/", "/data", "/data2"]
EXCLUDE_ROOT_DIRS = {"/data", "/data2"}


@dataclass
class ScanStats:
    dirs_scanned: int = 0
    errors_skipped: int = 0


def _now_local_iso(tz_name: str) -> str:
    tz = ZoneInfo(tz_name)
    return datetime.now(tz).isoformat(timespec="seconds")


def _read_mount_table() -> dict[str, str]:
    mounts: dict[str, str] = {}
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 2:
                    continue
                device, mountpoint = parts[0], parts[1]
                mounts[mountpoint] = device
    except OSError:
        return mounts
    return mounts


def _dir_size(path: str, root_dev: int, stats: ScanStats) -> int:
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            stats.dirs_scanned += 1
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        stats.errors_skipped += 1
                        continue

                    if entry.is_dir(follow_symlinks=False):
                        if stat.st_dev != root_dev:
                            continue
                        stack.append(entry.path)
                        continue

                    if entry.is_file(follow_symlinks=False):
                        total += int(stat.st_size)
        except OSError:
            stats.errors_skipped += 1
            continue
    return total


def _top_level_dir_sizes(
    base_path: str,
    *,
    root_dev: int,
    exclude_paths: set[str] | None = None,
    top_n: int = TOP_N_DEFAULT,
) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    stats = ScanStats()
    exclude_paths = exclude_paths or set()
    try:
        with os.scandir(base_path) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                path = entry.path
                if path in exclude_paths:
                    continue
                try:
                    stat = entry.stat(follow_symlinks=False)
                except OSError:
                    stats.errors_skipped += 1
                    continue
                if stat.st_dev != root_dev:
                    continue
                size = _dir_size(path, root_dev, stats)
                results.append((path, size))
    except OSError:
        stats.errors_skipped += 1
        return [], stats

    results.sort(key=lambda item: item[1], reverse=True)
    return results[: max(0, int(top_n))], stats


def collect_and_persist_snapshot(
    db: MemoryDB,
    *,
    timezone: str,
    home_path: str | None = None,
    mountpoints: list[str] | None = None,
    top_n: int = TOP_N_DEFAULT,
) -> dict[str, Any]:
    taken_at = _now_local_iso(timezone)
    snapshot_local_date = taken_at.split("T")[0]
    hostname = socket.gethostname()
    mountpoints = mountpoints or list(DEFAULT_MOUNTPOINTS)

    mount_table = _read_mount_table()
    mount_rows: list[dict[str, Any]] = []

    for mountpoint in mountpoints:
        if not os.path.exists(mountpoint):
            continue
        try:
            usage = shutil.disk_usage(mountpoint)
        except OSError:
            continue
        filesystem = mount_table.get(mountpoint)
        db.insert_disk_snapshot(
            taken_at=taken_at,
            snapshot_local_date=snapshot_local_date,
            hostname=hostname,
            mountpoint=mountpoint,
            filesystem=filesystem,
            total_bytes=int(usage.total),
            used_bytes=int(usage.used),
            free_bytes=int(usage.free),
        )
        mount_rows.append(
            {
                "mountpoint": mountpoint,
                "filesystem": filesystem,
                "total_bytes": int(usage.total),
                "used_bytes": int(usage.used),
                "free_bytes": int(usage.free),
            }
        )

    root_dev = os.stat("/").st_dev
    root_top, root_stats = _top_level_dir_sizes(
        "/",
        root_dev=root_dev,
        exclude_paths=EXCLUDE_ROOT_DIRS,
        top_n=top_n,
    )
    db.insert_dir_size_samples(taken_at, "root_top", root_top)
    db.insert_storage_scan_stats(
        taken_at, "root_top", root_stats.dirs_scanned, root_stats.errors_skipped
    )

    resolved_home = os.path.abspath(os.path.expanduser(home_path or "~"))
    home_top: list[tuple[str, int]] = []
    home_stats = ScanStats()
    try:
        home_dev = os.stat(resolved_home).st_dev
        home_top, home_stats = _top_level_dir_sizes(
            resolved_home,
            root_dev=home_dev,
            exclude_paths=set(),
            top_n=top_n,
        )
    except OSError:
        home_top = []
    db.insert_dir_size_samples(taken_at, "home_top", home_top)
    db.insert_storage_scan_stats(
        taken_at, "home_top", home_stats.dirs_scanned, home_stats.errors_skipped
    )

    return {
        "taken_at": taken_at,
        "hostname": hostname,
        "mounts": mount_rows,
        "root_top": root_top,
        "home_top": home_top,
        "root_stats": {
            "dirs_scanned": root_stats.dirs_scanned,
            "errors_skipped": root_stats.errors_skipped,
        },
        "home_stats": {
            "dirs_scanned": home_stats.dirs_scanned,
            "errors_skipped": home_stats.errors_skipped,
        },
        "top_n": int(top_n),
    }

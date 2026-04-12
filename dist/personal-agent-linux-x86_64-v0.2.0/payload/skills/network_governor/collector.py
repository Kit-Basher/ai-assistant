from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from memory.db import MemoryDB


@dataclass
class IfaceSample:
    name: str
    state: str
    rx_bytes: int
    tx_bytes: int
    rx_errors: int
    tx_errors: int


def _now_local_iso(tz_name: str) -> str:
    tz = ZoneInfo(tz_name)
    return datetime.now(tz).isoformat(timespec="seconds")


def _read_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def _read_operstate(iface: str) -> str:
    content = _read_file(f"/sys/class/net/{iface}/operstate")
    if not content:
        return "unknown"
    return content.strip()


def _parse_net_dev() -> list[IfaceSample]:
    content = _read_file("/proc/net/dev")
    if not content:
        return []
    lines = content.splitlines()
    if len(lines) < 3:
        return []
    samples: list[IfaceSample] = []
    for line in lines[2:]:
        if ":" not in line:
            continue
        name_part, data_part = line.split(":", 1)
        iface = name_part.strip()
        fields = data_part.split()
        if len(fields) < 16:
            continue
        try:
            rx_bytes = int(fields[0])
            rx_errors = int(fields[2])
            tx_bytes = int(fields[8])
            tx_errors = int(fields[10])
        except ValueError:
            continue
        state = _read_operstate(iface)
        samples.append(
            IfaceSample(
                name=iface,
                state=state,
                rx_bytes=rx_bytes,
                tx_bytes=tx_bytes,
                rx_errors=rx_errors,
                tx_errors=tx_errors,
            )
        )
    return samples


def _hex_to_ip(hex_str: str) -> str:
    try:
        val = int(hex_str, 16)
    except ValueError:
        return ""
    return ".".join(str((val >> (8 * i)) & 0xFF) for i in range(4))


def _parse_default_route() -> tuple[str, str]:
    content = _read_file("/proc/net/route")
    if not content:
        return "", ""
    lines = content.splitlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        iface, dest, gateway = parts[0], parts[1], parts[2]
        if dest != "00000000":
            continue
        return iface, _hex_to_ip(gateway)
    return "", ""


def _parse_resolv_conf() -> list[str]:
    content = _read_file("/etc/resolv.conf")
    if not content:
        return []
    nameservers: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("nameserver"):
            parts = line.split()
            if len(parts) >= 2:
                nameservers.append(parts[1])
    return nameservers


def collect_and_persist_snapshot(
    db: MemoryDB,
    *,
    timezone: str,
) -> dict[str, Any]:
    taken_at = _now_local_iso(timezone)
    snapshot_local_date = taken_at.split("T")[0]
    hostname = socket.gethostname()

    iface_samples = _parse_net_dev()
    default_iface, default_gateway = _parse_default_route()
    nameservers = _parse_resolv_conf()

    db.insert_network_snapshot(
        taken_at=taken_at,
        snapshot_local_date=snapshot_local_date,
        hostname=hostname,
        default_iface=default_iface,
        default_gateway=default_gateway,
    )

    db.replace_network_interfaces(
        taken_at,
        [
            (s.name, s.state, s.rx_bytes, s.tx_bytes, s.rx_errors, s.tx_errors)
            for s in iface_samples
        ],
    )
    db.replace_network_nameservers(taken_at, nameservers)

    return {
        "taken_at": taken_at,
        "snapshot_local_date": snapshot_local_date,
        "hostname": hostname,
        "default_iface": default_iface,
        "default_gateway": default_gateway,
        "nameservers": nameservers,
        "interfaces": iface_samples,
    }

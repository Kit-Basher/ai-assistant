from __future__ import annotations

import re
import socket
import subprocess
from typing import Any

from agent.cards import normalize_card
from skills.network_governor import collector

AUDIT_HARD_FAIL_MSG = "Audit logging failed. Operation aborted."


def _blocked(message: str) -> dict[str, Any]:
    return {"status": "blocked", "message": message, "text": message}


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


def _format_delta(delta: int | None) -> str:
    if delta is None:
        return "n/a"
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{_bytes_to_human(abs(delta))}"


def _log_audit(db: Any, event_type: str, payload: dict[str, Any]) -> None:
    try:
        db.log_activity(event_type, payload)
    except Exception as exc:
        raise RuntimeError("audit_log_failed") from exc


def network_snapshot(context: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    db = context.get("db") if context else None
    timezone = (context or {}).get("timezone") or "UTC"
    actor_id = user_id or (context or {}).get("user_id") or "system"
    if not db:
        return _blocked("Database not available.")

    try:
        with db.transaction():
            payload = collector.collect_and_persist_snapshot(db, timezone=timezone)
            _log_audit(
                db,
                "network_snapshot",
                {
                    "event_type": "network_snapshot",
                    "mode": "observe",
                    "status": "executed",
                    "actor_id": actor_id,
                    "timezone": timezone,
                    "taken_at": payload.get("taken_at"),
                },
            )
    except Exception as exc:
        if str(exc) == "audit_log_failed":
            return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}
        return {"status": "failed", "message": "Snapshot failed.", "text": "Snapshot failed."}

    text = "Snapshot stored: {} ({})".format(payload.get("taken_at", ""), timezone)
    return {"status": "ok", "text": text, "payload": payload}


def network_report(context: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    db = context.get("db") if context else None
    timezone = (context or {}).get("timezone") or "UTC"
    actor_id = user_id or (context or {}).get("user_id") or "system"
    read_only_mode = bool((context or {}).get("read_only_mode"))
    if not db:
        return _blocked("Database not available.")

    if read_only_mode:
        return _build_network_report(db, timezone)

    try:
        with db.transaction():
            report = _build_network_report(db, timezone)
            _log_audit(
                db,
                "network_report",
                {
                    "event_type": "network_report",
                    "mode": "observe",
                    "status": "executed",
                    "actor_id": actor_id,
                    "timezone": timezone,
                    "taken_at": report.get("taken_at"),
                },
            )
    except Exception as exc:
        if str(exc) == "audit_log_failed":
            return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}
        return {"status": "failed", "message": "Report failed.", "text": "Report failed."}

    return report


def _build_network_report(db: Any, timezone: str) -> dict[str, Any]:
    latest = db.get_latest_network_snapshot()
    if not latest:
        text = "No network snapshots found yet."
        return {"status": "ok", "text": text, "payload": {"message": text}}

    previous = db.get_previous_network_snapshot(latest.get("taken_at"))
    lines: list[str] = []
    taken_at = latest.get("taken_at", "")
    if taken_at:
        lines.append(f"Snapshot taken: {taken_at} ({timezone})")

    lines.append(
        "Default route: iface={}, gateway={}".format(
            latest.get("default_iface") or "", latest.get("default_gateway") or ""
        )
    )
    if previous:
        if latest.get("default_iface") != previous.get("default_iface"):
            lines.append(
                "Default iface changed: {} -> {}".format(
                    previous.get("default_iface"), latest.get("default_iface")
                )
            )
        if latest.get("default_gateway") != previous.get("default_gateway"):
            lines.append(
                "Default gateway changed: {} -> {}".format(
                    previous.get("default_gateway"), latest.get("default_gateway")
                )
            )

    nameservers = db.get_network_nameservers(latest.get("taken_at"))
    if nameservers:
        lines.append("Nameservers: {}".format(", ".join([ns["nameserver"] for ns in nameservers])))
    if previous:
        prev_nameservers = db.get_network_nameservers(previous.get("taken_at"))
        prev_list = [ns["nameserver"] for ns in prev_nameservers]
        if prev_list != [ns["nameserver"] for ns in nameservers]:
            lines.append("Nameservers changed")

    latest_ifaces = db.get_network_interfaces(latest.get("taken_at"))
    prev_ifaces = db.get_network_interfaces(previous.get("taken_at")) if previous else []
    prev_map = {row["name"]: row for row in prev_ifaces}

    if latest_ifaces:
        lines.append("Interfaces:")
        for row in latest_ifaces:
            prev = prev_map.get(row["name"])
            rx_delta = None
            tx_delta = None
            rx_err_delta = None
            tx_err_delta = None
            if prev:
                rx_delta = int(row["rx_bytes"]) - int(prev["rx_bytes"])
                tx_delta = int(row["tx_bytes"]) - int(prev["tx_bytes"])
                rx_err_delta = int(row["rx_errors"]) - int(prev["rx_errors"])
                tx_err_delta = int(row["tx_errors"]) - int(prev["tx_errors"])
            lines.append(
                "- {} state={} rx={} (Δ {}) tx={} (Δ {}) rx_err={} (Δ {}) tx_err={} (Δ {})".format(
                    row["name"],
                    row["state"],
                    _bytes_to_human(int(row["rx_bytes"])),
                    _format_delta(rx_delta),
                    _bytes_to_human(int(row["tx_bytes"])),
                    _format_delta(tx_delta),
                    int(row["rx_errors"]),
                    rx_err_delta if rx_err_delta is not None else "n/a",
                    int(row["tx_errors"]),
                    tx_err_delta if tx_err_delta is not None else "n/a",
                )
            )

    payload = {
        "taken_at": latest.get("taken_at"),
        "snapshot_local_date": latest.get("snapshot_local_date"),
        "default_iface": latest.get("default_iface"),
        "default_gateway": latest.get("default_gateway"),
        "nameservers": nameservers,
        "interfaces": latest_ifaces,
    }
    primary_ip = _primary_ip()
    ping_ms = _ping_latency_ms()
    up_count = len([row for row in latest_ifaces if str(row.get("state") or "").lower() == "up"])
    cards = [
        normalize_card(
            {
                "title": "Network diagnostics",
                "lines": [
                    f"interfaces up: {up_count}/{len(latest_ifaces)}",
                    f"default route: {latest.get('default_iface') or 'n/a'} via {latest.get('default_gateway') or 'n/a'}",
                    f"primary IP: {primary_ip or 'n/a'}",
                    "DNS: {}".format(", ".join([ns["nameserver"] for ns in nameservers]) if nameservers else "n/a"),
                    f"ping latency: {ping_ms}" if ping_ms else "ping latency: unavailable",
                ],
                "severity": "ok",
            },
            0,
        )
    ]
    return {"status": "ok", "text": "\n".join(lines), "payload": payload, "cards_payload": {"cards": cards, "raw_available": True}}


def _primary_ip() -> str | None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 53))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return None


def _ping_latency_ms() -> str | None:
    try:
        proc = subprocess.run(
            ["ping", "-c", "1", "-W", "1", "1.1.1.1"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return None
    match = re.search(r"time=([0-9.]+)\\s*ms", (proc.stdout or ""))
    if not match:
        return None
    return f"{float(match.group(1)):.1f} ms"

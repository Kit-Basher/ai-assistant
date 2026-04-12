from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

_TRUST_LEVELS = {"native", "trusted", "untrusted"}


class PackManifestError(ValueError):
    pass


@dataclass(frozen=True)
class PackManifest:
    pack_id: str
    version: str
    title: str
    description: str
    entrypoints: tuple[str, ...]
    trust: str
    permissions: dict[str, Any]


def _normalize_str_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PackManifestError(f"{field_name} must be a list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise PackManifestError(f"{field_name} entries must be strings")
        stripped = item.strip()
        if stripped:
            out.append(stripped)
    return sorted(set(out))


def normalize_permissions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PackManifestError("permissions must be an object")

    ifaces = _normalize_str_list(value.get("ifaces"), "permissions.ifaces")

    fs_raw = value.get("fs")
    if fs_raw is None:
        fs_raw = {}
    if not isinstance(fs_raw, dict):
        raise PackManifestError("permissions.fs must be an object")
    fs = {
        "read": _normalize_str_list(fs_raw.get("read"), "permissions.fs.read"),
        "write": _normalize_str_list(fs_raw.get("write"), "permissions.fs.write"),
    }

    net_raw = value.get("net")
    if net_raw is None:
        net_raw = {}
    if not isinstance(net_raw, dict):
        raise PackManifestError("permissions.net must be an object")
    net = {
        "allow_domains": _normalize_str_list(net_raw.get("allow_domains"), "permissions.net.allow_domains"),
        "deny": _normalize_str_list(net_raw.get("deny"), "permissions.net.deny"),
    }

    proc_raw = value.get("proc")
    if proc_raw is None:
        proc_raw = {}
    if not isinstance(proc_raw, dict):
        raise PackManifestError("permissions.proc must be an object")
    proc = {
        "spawn": _normalize_str_list(proc_raw.get("spawn"), "permissions.proc.spawn"),
    }

    return {
        "ifaces": ifaces,
        "fs": fs,
        "net": net,
        "proc": proc,
    }


def manifest_to_dict(manifest: PackManifest) -> dict[str, Any]:
    return {
        "pack_id": manifest.pack_id,
        "version": manifest.version,
        "title": manifest.title,
        "description": manifest.description,
        "entrypoints": list(manifest.entrypoints),
        "trust": manifest.trust,
        "permissions": normalize_permissions(manifest.permissions),
    }


def compute_permissions_hash(permissions: dict[str, Any]) -> str:
    normalized = normalize_permissions(permissions)
    payload = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(path: str) -> PackManifest:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise PackManifestError(f"manifest unreadable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PackManifestError("manifest is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise PackManifestError("manifest root must be an object")

    pack_id = str(payload.get("pack_id") or "").strip()
    if not pack_id:
        raise PackManifestError("pack_id is required")

    version = str(payload.get("version") or "").strip()
    if not version:
        raise PackManifestError("version is required")

    trust = str(payload.get("trust") or "").strip().lower()
    if trust not in _TRUST_LEVELS:
        raise PackManifestError("trust must be one of native|trusted|untrusted")

    entrypoints = _normalize_str_list(payload.get("entrypoints"), "entrypoints")
    if not entrypoints:
        raise PackManifestError("entrypoints is required")

    title = str(payload.get("title") or "").strip() or pack_id
    description = str(payload.get("description") or "").strip()
    permissions = normalize_permissions(payload.get("permissions"))

    return PackManifest(
        pack_id=pack_id,
        version=version,
        title=title,
        description=description,
        entrypoints=tuple(entrypoints),
        trust=trust,
        permissions=permissions,
    )

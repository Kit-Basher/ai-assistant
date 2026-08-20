from __future__ import annotations

"""Strict, bounded WP5 pack acquisition/broker contracts.

These structures are data at the trust boundary.  They never confer authority;
the mutation controller and lifecycle store bind their canonical digests later.
"""

from dataclasses import dataclass
import hashlib
import ipaddress
import json
import re
import time
import urllib.parse
from typing import Any, Mapping


ACQUISITION_SCHEMA = "personal-agent.pack-acquisition.v1"
REVIEW_SCHEMA = "personal-agent.pack-review.v1"
BROKER_SCHEMA = "personal-agent.pack-broker.v1"
DRAFT_SCHEMA = "personal-agent.pack-draft.v1"
UPDATE_SCHEMA = "personal-agent.pack-update.v1"
PRIVATE_STORE_SCHEMA = "personal-agent.pack-private-store.v1"
VISUALIZER_SCHEMA = "personal-agent.pack-visualizer.v1"
PROOF_SCHEMA = "personal-agent.pack-wp5-proof.v1"

MAX_CONTRACT_BYTES = 64 * 1024
MAX_TEXT = 1_000
MAX_EXAMPLES = 12
MAX_SCOPES = 16
MAX_CAPABILITIES = 8
MAX_VISUALIZER_STATES = 7
MAX_VISUALIZER_FRAMES = 256
MAX_ASSET_BYTES = 4 * 1024 * 1024
MAX_DECODED_PIXELS = 16_777_216
MAX_FRAME_DURATION_MS = 10_000
MAX_ANIMATION_DURATION_MS = 60_000
SUPPORTED_SOURCE_KINDS = {"github_repo", "github_archive", "catalog_entry", "generic_archive_url"}
SUPPORTED_BROKERS = {"selected_local_data", "pack_private_store", "scoped_https", "presence_visualizer"}
SUPPORTED_DRAFT_TEMPLATES = {"portable_text", "declarative_native", "local_data_search", "presence_visualizer"}
CORE_VISUAL_STATES = {"idle", "listening", "thinking", "acting", "success", "warning", "error"}
PACK_CLASSES = {"text", "declarative", "sandboxed_executable"}
_PACK_ID = re.compile(r"^[a-z][a-z0-9-]{1,47}$")
_CAP_NAME = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WP5ContractError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def contract_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _bounded(value: Any, field: str, *, maximum: int = MAX_TEXT, required: bool = True) -> str:
    text = " ".join(str(value or "").split())
    if required and not text:
        raise WP5ContractError(f"{field}_required")
    if len(text.encode("utf-8")) > maximum:
        raise WP5ContractError(f"{field}_too_large")
    return text


def _exact_fields(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise WP5ContractError(f"{field}_unknown_fields:{','.join(unknown)}")
    if len(canonical_json(value).encode("utf-8")) > MAX_CONTRACT_BYTES:
        raise WP5ContractError(f"{field}_too_large")


def _digest(value: Any, field: str, *, required: bool = True) -> str | None:
    text = str(value or "").strip().lower()
    if not text and not required:
        return None
    if not _SHA256.fullmatch(text):
        raise WP5ContractError(f"{field}_invalid")
    return text


def _pack_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not _PACK_ID.fullmatch(text):
        raise WP5ContractError("pack_id_invalid")
    return text


def _normalized_https_url(value: Any, field: str = "url", *, allow_query: bool = False) -> str:
    raw = str(value or "").strip()
    if len(raw.encode("utf-8")) > 2_048:
        raise WP5ContractError(f"{field}_too_large")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise WP5ContractError(f"{field}_invalid")
    if parsed.fragment:
        raise WP5ContractError(f"{field}_fragment_denied")
    if parsed.query and not allow_query:
        raise WP5ContractError(f"{field}_query_denied")
    raw_host = parsed.hostname
    if raw_host.endswith("."):
        raise WP5ContractError(f"{field}_trailing_dot_denied")
    try:
        host = raw_host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise WP5ContractError(f"{field}_host_invalid") from exc
    if not host or len(host) > 253 or parsed.port not in {None, 443}:
        raise WP5ContractError(f"{field}_host_invalid")
    # Literal addresses are allowed only when globally routable.  DNS answers
    # receive the same check in SafeHttpsTransport.
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise WP5ContractError(f"{field}_private_target_denied")
    netloc = f"[{host}]" if ":" in host else host
    path = parsed.path or "/"
    return urllib.parse.urlunsplit(("https", netloc, path, parsed.query, ""))


@dataclass(frozen=True)
class AcquisitionSourceV1:
    source_kind: str
    requested_url: str
    requested_ref: str | None
    resolved_commit: str | None
    pinned: bool
    catalog_source_id: str | None

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "AcquisitionSourceV1":
        _exact_fields(value, {"schema_version", "source_kind", "requested_url", "requested_ref", "resolved_commit", "catalog_source_id"}, "acquisition_source")
        if value.get("schema_version") != ACQUISITION_SCHEMA:
            raise WP5ContractError("acquisition_schema_unsupported")
        kind = str(value.get("source_kind") or "").strip().lower()
        if kind not in SUPPORTED_SOURCE_KINDS:
            raise WP5ContractError("source_kind_unsupported")
        requested_ref = _bounded(value.get("requested_ref"), "requested_ref", maximum=160, required=False) or None
        resolved = str(value.get("resolved_commit") or "").strip().lower() or None
        if resolved is not None and not re.fullmatch(r"[0-9a-f]{40}", resolved):
            raise WP5ContractError("resolved_commit_invalid")
        source_id = _bounded(value.get("catalog_source_id"), "catalog_source_id", maximum=96, required=False) or None
        return cls(kind, _normalized_https_url(value.get("requested_url"), "requested_url"), requested_ref, resolved, bool(resolved), source_id)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": ACQUISITION_SCHEMA, "source_kind": self.source_kind, "requested_url": self.requested_url, "requested_ref": self.requested_ref, "resolved_commit": self.resolved_commit, "pinned": self.pinned, "catalog_source_id": self.catalog_source_id}


@dataclass(frozen=True)
class BrokerDeclarationV1:
    kind: str
    mode: str
    scopes: tuple[str, ...]
    data_flow: str
    limits: dict[str, int]
    config: dict[str, Any]

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "BrokerDeclarationV1":
        _exact_fields(value, {"schema_version", "kind", "mode", "scopes", "data_flow", "limits", "config"}, "broker")
        if value.get("schema_version") != BROKER_SCHEMA:
            raise WP5ContractError("broker_schema_unsupported")
        kind = str(value.get("kind") or "").strip().lower()
        if kind not in SUPPORTED_BROKERS:
            raise WP5ContractError("broker_kind_unsupported")
        mode = str(value.get("mode") or "read_only").strip().lower()
        if mode not in {"read_only", "mutating"}:
            raise WP5ContractError("broker_mode_invalid")
        scopes = value.get("scopes") or []
        if not isinstance(scopes, list) or len(scopes) > MAX_SCOPES:
            raise WP5ContractError("broker_scopes_invalid")
        normalized_scopes = tuple(sorted({_bounded(item, "broker_scope", maximum=160) for item in scopes}))
        data_flow = str(value.get("data_flow") or "none").strip().lower()
        if data_flow not in {"none", "local_private", "public_network", "ui_asset"}:
            raise WP5ContractError("broker_data_flow_invalid")
        limits_raw = value.get("limits") or {}
        config = value.get("config") or {}
        if not isinstance(limits_raw, dict) or set(limits_raw) - {"input_bytes", "output_bytes", "items", "wall_ms", "requests"}:
            raise WP5ContractError("broker_limits_invalid")
        if not isinstance(config, dict) or len(config) > 24 or len(canonical_json(config).encode()) > 16_384:
            raise WP5ContractError("broker_config_invalid")
        caps = {"input_bytes": 8 * 1024 * 1024, "output_bytes": 256 * 1024, "items": 10_000, "wall_ms": 30_000, "requests": 4}
        limits: dict[str, int] = {}
        for name, ceiling in caps.items():
            amount = int(limits_raw.get(name) or min(ceiling, 1_000 if name == "items" else ceiling))
            if amount < 1 or amount > ceiling:
                raise WP5ContractError(f"broker_limit_{name}_invalid")
            limits[name] = amount
        if kind == "scoped_https" and data_flow != "public_network":
            raise WP5ContractError("network_broker_data_flow_invalid")
        if kind in {"selected_local_data", "pack_private_store"} and data_flow != "local_private":
            raise WP5ContractError("local_broker_data_flow_invalid")
        if kind == "presence_visualizer" and data_flow != "ui_asset":
            raise WP5ContractError("visualizer_data_flow_invalid")
        allowed_config = {
            "selected_local_data": {"extensions", "raw_content_retained"},
            "pack_private_store": {"schema", "retention_days"},
            "scoped_https": {"origins", "path_templates", "parameter_names", "methods", "content_types"},
            "presence_visualizer": {"asset_digest", "declaration_digest"},
        }[kind]
        if set(config) - allowed_config:
            raise WP5ContractError("broker_config_authority_field_unknown")
        if kind == "scoped_https":
            origins = config.get("origins") or []
            paths = config.get("path_templates") or []
            methods = config.get("methods") or ["GET", "HEAD"]
            parameters = config.get("parameter_names") or []
            if not isinstance(origins, list) or len(origins) != 1 or not isinstance(paths, list) or not 1 <= len(paths) <= 16:
                raise WP5ContractError("network_scope_invalid")
            config["origins"] = [_normalized_https_url(origins[0], "network_origin").rstrip("/")]
            if any(not isinstance(path, str) or not path.startswith("/") or ".." in path or "?" in path or "#" in path or len(path) > 512 for path in paths):
                raise WP5ContractError("network_path_template_invalid")
            if not isinstance(methods, list) or any(str(method).upper() not in {"GET", "HEAD"} for method in methods):
                raise WP5ContractError("network_methods_invalid")
            if not isinstance(parameters, list) or len(parameters) > 16 or any(not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", str(name)) for name in parameters):
                raise WP5ContractError("network_parameters_invalid")
            config.update({"path_templates": sorted(set(paths)), "methods": sorted(set(str(method).upper() for method in methods)), "parameter_names": sorted(set(str(name) for name in parameters))})
        if kind == "selected_local_data":
            extensions = config.get("extensions") or []
            if not isinstance(extensions, list) or not extensions or any(str(item).lower() not in {".json", ".csv", ".html", ".htm", ".txt"} for item in extensions):
                raise WP5ContractError("local_data_extensions_invalid")
            if config.get("raw_content_retained") is not False:
                raise WP5ContractError("local_data_raw_retention_denied")
        return cls(kind, mode, normalized_scopes, data_flow, limits, dict(config))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": BROKER_SCHEMA, "kind": self.kind, "mode": self.mode, "scopes": list(self.scopes), "data_flow": self.data_flow, "limits": dict(self.limits), "config": dict(self.config)}

    @property
    def digest(self) -> str:
        return contract_digest(self.to_dict())


def normalize_draft(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(value, {"schema_version", "pack_id", "version", "pack_class", "display_name", "description", "template", "capabilities", "brokers", "limitations", "visualizer"}, "draft")
    if value.get("schema_version") != DRAFT_SCHEMA:
        raise WP5ContractError("draft_schema_unsupported")
    template = str(value.get("template") or "").strip().lower()
    if template not in SUPPORTED_DRAFT_TEMPLATES:
        raise WP5ContractError("draft_template_unsupported")
    pack_class = str(value.get("pack_class") or "").strip().lower()
    if pack_class not in {"text", "declarative"}:
        raise WP5ContractError("generated_executable_pack_denied")
    caps = value.get("capabilities") or []
    brokers = value.get("brokers") or []
    limitations = value.get("limitations") or []
    if not isinstance(caps, list) or len(caps) > MAX_CAPABILITIES or not isinstance(brokers, list) or len(brokers) > 4 or not isinstance(limitations, list) or len(limitations) > 12:
        raise WP5ContractError("draft_collection_bounds")
    normalized_caps: list[dict[str, Any]] = []
    for row in caps:
        if not isinstance(row, dict):
            raise WP5ContractError("draft_capability_invalid")
        _exact_fields(row, {"name", "display_name", "description", "examples", "input_schema", "output_schema", "invocation"}, "draft_capability")
        name = str(row.get("name") or "").strip().lower()
        if not _CAP_NAME.fullmatch(name):
            raise WP5ContractError("draft_capability_name_invalid")
        invocation = row.get("invocation") or {}
        if not isinstance(invocation, dict):
            raise WP5ContractError("draft_invocation_invalid")
        _exact_fields(invocation, {"kind", "capability_id", "broker_kind", "operation", "inputs"}, "draft_invocation")
        if invocation.get("kind") not in {"registered_capability", "core_broker"}:
            raise WP5ContractError("draft_invocation_kind_unsupported")
        if any(term in canonical_json(invocation).lower() for term in ("shell", "endpoint", "handler.py", "javascript", "python", "oauth", "docker")):
            raise WP5ContractError("draft_raw_authority_denied")
        examples = row.get("examples") or []
        if not isinstance(examples, list) or not 1 <= len(examples) <= MAX_EXAMPLES:
            raise WP5ContractError("draft_examples_invalid")
        normalized_caps.append({"name": name, "display_name": _bounded(row.get("display_name") or name, "capability_display_name"), "description": _bounded(row.get("description"), "capability_description"), "examples": [_bounded(item, "example") for item in examples], "input_schema": row.get("input_schema") or {"type": "object"}, "output_schema": row.get("output_schema") or {"type": "object"}, "invocation": dict(invocation)})
    normalized_brokers = [BrokerDeclarationV1.parse(row).to_dict() if isinstance(row, Mapping) else (_ for _ in ()).throw(WP5ContractError("draft_broker_invalid")) for row in brokers]
    visualizer = None
    if isinstance(value.get("visualizer"), dict):
        raw_visualizer = dict(value["visualizer"])
        raw_visualizer.pop("declaration_digest", None)
        visualizer = normalize_visualizer(raw_visualizer)
    normalized = {"schema_version": DRAFT_SCHEMA, "pack_id": _pack_id(value.get("pack_id")), "version": _bounded(value.get("version"), "version", maximum=80), "pack_class": pack_class, "display_name": _bounded(value.get("display_name"), "display_name"), "description": _bounded(value.get("description"), "description"), "template": template, "capabilities": normalized_caps, "brokers": normalized_brokers, "limitations": [_bounded(item, "limitation", maximum=300) for item in limitations], "visualizer": visualizer}
    normalized["draft_digest"] = contract_digest(normalized)
    return normalized


def normalize_visualizer(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(value, {"schema_version", "asset", "frame_width", "frame_height", "columns", "rows", "animations"}, "visualizer")
    if value.get("schema_version") != VISUALIZER_SCHEMA:
        raise WP5ContractError("visualizer_schema_unsupported")
    asset = value.get("asset") or {}
    if not isinstance(asset, dict):
        raise WP5ContractError("visualizer_asset_invalid")
    _exact_fields(asset, {"path", "sha256", "media_type", "bytes", "width", "height"}, "visualizer_asset")
    media = str(asset.get("media_type") or "").lower()
    if media != "image/png":
        raise WP5ContractError("visualizer_media_type_unsupported")
    width, height = int(asset.get("width") or 0), int(asset.get("height") or 0)
    size = int(asset.get("bytes") or 0)
    if width < 1 or height < 1 or width * height > MAX_DECODED_PIXELS or size < 1 or size > MAX_ASSET_BYTES:
        raise WP5ContractError("visualizer_asset_bounds")
    frame_w, frame_h = int(value.get("frame_width") or 0), int(value.get("frame_height") or 0)
    columns, rows = int(value.get("columns") or 0), int(value.get("rows") or 0)
    if min(frame_w, frame_h, columns, rows) < 1 or frame_w * columns != width or frame_h * rows != height or columns * rows > MAX_VISUALIZER_FRAMES:
        raise WP5ContractError("visualizer_grid_invalid")
    animations = value.get("animations") or {}
    if not isinstance(animations, dict) or not animations or len(animations) > MAX_VISUALIZER_STATES or set(animations) - CORE_VISUAL_STATES:
        raise WP5ContractError("visualizer_states_invalid")
    normalized_animations: dict[str, Any] = {}
    for state, row in sorted(animations.items()):
        if not isinstance(row, dict):
            raise WP5ContractError("visualizer_animation_invalid")
        _exact_fields(row, {"frames", "frame_duration_ms", "loop"}, "visualizer_animation")
        frames = row.get("frames") or []
        duration = int(row.get("frame_duration_ms") or 0)
        if not isinstance(frames, list) or not frames or len(frames) > MAX_VISUALIZER_FRAMES or any(not isinstance(frame, int) or frame < 0 or frame >= columns * rows for frame in frames):
            raise WP5ContractError("visualizer_frames_invalid")
        if duration < 16 or duration > MAX_FRAME_DURATION_MS or duration * len(frames) > MAX_ANIMATION_DURATION_MS:
            raise WP5ContractError("visualizer_duration_invalid")
        normalized_animations[state] = {"frames": frames, "frame_duration_ms": duration, "loop": bool(row.get("loop", True))}
    normalized = {"schema_version": VISUALIZER_SCHEMA, "asset": {"path": _bounded(asset.get("path"), "asset_path", maximum=160), "sha256": _digest(asset.get("sha256"), "asset_sha256"), "media_type": media, "bytes": size, "width": width, "height": height}, "frame_width": frame_w, "frame_height": frame_h, "columns": columns, "rows": rows, "animations": normalized_animations}
    normalized["declaration_digest"] = contract_digest(normalized)
    return normalized


def authority_binding(*, pack_id: str, version: str, content_digest: str, capability_contract_digest: str, broker_digest: str | None, actor_id: str, session_id: str, thread_id: str, plan_id: str, expires_at: int) -> dict[str, Any]:
    payload = {"schema_version": ACQUISITION_SCHEMA, "pack_id": _pack_id(pack_id), "version": _bounded(version, "version", maximum=80), "content_digest": _digest(content_digest, "content_digest"), "capability_contract_digest": _digest(capability_contract_digest, "capability_contract_digest"), "broker_digest": _digest(broker_digest, "broker_digest", required=False), "actor_id": _bounded(actor_id, "actor_id", maximum=160), "session_id": _bounded(session_id, "session_id", maximum=160), "thread_id": _bounded(thread_id, "thread_id", maximum=160), "plan_id": _bounded(plan_id, "plan_id", maximum=160), "expires_at": int(expires_at)}
    if payload["expires_at"] <= int(time.time()):
        raise WP5ContractError("authority_binding_expired")
    payload["binding_digest"] = contract_digest(payload)
    return payload


__all__ = [
    "ACQUISITION_SCHEMA", "REVIEW_SCHEMA", "BROKER_SCHEMA", "DRAFT_SCHEMA", "UPDATE_SCHEMA",
    "PRIVATE_STORE_SCHEMA", "VISUALIZER_SCHEMA", "PROOF_SCHEMA", "WP5ContractError",
    "AcquisitionSourceV1", "BrokerDeclarationV1", "normalize_draft", "normalize_visualizer",
    "authority_binding", "canonical_json", "contract_digest", "_normalized_https_url",
]

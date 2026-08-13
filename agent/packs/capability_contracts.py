from __future__ import annotations

"""Strict WP4 pack-capability contracts.

Pack documents are data, never Python authority.  This module accepts a small
JSON subset and produces a deterministic digest-bound representation.
"""

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


PACK_SCHEMA = "personal-agent.pack.v1"
CAPABILITY_SCHEMA = "personal-agent.pack-capability.v1"
WORKER_ABI = "personal-agent.pack-worker.v1"
PROOF_SCHEMA = "personal-agent.pack-proof.v1"
PACK_CLASSES = {"text", "declarative", "sandboxed_executable"}
MAX_MANIFEST_BYTES = 64 * 1024
MAX_CAPABILITIES = 8
MAX_PROPERTIES = 24
MAX_EXAMPLES = 12
MAX_TEXT = 1_000
MAX_SCHEMA_DEPTH = 5
MAX_WASM_BYTES = 2 * 1024 * 1024
MAX_CONTENT_FILES = 64
MAX_CONTENT_BYTES = 4 * 1024 * 1024
_ID = re.compile(r"^[a-z][a-z0-9-]{1,47}$")
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
_ALLOWED_PACK = {"schema_version", "id", "version", "pack_class", "display_name", "description", "capabilities"}
_ALLOWED_CAP = {"schema_version", "name", "display_name", "description", "examples", "input_schema", "output_schema", "mode", "task_composable", "permissions", "invocation", "verifier", "limits", "self_test_input"}
_ALLOWED_TYPES = {"string", "integer", "number", "boolean", "object", "array"}


class PackCapabilityContractError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_digest(path: Path, *, max_bytes: int = MAX_WASM_BYTES) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise PackCapabilityContractError("pack_artifact_too_large")
    return hashlib.sha256(data).hexdigest()


def _content_files(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink() or any(part.startswith(".") for part in relative.parts):
            raise PackCapabilityContractError("pack_content_link_or_hidden_denied")
        if path.is_dir():
            continue
        if not path.is_file() or path.stat().st_nlink != 1:
            raise PackCapabilityContractError("pack_content_file_invalid")
        if relative.as_posix() == "personal-agent-pack.json":
            continue
        data = path.read_bytes()
        if len(data) > MAX_WASM_BYTES:
            raise PackCapabilityContractError("pack_content_file_too_large")
        total += len(data)
        rows.append({"path": relative.as_posix(), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if len(rows) > MAX_CONTENT_FILES or total > MAX_CONTENT_BYTES:
            raise PackCapabilityContractError("pack_content_bounds_exceeded")
    return rows


def _bounded_text(value: Any, field: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise PackCapabilityContractError(f"{field}_required")
    if len(text.encode()) > MAX_TEXT:
        raise PackCapabilityContractError(f"{field}_too_large")
    return text


def _schema(value: Any, field: str, *, depth: int = 0) -> dict[str, Any]:
    if depth > MAX_SCHEMA_DEPTH or not isinstance(value, dict):
        raise PackCapabilityContractError(f"{field}_invalid")
    allowed = {"type", "properties", "required", "items", "enum", "maxLength", "minimum", "maximum"}
    unknown = set(value) - allowed
    if unknown:
        raise PackCapabilityContractError(f"{field}_unknown_fields:{','.join(sorted(unknown))}")
    kind = str(value.get("type") or "object")
    if kind not in _ALLOWED_TYPES:
        raise PackCapabilityContractError(f"{field}_type_invalid")
    out: dict[str, Any] = {"type": kind}
    if kind == "object":
        props = value.get("properties") or {}
        if not isinstance(props, dict) or len(props) > MAX_PROPERTIES:
            raise PackCapabilityContractError(f"{field}_properties_invalid")
        normalized: dict[str, Any] = {}
        for key, row in sorted(props.items()):
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", str(key)):
                raise PackCapabilityContractError(f"{field}_property_name_invalid")
            normalized[str(key)] = _schema(row, f"{field}.{key}", depth=depth + 1)
        required = value.get("required") or []
        if not isinstance(required, list) or any(item not in normalized for item in required):
            raise PackCapabilityContractError(f"{field}_required_invalid")
        out.update({"properties": normalized, "required": sorted(set(str(x) for x in required))})
    elif kind == "array":
        out["items"] = _schema(value.get("items") or {"type": "string"}, f"{field}.items", depth=depth + 1)
    if "enum" in value:
        enum = value["enum"]
        if not isinstance(enum, list) or not 1 <= len(enum) <= 32:
            raise PackCapabilityContractError(f"{field}_enum_invalid")
        out["enum"] = enum
    for key in ("maxLength", "minimum", "maximum"):
        if key in value:
            if not isinstance(value[key], (int, float)) or abs(float(value[key])) > 1_000_000:
                raise PackCapabilityContractError(f"{field}_{key}_invalid")
            out[key] = value[key]
    return out


def validate_value(schema: Mapping[str, Any], value: Any, *, field: str = "input", depth: int = 0) -> Any:
    if depth > MAX_SCHEMA_DEPTH:
        raise PackCapabilityContractError(f"{field}_depth_exceeded")
    kind = schema.get("type")
    expected = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}[str(kind)]
    if not isinstance(value, expected) or kind == "integer" and isinstance(value, bool) or kind == "number" and isinstance(value, bool):
        raise PackCapabilityContractError(f"{field}_type_invalid")
    if kind == "string" and len(value.encode()) > int(schema.get("maxLength") or 4096):
        raise PackCapabilityContractError(f"{field}_too_large")
    if kind in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise PackCapabilityContractError(f"{field}_below_minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise PackCapabilityContractError(f"{field}_above_maximum")
    if kind == "object":
        props = schema.get("properties") or {}
        unknown = set(value) - set(props)
        if unknown:
            raise PackCapabilityContractError(f"{field}_unknown_fields:{','.join(sorted(unknown))}")
        missing = set(schema.get("required") or []) - set(value)
        if missing:
            raise PackCapabilityContractError(f"{field}_missing_fields:{','.join(sorted(missing))}")
        return {k: validate_value(props[k], v, field=f"{field}.{k}", depth=depth + 1) for k, v in value.items()}
    if kind == "array":
        if len(value) > 64:
            raise PackCapabilityContractError(f"{field}_too_many_items")
        return [validate_value(schema["items"], x, field=f"{field}[]", depth=depth + 1) for x in value]
    if "enum" in schema and value not in schema["enum"]:
        raise PackCapabilityContractError(f"{field}_enum_invalid")
    return value


def _normalize_invocation(pack_class: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PackCapabilityContractError("invocation_invalid")
    if pack_class == "declarative":
        if set(value) - {"kind", "capability_id", "inputs", "result_field"}:
            raise PackCapabilityContractError("declarative_unknown_authority_field")
        if value.get("kind") != "registered_capability":
            raise PackCapabilityContractError("declarative_kind_invalid")
        capability_id = str(value.get("capability_id") or "").strip().lower()
        if not capability_id or capability_id.startswith("pack."):
            raise PackCapabilityContractError("declarative_capability_invalid")
        inputs = value.get("inputs") or {}
        if not isinstance(inputs, dict) or len(inputs) > MAX_PROPERTIES:
            raise PackCapabilityContractError("declarative_inputs_invalid")
        for item in inputs.values():
            if isinstance(item, str) and item.startswith("$") and not re.fullmatch(r"\$input\.[a-z][a-z0-9_]{0,47}", item):
                raise PackCapabilityContractError("declarative_template_invalid")
            if isinstance(item, (dict, list)):
                raise PackCapabilityContractError("declarative_expression_invalid")
        result_field = str(value.get("result_field") or "text")
        if result_field != "text":
            raise PackCapabilityContractError("declarative_result_field_unsupported")
        return {"kind": "registered_capability", "capability_id": capability_id, "inputs": dict(sorted(inputs.items())), "result_field": result_field}
    if pack_class == "sandboxed_executable":
        if set(value) - {"kind", "abi", "module", "export", "input_field"}:
            raise PackCapabilityContractError("executable_unknown_authority_field")
        module = str(value.get("module") or "").strip()
        if value.get("kind") != "wasm" or value.get("abi") != WORKER_ABI:
            raise PackCapabilityContractError("worker_abi_invalid")
        if Path(module).name != module or not module.endswith(".wasm"):
            raise PackCapabilityContractError("worker_module_path_invalid")
        export = str(value.get("export") or "invoke")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", export):
            raise PackCapabilityContractError("worker_export_invalid")
        return {"kind": "wasm", "abi": WORKER_ABI, "module": module, "export": export, "input_field": str(value.get("input_field") or "value")}
    raise PackCapabilityContractError("text_pack_cannot_declare_invocation")


def normalize_manifest(payload: Mapping[str, Any], *, source_dir: Path) -> dict[str, Any]:
    raw = canonical_json(payload)
    if len(raw.encode()) > MAX_MANIFEST_BYTES:
        raise PackCapabilityContractError("pack_manifest_too_large")
    if set(payload) - _ALLOWED_PACK:
        raise PackCapabilityContractError("pack_unknown_authority_fields")
    if payload.get("schema_version") != PACK_SCHEMA:
        raise PackCapabilityContractError("pack_schema_unsupported")
    pack_id = str(payload.get("id") or "").strip().lower()
    if not _ID.fullmatch(pack_id):
        raise PackCapabilityContractError("pack_id_invalid")
    pack_class = str(payload.get("pack_class") or "")
    if pack_class not in PACK_CLASSES:
        raise PackCapabilityContractError("pack_class_invalid")
    caps = payload.get("capabilities") or []
    if not isinstance(caps, list) or len(caps) > MAX_CAPABILITIES or pack_class != "text" and not caps or pack_class == "text" and caps:
        raise PackCapabilityContractError("pack_capabilities_invalid")
    normalized_caps = []
    seen: set[str] = set()
    for row in caps:
        if not isinstance(row, dict) or set(row) - _ALLOWED_CAP or row.get("schema_version") != CAPABILITY_SCHEMA:
            raise PackCapabilityContractError("pack_capability_contract_invalid")
        name = str(row.get("name") or "").strip().lower()
        if not _NAME.fullmatch(name) or name in seen:
            raise PackCapabilityContractError("pack_capability_name_invalid")
        seen.add(name)
        examples = row.get("examples") or []
        if not isinstance(examples, list) or not 1 <= len(examples) <= MAX_EXAMPLES:
            raise PackCapabilityContractError("pack_capability_examples_invalid")
        examples = [_bounded_text(x, "example") for x in examples]
        mode = str(row.get("mode") or "read_only")
        if mode not in {"read_only", "mutating"}:
            raise PackCapabilityContractError("pack_capability_mode_invalid")
        permissions = row.get("permissions") or []
        if not isinstance(permissions, list) or any(x not in {"pure_compute"} for x in permissions):
            raise PackCapabilityContractError("pack_permission_unsupported")
        invocation = _normalize_invocation(pack_class, row.get("invocation"))
        verifier = row.get("verifier") or {}
        if not isinstance(verifier, dict) or set(verifier) - {"kind", "field"} or verifier.get("kind") not in {"nonempty", "integer_result"}:
            raise PackCapabilityContractError("pack_verifier_invalid")
        limits = row.get("limits") or {}
        if not isinstance(limits, dict) or set(limits) - {"fuel", "memory_bytes", "wall_ms", "output_bytes"}:
            raise PackCapabilityContractError("pack_limits_invalid")
        bounded_limits = {
            "fuel": min(10_000_000, max(10_000, int(limits.get("fuel") or 1_000_000))),
            "memory_bytes": min(32 * 1024 * 1024, max(64 * 1024, int(limits.get("memory_bytes") or 8 * 1024 * 1024))),
            "wall_ms": min(5_000, max(50, int(limits.get("wall_ms") or 1_000))),
            "output_bytes": min(64 * 1024, max(256, int(limits.get("output_bytes") or 8 * 1024))),
        }
        cap = {
            "schema_version": CAPABILITY_SCHEMA, "name": name,
            "capability_id": f"pack.{pack_id}.{name}",
            "display_name": _bounded_text(row.get("display_name") or name.replace("_", " "), "display_name"),
            "description": _bounded_text(row.get("description"), "description"),
            "examples": examples, "input_schema": _schema(row.get("input_schema") or {"type": "object"}, "input_schema"),
            "output_schema": _schema(row.get("output_schema") or {"type": "object"}, "output_schema"),
            "mode": mode, "task_composable": bool(row.get("task_composable", True)),
            "permissions": sorted(set(permissions)), "invocation": invocation,
            "verifier": {"kind": verifier["kind"], "field": str(verifier.get("field") or "result")}, "limits": bounded_limits,
        }
        if cap["input_schema"].get("type") != "object" or cap["output_schema"].get("type") != "object":
            raise PackCapabilityContractError("pack_root_schema_must_be_object")
        cap["self_test_input"] = validate_value(cap["input_schema"], row.get("self_test_input") or {}, field="self_test_input")
        input_props = cap["input_schema"].get("properties") or {}
        output_props = cap["output_schema"].get("properties") or {}
        if invocation["kind"] == "wasm":
            input_field = invocation["input_field"]
            if input_field not in input_props or input_props[input_field].get("type") != "integer":
                raise PackCapabilityContractError("worker_input_field_contract_invalid")
        else:
            for mapped in invocation["inputs"].values():
                if isinstance(mapped, str) and mapped.startswith("$input.") and mapped.split(".", 1)[1] not in input_props:
                    raise PackCapabilityContractError("declarative_input_reference_unknown")
        verifier_field = cap["verifier"]["field"]
        if verifier_field not in output_props or verifier_field not in set(cap["output_schema"].get("required") or []):
            raise PackCapabilityContractError("verifier_output_contract_invalid")
        if cap["verifier"]["kind"] == "integer_result" and output_props[verifier_field].get("type") != "integer":
            raise PackCapabilityContractError("verifier_output_type_invalid")
        if pack_class == "sandboxed_executable":
            module_path = (source_dir / invocation["module"]).resolve()
            try:
                module_path.relative_to(source_dir.resolve())
            except ValueError as exc:
                raise PackCapabilityContractError("worker_module_outside_pack") from exc
            if module_path.is_symlink() or not module_path.is_file():
                raise PackCapabilityContractError("worker_module_invalid")
            cap["artifact_digest"] = file_digest(module_path)
        cap["contract_digest"] = digest(cap)
        normalized_caps.append(cap)
    normalized = {
        "schema_version": PACK_SCHEMA, "id": pack_id,
        "version": _bounded_text(payload.get("version"), "version"), "pack_class": pack_class,
        "display_name": _bounded_text(payload.get("display_name") or pack_id, "display_name"),
        "description": _bounded_text(payload.get("description"), "description"),
        "capabilities": normalized_caps, "content_files": _content_files(source_dir.resolve()),
    }
    normalized["content_digest"] = digest(normalized)
    return normalized


def load_manifest(source_dir: str | Path) -> dict[str, Any]:
    root = Path(source_dir).expanduser().resolve()
    manifest = root / "personal-agent-pack.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise PackCapabilityContractError("pack_manifest_missing")
    data = manifest.read_bytes()
    if len(data) > MAX_MANIFEST_BYTES:
        raise PackCapabilityContractError("pack_manifest_too_large")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise PackCapabilityContractError("pack_manifest_malformed") from exc
    if not isinstance(payload, dict):
        raise PackCapabilityContractError("pack_manifest_invalid")
    return normalize_manifest(payload, source_dir=root)

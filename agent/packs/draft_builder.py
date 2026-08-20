from __future__ import annotations

"""Deterministic, non-self-authorizing WP5 pack draft builder."""

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping

from agent.packs.capability_contracts import CAPABILITY_SCHEMA, PACK_SCHEMA, normalize_manifest
from agent.packs.wp5_contracts import BROKER_SCHEMA, DRAFT_SCHEMA, normalize_draft
from agent.packs.wp5_contracts import VISUALIZER_SCHEMA, normalize_visualizer
from agent.packs.brokers import decoded_raster_metadata


class PackDraftError(ValueError):
    pass


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return (text or "created-pack")[:48]


class PackDraftBuilder:
    def __init__(self, storage_root: str | Path, *, allowed_asset_roots: tuple[str | Path, ...] | None = None) -> None:
        storage = Path(storage_root).expanduser().resolve()
        self.root = storage / "drafts-v1"
        self.allowed_asset_roots = tuple(
            Path(item).expanduser().resolve()
            for item in (allowed_asset_roots or (storage,))
        )
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _read_visualizer_asset(self, value: Any) -> tuple[Path, bytes]:
        source = Path(str(value or "")).expanduser()
        if not source.is_absolute():
            raise PackDraftError("visualizer_asset_absolute_path_required")
        try:
            resolved = source.resolve(strict=True)
            if not any(resolved.is_relative_to(root) for root in self.allowed_asset_roots):
                raise PackDraftError("visualizer_asset_outside_allowed_roots")
            row = source.stat(follow_symlinks=False)
            if source.is_symlink() or not source.is_file() or row.st_nlink != 1 or row.st_uid != os.getuid():
                raise PackDraftError("visualizer_asset_invalid")
            if row.st_size <= 0 or row.st_size > 8 * 1024 * 1024 or row.st_mode & 0o002:
                raise PackDraftError("visualizer_asset_invalid")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(source, flags)
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                    row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns
                ):
                    raise PackDraftError("visualizer_asset_changed_during_read")
                data = b""
                while len(data) <= 8 * 1024 * 1024:
                    chunk = os.read(descriptor, min(64 * 1024, 8 * 1024 * 1024 + 1 - len(data)))
                    if not chunk:
                        break
                    data += chunk
                if len(data) != row.st_size:
                    raise PackDraftError("visualizer_asset_changed_during_read")
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise PackDraftError("visualizer_asset_invalid") from exc
        return resolved, data

    def preview(self, request: Mapping[str, Any]) -> dict[str, Any]:
        template = str(request.get("template") or "").strip().lower()
        pack_id = _slug(request.get("pack_id") or request.get("name") or template)
        display = " ".join(str(request.get("display_name") or request.get("name") or pack_id).split())[:120]
        version = str(request.get("version") or "1.0.0")[:80]
        if template == "local_data_search":
            broker = {"schema_version": BROKER_SCHEMA, "kind": "selected_local_data", "mode": "read_only", "scopes": ["one_exact_user_selected_file", "derived_index_search"], "data_flow": "local_private", "limits": {"input_bytes": 8 * 1024 * 1024, "output_bytes": 64 * 1024, "items": 10_000, "wall_ms": 2_000, "requests": 1}, "config": {"extensions": [".json", ".csv", ".html", ".htm", ".txt"], "raw_content_retained": False}}
            cap = {"name": "search", "display_name": f"Search {display}", "description": "Search the bounded private index derived from one explicitly selected local file", "examples": [f"search my {display} export", f"find an entry in {display}"], "input_schema": {"type": "object", "properties": {"query": {"type": "string", "maxLength": 512}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["query"]}, "output_schema": {"type": "object", "properties": {"result": {"type": "string", "maxLength": 65536}}, "required": ["result"]}, "invocation": {"kind": "core_broker", "broker_kind": "selected_local_data", "operation": "search", "inputs": {"query": "$input.query", "limit": "$input.limit"}}}
            limitations = ["Reads only one separately granted file during indexing", "Never uses the network", "Stores bounded derived records rather than the raw source file"]
            brokers = [broker]
        elif template == "portable_text":
            cap, brokers = None, []
            limitations = ["Guidance only", "Never executable", "Not globally injected into conversations"]
        elif template == "declarative_native":
            capability_id = str(request.get("capability_id") or "system.status")
            cap = {"name": "report", "display_name": display, "description": f"Use the reviewed native {capability_id} capability", "examples": [f"use {display}", f"run the {display} report"], "input_schema": {"type": "object", "properties": {}, "required": []}, "output_schema": {"type": "object", "properties": {"result": {"type": "string", "maxLength": 65536}}, "required": ["result"]}, "invocation": {"kind": "registered_capability", "capability_id": capability_id, "inputs": {}}}
            brokers, limitations = [], ["Can invoke only the exact registered native capability shown in this draft"]
        elif template == "presence_visualizer":
            source, data = self._read_visualizer_asset(request.get("asset_path"))
            media = "image/png" if source.suffix.lower() == ".png" else ""
            width, height = decoded_raster_metadata(data, media)
            frame_width = int(request.get("frame_width") or width)
            frame_height = int(request.get("frame_height") or height)
            columns, rows = width // frame_width, height // frame_height
            animations = request.get("animations") if isinstance(request.get("animations"), Mapping) else {"idle": {"frames": [0], "frame_duration_ms": 250, "loop": True}, "thinking": {"frames": list(range(min(columns * rows, 4))), "frame_duration_ms": 160, "loop": True}}
            visualizer = normalize_visualizer({"schema_version": VISUALIZER_SCHEMA, "asset": {"path": f"assets/sprite{source.suffix.lower()}", "sha256": hashlib.sha256(data).hexdigest(), "media_type": media, "bytes": len(data), "width": width, "height": height}, "frame_width": frame_width, "frame_height": frame_height, "columns": columns, "rows": rows, "animations": animations})
            cap, brokers = None, []
            limitations = ["Core UI renders reviewed raster frames", "No script, SVG, HTML, CSS, audio, font, or remote asset authority"]
        else:
            raise PackDraftError("draft_template_unsupported")
        draft = normalize_draft({"schema_version": DRAFT_SCHEMA, "pack_id": pack_id, "version": version, "pack_class": "text" if template in {"portable_text", "presence_visualizer"} else "declarative", "display_name": display, "description": str(request.get("description") or f"Assistant-created {display} pack")[:1000], "template": template, "capabilities": [] if cap is None else [cap], "brokers": brokers, "limitations": limitations, "visualizer": visualizer if template == "presence_visualizer" else None})
        return {"ok": True, "draft": draft, "source_path": str(source) if template == "presence_visualizer" else None, "requires_confirmation": True, "created": False, "approved": False, "enabled": False, "granted": False, "usable": False, "message": "Draft validated. Confirmation creates only an exact quarantine candidate; it cannot approve, grant, enable, or invoke itself."}

    def create_quarantine(self, preview: Mapping[str, Any]) -> dict[str, Any]:
        proposed = dict(preview.get("draft") if isinstance(preview.get("draft"), Mapping) else preview)
        expected_digest = str(proposed.pop("draft_digest", ""))
        draft = normalize_draft(proposed)
        if expected_digest and expected_digest != draft["draft_digest"]:
            raise PackDraftError("draft_changed_after_preview")
        target = self.root / f"{draft['pack_id']}-{draft['draft_digest'][:12]}"
        if target.exists():
            return {"ok": True, "path": str(target), "draft_digest": draft["draft_digest"], "created": False}
        temp = self.root / f".{target.name}.{time.time_ns()}.tmp"
        temp.mkdir(mode=0o700)
        try:
            if draft["pack_class"] == "text":
                (temp / "SKILL.md").write_text(f"# {draft['display_name']}\n\n{draft['description']}\n", encoding="utf-8")
                manifest = {"schema_version": PACK_SCHEMA, "id": draft["pack_id"], "version": draft["version"], "pack_class": "text", "display_name": draft["display_name"], "description": draft["description"], "brokers": [], "capabilities": []}
                (temp / "personal-agent-pack.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                if draft["template"] == "presence_visualizer":
                    source_path, data = self._read_visualizer_asset(preview.get("source_path"))
                    visualizer = draft.get("visualizer") or {}
                    expected = str((visualizer.get("asset") or {}).get("sha256") or "")
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise PackDraftError("visualizer_asset_changed_after_preview")
                    asset_rel = str((visualizer.get("asset") or {}).get("path") or "")
                    asset_target = temp / asset_rel
                    asset_target.parent.mkdir(parents=True, exist_ok=True)
                    asset_target.write_bytes(data)
                    (temp / "visualizer.json").write_text(json.dumps(visualizer, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            else:
                brokers = draft["brokers"]
                capabilities = []
                for cap in draft["capabilities"]:
                    invocation = cap["invocation"]
                    permissions = [f"broker:{invocation['broker_kind']}"] if invocation["kind"] == "core_broker" else []
                    result_field = "result" if invocation["kind"] == "core_broker" else "text"
                    capabilities.append({"schema_version": CAPABILITY_SCHEMA, "name": cap["name"], "display_name": cap["display_name"], "description": cap["description"], "examples": cap["examples"], "input_schema": cap["input_schema"], "output_schema": cap["output_schema"], "mode": "read_only", "task_composable": True, "permissions": permissions, "invocation": {**invocation, "result_field": result_field}, "verifier": {"kind": "nonempty", "field": "result"}, "limits": {"fuel": 1000000, "memory_bytes": 8388608, "wall_ms": 2000, "output_bytes": 65536}, "self_test_input": ({"query": "self-test", "limit": 1} if draft["template"] == "local_data_search" else {})})
                manifest = {"schema_version": PACK_SCHEMA, "id": draft["pack_id"], "version": draft["version"], "pack_class": "declarative", "display_name": draft["display_name"], "description": draft["description"], "brokers": brokers, "capabilities": capabilities}
                normalize_manifest(manifest, source_dir=temp)
                (temp / "personal-agent-pack.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            for path in temp.rglob("*"):
                if path.is_file(): path.chmod(0o400)
            os.chmod(temp, 0o500)
            temp.rename(target)
        finally:
            if temp.exists():
                for child in temp.iterdir(): child.unlink(missing_ok=True)
                temp.rmdir()
        return {"ok": True, "path": str(target), "draft_digest": draft["draft_digest"], "created": True, "quarantine_only": True, "approved": False, "enabled": False, "granted": False, "usable": False}


__all__ = ["PackDraftBuilder", "PackDraftError"]

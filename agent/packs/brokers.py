from __future__ import annotations

"""Core-owned useful brokers for reviewed WP5 packs.

No pack code is called from this module.  Broker declarations and grants are
exact data; host access is performed and verified entirely by core code.
"""

import csv
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading
import time
import urllib.parse
import zlib
from typing import Any, Mapping

from agent.packs.secure_transport import ALLOWED_PUBLIC_DATA_TYPES, SafeHttpsTransport
from agent.packs.wp5_contracts import (
    PRIVATE_STORE_SCHEMA,
    BrokerDeclarationV1,
    WP5ContractError,
    canonical_json,
    contract_digest,
    normalize_visualizer,
)


MAX_LOCAL_FILE_BYTES = 8 * 1024 * 1024
MAX_INDEX_RECORDS = 10_000
MAX_INDEX_BYTES = 4 * 1024 * 1024
MAX_RECORD_BYTES = 4_096
MAX_SEARCH_RESULTS = 50
MAX_QUERY_BYTES = 512
MAX_PRIVATE_ITEMS = 10_000
MAX_PRIVATE_VALUE_BYTES = 16 * 1024
MAX_PRIVATE_TOTAL_BYTES = 8 * 1024 * 1024
MAX_PRIVATE_WRITES_PER_MINUTE = 120
SUPPORTED_LOCAL_EXTENSIONS = {".json", ".csv", ".html", ".htm", ".txt"}
DISALLOWED_NAME_MARKERS = {"credential", "credentials", "secret", "secrets", "password", "passwd", "shadow", "token", "wallet", "keyring"}


class BrokerError(RuntimeError):
    pass


def _now() -> int:
    return int(time.time())


def _redacted_path(path: str | Path) -> str:
    return f"<selected-file>/{Path(path).name or 'file'}"


def _safe_text(value: Any, *, maximum: int = 1_000) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"(?i)(?:token|secret|api[_-]?key|password)\s*[:=]\s*\S+", "[REDACTED]", text)
    return text[:maximum]


def _within_roots(path: Path, roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve(strict=True)
    for root in roots:
        try:
            resolved.relative_to(root.resolve(strict=True))
            return True
        except (ValueError, OSError):
            continue
    return False


def file_metadata(path: str | Path, *, allowed_roots: tuple[str | Path, ...], max_bytes: int = MAX_LOCAL_FILE_BYTES) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    roots = tuple(Path(root).expanduser() for root in allowed_roots)
    if not roots or not candidate.is_absolute():
        raise BrokerError("selected_file_absolute_path_required")
    try:
        if not _within_roots(candidate, roots):
            raise BrokerError("selected_file_outside_allowed_roots")
        if candidate.is_symlink():
            raise BrokerError("selected_file_symlink_denied")
        row = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise BrokerError("selected_file_unavailable") from exc
    if not stat.S_ISREG(row.st_mode) or row.st_nlink != 1:
        raise BrokerError("selected_file_not_regular")
    if row.st_size < 0 or row.st_size > max_bytes:
        raise BrokerError("selected_file_size_denied")
    if row.st_mode & stat.S_IWOTH:
        raise BrokerError("selected_file_unsafe_permissions")
    if row.st_uid != os.getuid():
        raise BrokerError("selected_file_owner_denied")
    extension = candidate.suffix.lower()
    if extension not in SUPPORTED_LOCAL_EXTENSIONS:
        raise BrokerError("selected_file_type_denied")
    lowered_name = candidate.name.lower()
    if any(marker in lowered_name for marker in DISALLOWED_NAME_MARKERS):
        raise BrokerError("selected_file_sensitive_name_denied")
    return {
        "path_redacted": _redacted_path(candidate),
        "extension": extension,
        "size_bytes": int(row.st_size),
        "device": int(row.st_dev),
        "inode": int(row.st_ino),
        "mtime_ns": int(row.st_mtime_ns),
        "mode": stat.S_IMODE(row.st_mode),
        "owner_uid": int(row.st_uid),
        "fingerprint": hashlib.sha256(f"{row.st_dev}:{row.st_ino}:{row.st_size}:{row.st_mtime_ns}".encode()).hexdigest(),
    }


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data)


def _flatten_scalars(value: Any, *, prefix: str = "", depth: int = 0, output: dict[str, str] | None = None) -> dict[str, str]:
    out = output if output is not None else {}
    if depth > 5 or len(out) >= 32:
        return out
    if isinstance(value, Mapping):
        for key, item in list(value.items())[:32]:
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten_scalars(item, prefix=name[:120], depth=depth + 1, output=out)
    elif isinstance(value, list):
        for index, item in enumerate(value[:32]):
            _flatten_scalars(item, prefix=f"{prefix}[{index}]"[:120], depth=depth + 1, output=out)
    elif value is None or isinstance(value, (str, int, float, bool)):
        out[prefix or "value"] = _safe_text(value, maximum=1_000)
    return out


def _records_from_bytes(data: bytes, extension: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BrokerError("selected_file_encoding_unsupported") from exc
    records: list[dict[str, Any]] = []
    if extension == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BrokerError("selected_file_json_invalid") from exc
        rows = parsed if isinstance(parsed, list) else [parsed]
        for index, row in enumerate(rows[:MAX_INDEX_RECORDS]):
            fields = _flatten_scalars(row)
            records.append({"id": index, "fields": fields, "search_text": _safe_text(" ".join(fields.values()), maximum=MAX_RECORD_BYTES)})
    elif extension == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None or len(reader.fieldnames) > 64:
            raise BrokerError("selected_file_csv_invalid")
        for index, row in enumerate(reader):
            if index >= MAX_INDEX_RECORDS:
                break
            fields = {str(key)[:120]: _safe_text(value, maximum=1_000) for key, value in list(row.items())[:64]}
            records.append({"id": index, "fields": fields, "search_text": _safe_text(" ".join(fields.values()), maximum=MAX_RECORD_BYTES)})
    elif extension in {".html", ".htm"}:
        parser = _TextExtractor()
        try:
            parser.feed(text)
        except Exception as exc:
            raise BrokerError("selected_file_html_invalid") from exc
        for index, part in enumerate(parser.parts[:MAX_INDEX_RECORDS]):
            value = _safe_text(part, maximum=MAX_RECORD_BYTES)
            records.append({"id": index, "fields": {"text": value}, "search_text": value})
    else:
        for index, line in enumerate(text.splitlines()[:MAX_INDEX_RECORDS]):
            value = _safe_text(line, maximum=MAX_RECORD_BYTES)
            if value:
                records.append({"id": index, "fields": {"text": value}, "search_text": value})
    encoded = len(canonical_json(records).encode("utf-8"))
    if len(records) > MAX_INDEX_RECORDS or encoded > MAX_INDEX_BYTES:
        raise BrokerError("derived_index_bounds_exceeded")
    return records


class PackPrivateStore:
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS external_pack_private_records (
                    pack_id TEXT NOT NULL, version TEXT NOT NULL, actor_id TEXT NOT NULL,
                    namespace TEXT NOT NULL, item_key TEXT NOT NULL, schema_version TEXT NOT NULL,
                    value_json TEXT NOT NULL, value_digest TEXT NOT NULL, created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY(pack_id,version,actor_id,namespace,item_key)
                );
                CREATE INDEX IF NOT EXISTS idx_external_pack_private_namespace
                ON external_pack_private_records(pack_id,version,actor_id,namespace);
                CREATE TABLE IF NOT EXISTS external_pack_private_writes (
                    pack_id TEXT NOT NULL, version TEXT NOT NULL, actor_id TEXT NOT NULL, written_at INTEGER NOT NULL
                );
                """
            )
            self._conn.commit()

    def replace_namespace(self, *, pack_id: str, version: str, actor_id: str, namespace: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if len(rows) > MAX_PRIVATE_ITEMS:
            raise BrokerError("private_store_item_quota_exceeded")
        values: list[tuple[str, str, str]] = []
        total = 0
        for index, row in enumerate(rows):
            payload = canonical_json(row)
            size = len(payload.encode("utf-8"))
            if size > MAX_PRIVATE_VALUE_BYTES:
                raise BrokerError("private_store_value_quota_exceeded")
            total += size
            values.append((str(index), payload, hashlib.sha256(payload.encode()).hexdigest()))
        if total > MAX_PRIVATE_TOTAL_BYTES:
            raise BrokerError("private_store_total_quota_exceeded")
        now = _now()
        with self._lock:
            recent = int(self._conn.execute("SELECT COUNT(*) FROM external_pack_private_writes WHERE pack_id=? AND version=? AND actor_id=? AND written_at>?", (pack_id, version, actor_id, now - 60)).fetchone()[0])
            if recent >= MAX_PRIVATE_WRITES_PER_MINUTE:
                raise BrokerError("private_store_write_rate_exceeded")
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute("DELETE FROM external_pack_private_records WHERE pack_id=? AND version=? AND actor_id=? AND namespace=?", (pack_id, version, actor_id, namespace))
                self._conn.executemany(
                    "INSERT INTO external_pack_private_records(pack_id,version,actor_id,namespace,item_key,schema_version,value_json,value_digest,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    [(pack_id, version, actor_id, namespace, key, PRIVATE_STORE_SCHEMA, value, digest, now, now) for key, value, digest in values],
                )
                self._conn.execute("INSERT INTO external_pack_private_writes(pack_id,version,actor_id,written_at) VALUES(?,?,?,?)", (pack_id, version, actor_id, now))
                self._conn.execute("DELETE FROM external_pack_private_writes WHERE written_at<?", (now - 3600,))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {"ok": True, "items": len(values), "bytes": total, "namespace_digest": contract_digest([item[2] for item in values])}

    def list_namespace(self, *, pack_id: str, version: str, actor_id: str, namespace: str, limit: int = MAX_PRIVATE_ITEMS) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT value_json,value_digest FROM external_pack_private_records WHERE pack_id=? AND version=? AND actor_id=? AND namespace=? ORDER BY CAST(item_key AS INTEGER),item_key LIMIT ?", (pack_id, version, actor_id, namespace, min(MAX_PRIVATE_ITEMS, max(1, int(limit))))).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            value = str(row["value_json"])
            if hashlib.sha256(value.encode()).hexdigest() != str(row["value_digest"]):
                raise BrokerError("private_store_integrity_failed")
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                output.append(parsed)
        return output

    def remove_pack(self, *, pack_id: str, version: str | None = None, actor_id: str | None = None) -> int:
        query = "DELETE FROM external_pack_private_records WHERE pack_id=?"
        params: list[Any] = [pack_id]
        if version is not None:
            query += " AND version=?"; params.append(version)
        if actor_id is not None:
            query += " AND actor_id=?"; params.append(actor_id)
        with self._lock:
            changed = self._conn.execute(query, tuple(params)).rowcount
            self._conn.commit()
        return int(changed)


class SelectedLocalDataBroker:
    def __init__(self, *, store: PackPrivateStore, allowed_roots: tuple[str | Path, ...]) -> None:
        self.store = store
        self.allowed_roots = tuple(Path(root).expanduser() for root in allowed_roots)

    def import_index(self, *, pack_id: str, version: str, actor_id: str, path: str, grant: Mapping[str, Any], max_bytes: int = MAX_LOCAL_FILE_BYTES) -> dict[str, Any]:
        metadata = file_metadata(path, allowed_roots=self.allowed_roots, max_bytes=max_bytes)
        grant_meta = grant.get("path_metadata") if isinstance(grant.get("path_metadata"), Mapping) else {}
        if str(grant.get("state") or "") != "granted" or str(grant.get("pack_id") or "") != pack_id:
            raise BrokerError("selected_file_grant_invalid")
        for field in ("device", "inode", "size_bytes", "mtime_ns", "fingerprint"):
            if str(grant_meta.get(field)) != str(metadata.get(field)):
                raise BrokerError("selected_file_changed_after_grant")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path, flags)
        try:
            row = os.fstat(fd)
            if not stat.S_ISREG(row.st_mode) or row.st_nlink != 1 or int(row.st_dev) != metadata["device"] or int(row.st_ino) != metadata["inode"] or int(row.st_size) != metadata["size_bytes"] or int(row.st_mtime_ns) != metadata["mtime_ns"]:
                raise BrokerError("selected_file_changed_during_open")
            chunks: list[bytes] = []
            remaining = min(max_bytes, int(row.st_size))
            while remaining > 0:
                chunk = os.read(fd, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk); remaining -= len(chunk)
            if sum(map(len, chunks)) != int(row.st_size):
                raise BrokerError("selected_file_truncated")
        finally:
            os.close(fd)
        records = _records_from_bytes(b"".join(chunks), metadata["extension"])
        stored = self.store.replace_namespace(pack_id=pack_id, version=version, actor_id=actor_id, namespace="search-index-v1", rows=records)
        return {"ok": True, "indexed_records": len(records), "source": metadata["path_redacted"], "source_fingerprint": metadata["fingerprint"], "raw_content_retained": False, "store": stored, "verification": {"index_present": True, "item_count": len(records)}}

    def search(self, *, pack_id: str, version: str, actor_id: str, query: str, limit: int = 10) -> dict[str, Any]:
        if not query or len(query.encode("utf-8")) > MAX_QUERY_BYTES:
            raise BrokerError("search_query_invalid")
        terms = [term.casefold() for term in re.findall(r"[\w.-]+", query, flags=re.UNICODE)[:16] if len(term) > 1]
        if not terms:
            raise BrokerError("search_query_empty")
        results: list[dict[str, Any]] = []
        for row in self.store.list_namespace(pack_id=pack_id, version=version, actor_id=actor_id, namespace="search-index-v1"):
            haystack = str(row.get("search_text") or "").casefold()
            score = sum(haystack.count(term) for term in terms)
            if score:
                fields = row.get("fields") if isinstance(row.get("fields"), dict) else {}
                results.append({"record_id": row.get("id"), "score": score, "fields": {str(key)[:120]: _safe_text(value, maximum=400) for key, value in list(fields.items())[:16]}})
        results.sort(key=lambda item: (-int(item["score"]), str(item["record_id"])))
        bounded = results[: min(MAX_SEARCH_RESULTS, max(1, int(limit)))]
        return {"ok": True, "query": _safe_text(query, maximum=MAX_QUERY_BYTES), "matches": bounded, "match_count": len(bounded), "network_used": False, "verification": {"query_terms": terms, "all_results_matched": all(int(item["score"]) > 0 for item in bounded)}}


class ScopedHttpsBroker:
    def __init__(self, *, transport: SafeHttpsTransport | None = None) -> None:
        self.transport = transport or SafeHttpsTransport(allow_query=True)

    def request(self, declaration: BrokerDeclarationV1, *, method: str, path: str, parameters: Mapping[str, Any] | None = None, taint_sources: tuple[str, ...] = ()) -> dict[str, Any]:
        if declaration.kind != "scoped_https":
            raise BrokerError("network_broker_declaration_required")
        if any(source in {"selected_local_data", "pack_private_store", "private"} for source in taint_sources):
            raise BrokerError("private_to_network_composition_denied")
        config = declaration.config
        origins = config.get("origins") or []
        path_templates = config.get("path_templates") or []
        parameter_names = set(config.get("parameter_names") or [])
        methods = set(str(item).upper() for item in (config.get("methods") or ["GET", "HEAD"]))
        if method.upper() not in methods or method.upper() not in {"GET", "HEAD"}:
            raise BrokerError("network_method_denied")
        if not isinstance(origins, list) or len(origins) != 1 or not isinstance(path_templates, list) or path not in path_templates:
            raise BrokerError("network_destination_denied")
        origin = str(origins[0]).rstrip("/")
        parsed_origin = urllib.parse.urlsplit(origin)
        if parsed_origin.scheme != "https" or parsed_origin.path not in {"", "/"} or parsed_origin.query or parsed_origin.fragment:
            raise BrokerError("network_origin_invalid")
        params = dict(parameters or {})
        if set(params) - parameter_names or any(isinstance(value, (dict, list)) for value in params.values()) or len(params) > 16:
            raise BrokerError("network_parameters_denied")
        query = urllib.parse.urlencode({str(key): _safe_text(value, maximum=256) for key, value in params.items()})
        url = f"{origin}{path}" + (f"?{query}" if query else "")
        body, result = self.transport.fetch_bytes(url, method=method, max_bytes=declaration.limits["output_bytes"], allowed_content_types=ALLOWED_PUBLIC_DATA_TYPES, total_timeout=declaration.limits["wall_ms"] / 1000.0)
        preview = body.decode("utf-8", errors="replace")[: declaration.limits["output_bytes"]] if method.upper() == "GET" else ""
        return {"ok": True, "status": result.status, "media_type": result.media_type, "bytes": result.bytes_received, "sha256": result.sha256, "body": preview, "tainted": True, "target": result.final_target, "verification": {"transport_verified": True, "within_scope": True}}


def decoded_raster_metadata(data: bytes, media_type: str) -> tuple[int, int]:
    if media_type == "image/png":
        if len(data) < 57 or len(data) > 8 * 1024 * 1024 or data[:8] != b"\x89PNG\r\n\x1a\n":
            raise BrokerError("visualizer_png_invalid")
        offset, chunks, compressed = 8, 0, bytearray()
        width = height = bit_depth = color_type = 0
        seen_header = seen_end = False
        while offset + 12 <= len(data):
            length = int.from_bytes(data[offset : offset + 4], "big")
            kind = data[offset + 4 : offset + 8]
            end = offset + 12 + length
            if length > 8 * 1024 * 1024 or end > len(data):
                raise BrokerError("visualizer_png_invalid")
            payload = data[offset + 8 : offset + 8 + length]
            checksum = int.from_bytes(data[offset + 8 + length : end], "big")
            if zlib.crc32(kind + payload) & 0xFFFFFFFF != checksum:
                raise BrokerError("visualizer_png_crc_invalid")
            chunks += 1
            if chunks > 256:
                raise BrokerError("visualizer_png_chunk_limit")
            if kind == b"IHDR":
                if seen_header or chunks != 1 or length != 13:
                    raise BrokerError("visualizer_png_invalid")
                seen_header = True
                width, height = int.from_bytes(payload[:4], "big"), int.from_bytes(payload[4:8], "big")
                bit_depth, color_type = payload[8], payload[9]
                if payload[10:] != b"\x00\x00\x00":
                    raise BrokerError("visualizer_png_encoding_unsupported")
            elif kind == b"IDAT":
                if not seen_header or seen_end:
                    raise BrokerError("visualizer_png_invalid")
                compressed.extend(payload)
                if len(compressed) > 8 * 1024 * 1024:
                    raise BrokerError("visualizer_png_compressed_limit")
            elif kind == b"IEND":
                if length != 0 or not seen_header:
                    raise BrokerError("visualizer_png_invalid")
                seen_end = True
                offset = end
                break
            offset = end
        channel_counts = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
        valid_depths = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
        if not seen_end or offset != len(data) or width < 1 or height < 1 or bit_depth not in valid_depths.get(color_type, set()) or not compressed:
            raise BrokerError("visualizer_png_invalid")
        row_bytes = (width * channel_counts[color_type] * bit_depth + 7) // 8
        expected = (row_bytes + 1) * height
        if expected > 64 * 1024 * 1024:
            raise BrokerError("visualizer_png_decoded_limit")
        try:
            decoder = zlib.decompressobj()
            raster = decoder.decompress(bytes(compressed), expected + 1)
            raster += decoder.flush()
        except zlib.error as exc:
            raise BrokerError("visualizer_png_decode_failed") from exc
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or len(raster) != expected:
            raise BrokerError("visualizer_png_decode_failed")
        for row in range(height):
            if raster[row * (row_bytes + 1)] > 4:
                raise BrokerError("visualizer_png_filter_invalid")
        return width, height
    raise BrokerError("visualizer_media_type_unsupported")


class PresenceVisualizerBroker:
    def __init__(self, storage_root: str | Path) -> None:
        self.root = Path(storage_root).expanduser().resolve() / "visualizers-v1"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def install(self, *, pack_id: str, version: str, asset_path: str | Path, declaration: Mapping[str, Any]) -> dict[str, Any]:
        source = Path(asset_path)
        try:
            row = source.stat(follow_symlinks=False)
        except OSError as exc:
            raise BrokerError("visualizer_asset_file_invalid") from exc
        if source.is_symlink() or not stat.S_ISREG(row.st_mode) or row.st_nlink != 1 or row.st_size > 8 * 1024 * 1024:
            raise BrokerError("visualizer_asset_file_invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(source, flags)
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                    row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns
                ):
                    raise BrokerError("visualizer_asset_changed_during_read")
                data = b""
                while len(data) <= 8 * 1024 * 1024:
                    part = os.read(descriptor, min(64 * 1024, 8 * 1024 * 1024 + 1 - len(data)))
                    if not part:
                        break
                    data += part
                if len(data) != row.st_size:
                    raise BrokerError("visualizer_asset_changed_during_read")
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise BrokerError("visualizer_asset_file_invalid") from exc
        draft = dict(declaration)
        asset = dict(draft.get("asset") if isinstance(draft.get("asset"), Mapping) else {})
        media = str(asset.get("media_type") or "")
        width, height = decoded_raster_metadata(data, media)
        asset.update({"bytes": len(data), "width": width, "height": height, "sha256": hashlib.sha256(data).hexdigest()})
        draft["asset"] = asset
        normalized = normalize_visualizer(draft)
        target = self.root / f"{pack_id}-{version}-{normalized['declaration_digest'][:12]}"
        temp = self.root / f".{target.name}.{time.time_ns()}.tmp"
        temp.mkdir(mode=0o700)
        try:
            extension = ".png"
            (temp / f"sprite{extension}").write_bytes(data)
            (temp / f"sprite{extension}").chmod(0o400)
            (temp / "visualizer.json").write_text(canonical_json(normalized) + "\n", encoding="utf-8")
            (temp / "visualizer.json").chmod(0o400)
            if target.exists():
                for child in temp.iterdir(): child.unlink(missing_ok=True)
                temp.rmdir()
            else:
                temp.rename(target)
        finally:
            if temp.exists():
                for child in temp.iterdir():
                    child.unlink(missing_ok=True)
                temp.rmdir()
        return {"ok": True, "pack_id": pack_id, "version": version, "storage_key": target.name, "declaration": normalized, "asset_url": f"/packs/visualizer/{pack_id}/{version}/asset", "enabled": False, "reduced_motion_supported": True, "scripts_allowed": False, "remote_requests": False}

    def activate(self, installed: Mapping[str, Any]) -> dict[str, Any]:
        declaration = installed.get("declaration") if isinstance(installed.get("declaration"), Mapping) else {}
        row = {"schema_version": "personal-agent.pack-visualizer-state.v1", "pack_id": str(installed.get("pack_id") or ""), "version": str(installed.get("version") or ""), "storage_key": str(installed.get("storage_key") or ""), "declaration": declaration, "asset_url": str(installed.get("asset_url") or ""), "enabled": True, "reduced_motion_supported": True, "scripts_allowed": False, "remote_requests": False, "updated_at": _now()}
        temp = self.root / f".active.{time.time_ns()}.tmp"
        temp.write_text(canonical_json(row) + "\n", encoding="utf-8"); temp.chmod(0o600)
        temp.replace(self.root / "active.json")
        return row

    def status(self) -> dict[str, Any]:
        path = self.root / "active.json"
        if not path.is_file():
            return {"available": False, "enabled": False, "reason": "no_visualizer_selected", "fallback": "core_default"}
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            declaration = row.get("declaration") if isinstance(row.get("declaration"), Mapping) else {}
            normalized = normalize_visualizer({key: value for key, value in declaration.items() if key != "declaration_digest"})
            if normalized["declaration_digest"] != declaration.get("declaration_digest"):
                raise BrokerError("visualizer_state_digest_mismatch")
            return {**row, "available": True, "fallback": "core_default"}
        except Exception:
            return {"available": False, "enabled": False, "reason": "visualizer_state_invalid", "fallback": "core_default"}

    def disable_pack(self, pack_id: str) -> bool:
        current = self.status()
        if current.get("pack_id") != pack_id:
            return False
        (self.root / "active.json").unlink(missing_ok=True)
        return True

    def asset(self, pack_id: str, version: str) -> tuple[bytes, str] | None:
        current = self.status()
        if not current.get("available") or current.get("pack_id") != pack_id or current.get("version") != version:
            return None
        key = str(current.get("storage_key") or "")
        if not key or Path(key).name != key:
            return None
        declaration = current.get("declaration") if isinstance(current.get("declaration"), Mapping) else {}
        asset = declaration.get("asset") if isinstance(declaration.get("asset"), Mapping) else {}
        if asset.get("media_type") != "image/png":
            return None
        path = self.root / key / "sprite.png"
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if hashlib.sha256(data).hexdigest() != asset.get("sha256"):
            return None
        return data, str(asset.get("media_type"))


class CoreBrokerRuntime:
    """Dispatch table for exact reviewed broker declarations.

    It is intentionally not a capability registry.  Only a live dynamic pack
    capability can call this object, and only the declaration already bound
    into that capability contract is accepted.
    """

    def __init__(self, *, db_path: str, storage_root: str | Path, allowed_roots: tuple[str | Path, ...]) -> None:
        self.private_store = PackPrivateStore(db_path)
        self.local_data = SelectedLocalDataBroker(store=self.private_store, allowed_roots=allowed_roots)
        self.https = ScopedHttpsBroker()
        self.visualizer = PresenceVisualizerBroker(storage_root)

    def invoke(
        self,
        *,
        pack_id: str,
        version: str,
        actor_id: str,
        declaration: Mapping[str, Any],
        operation: str,
        inputs: Mapping[str, Any],
        grant: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        broker = BrokerDeclarationV1.parse(declaration)
        if broker.kind == "selected_local_data" and operation == "search":
            return self.local_data.search(
                pack_id=pack_id,
                version=version,
                actor_id=actor_id,
                query=str(inputs.get("query") or ""),
                limit=int(inputs.get("limit") or 10),
            )
        if broker.kind == "selected_local_data" and operation == "import_index":
            if grant is None:
                raise BrokerError("selected_file_grant_missing")
            path = str(grant.get("granted_path") or "")
            return self.local_data.import_index(
                pack_id=pack_id,
                version=version,
                actor_id=actor_id,
                path=path,
                grant=grant,
                max_bytes=broker.limits["input_bytes"],
            )
        if broker.kind == "scoped_https" and operation in {"get", "head"}:
            return self.https.request(
                broker,
                method=operation.upper(),
                path=str(inputs.get("path") or ""),
                parameters=inputs.get("parameters") if isinstance(inputs.get("parameters"), Mapping) else {},
                taint_sources=tuple(str(item) for item in (inputs.get("taint_sources") or []) if isinstance(item, str)),
            )
        if broker.kind == "pack_private_store" and operation == "status":
            rows = self.private_store.list_namespace(pack_id=pack_id, version=version, actor_id=actor_id, namespace=str(inputs.get("namespace") or "search-index-v1"), limit=1)
            return {"ok": True, "available": True, "has_data": bool(rows), "raw_values_exposed": False, "verification": {"integrity_checked": True}}
        if broker.kind == "presence_visualizer" and operation == "status":
            return {"ok": True, "available": True, "core_rendered": True, "scripts_allowed": False, "remote_requests": False, "verification": {"declaration_bound": True}}
        raise BrokerError("broker_operation_unsupported")


__all__ = [
    "BrokerError", "CoreBrokerRuntime", "PackPrivateStore", "PresenceVisualizerBroker", "ScopedHttpsBroker",
    "SelectedLocalDataBroker", "decoded_raster_metadata", "file_metadata",
]

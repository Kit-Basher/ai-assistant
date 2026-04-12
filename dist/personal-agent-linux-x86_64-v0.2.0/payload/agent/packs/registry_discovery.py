from __future__ import annotations

import hashlib
import json
import re
import time
import sqlite3
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REGISTRY_KIND_CLAWHUB_TEXT = "clawhub_text_registry"
REGISTRY_KIND_GITHUB_INDEX = "github_repo_index"
REGISTRY_KIND_GENERIC_API = "generic_registry_api"
REGISTRY_KIND_LOCAL_CATALOG = "local_catalog"
ALLOWED_REGISTRY_SOURCE_KINDS = {
    REGISTRY_KIND_CLAWHUB_TEXT,
    REGISTRY_KIND_GITHUB_INDEX,
    REGISTRY_KIND_GENERIC_API,
    REGISTRY_KIND_LOCAL_CATALOG,
}
ALLOWED_ARTIFACT_TYPE_HINTS = {
    "portable_text_skill",
    "experience_pack",
    "native_code_pack",
    "unknown",
}
DEFAULT_DISCOVERY_CACHE_TTL_SECONDS = 300
DEFAULT_DISCOVERY_MAX_RESULTS = 50
MAX_DISCOVERY_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
MIN_DISCOVERY_MAX_RESULTS = 1
MAX_DISCOVERY_MAX_RESULTS = 200
_POLICY_WRITE_FIELDS = {
    "enabled",
    "allowlisted",
    "denied",
    "allowed_source_kinds",
    "cache_ttl_seconds",
    "max_results",
    "notes",
}
_SOURCE_WRITE_FIELDS = {
    "source_id",
    "name",
    "kind",
    "base_url",
    "enabled",
    "discovery_only",
    "supports_search",
    "supports_preview",
    "supports_compare_hint",
    "notes",
}
_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slugify(value: str) -> str:
    lowered = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-") or "pack"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_commit_like(value: str | None) -> bool:
    ref = str(value or "").strip()
    return bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", ref))


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    if isinstance(value, str):
        return sorted({part.strip() for part in value.split(",") if part.strip()})
    return []


@dataclass(frozen=True)
class RegistrySource:
    id: str
    kind: str
    name: str
    base_url: str
    enabled: bool
    discovery_only: bool
    supports_search: bool
    supports_preview: bool
    supports_compare_hint: bool
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegistrySourcePolicy:
    source_id: str
    enabled: bool
    allowlisted: bool
    denied: bool
    allowed_source_kinds: tuple[str, ...]
    cache_ttl_seconds: int
    max_results: int
    notes: str | None = None
    allowed_by_policy: bool = True
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "enabled": bool(self.enabled),
            "allowlisted": bool(self.allowlisted),
            "denied": bool(self.denied),
            "allowed_source_kinds": list(self.allowed_source_kinds),
            "cache_ttl_seconds": int(self.cache_ttl_seconds),
            "max_results": int(self.max_results),
            "notes": self.notes,
            "allowed_by_policy": bool(self.allowed_by_policy),
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True)
class RegistryDiscoveryPolicy:
    defaults: RegistrySourcePolicy
    overrides: tuple[RegistrySourcePolicy, ...]

    def override_for_source(self, source_id: str) -> RegistrySourcePolicy | None:
        normalized = str(source_id or "").strip()
        for row in self.overrides:
            if row.source_id == normalized:
                return row
        return None


@dataclass(frozen=True)
class RegistryBadgeSet:
    badges: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"badges": list(self.badges)}


@dataclass(frozen=True)
class RegistryPackListing:
    registry_source_id: str
    remote_id: str
    name: str
    summary: str
    author: str | None
    homepage_url: str | None
    source_url: str | None
    source_kind_hint: str | None
    latest_ref_hint: str | None
    artifact_type_hint: str
    tags: tuple[str, ...]
    badges: tuple[str, ...]
    last_updated: str | None
    preview_available: bool
    installable_by_current_policy: bool
    install_block_reason_if_known: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegistrySearchResult:
    registry_source_id: str
    query: str
    results: tuple[dict[str, Any], ...]
    count: int
    from_cache: bool
    stale: bool
    fetched_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_source_id": self.registry_source_id,
            "query": self.query,
            "results": list(self.results),
            "count": self.count,
            "from_cache": self.from_cache,
            "stale": self.stale,
            "fetched_at": self.fetched_at,
        }


@dataclass(frozen=True)
class RegistryPackPreview:
    source: dict[str, Any]
    listing: dict[str, Any]
    fetched: bool
    summary: str
    appears_to_do: str
    artifact_type_hint: str
    policy_hint: str
    badges: tuple[str, ...]
    source_hints: tuple[str, ...]
    related_local_pack: dict[str, Any] | None
    compare_hint: dict[str, Any] | None
    install_handoff: dict[str, Any] | None
    choices: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": dict(self.source),
            "listing": dict(self.listing),
            "fetched": bool(self.fetched),
            "summary": self.summary,
            "appears_to_do": self.appears_to_do,
            "artifact_type_hint": self.artifact_type_hint,
            "policy_hint": self.policy_hint,
            "badges": list(self.badges),
            "source_hints": list(self.source_hints),
            "related_local_pack": self.related_local_pack,
            "compare_hint": self.compare_hint,
            "install_handoff": self.install_handoff,
            "choices": list(self.choices),
        }


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        parsed = urllib.parse.urlparse(newurl)
        if str(parsed.scheme or "").lower() != "https":
            raise urllib.error.HTTPError(newurl, code, "redirect_to_non_https_blocked", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class RegistrySourcePolicyError(RuntimeError):
    def __init__(self, source_id: str, policy: RegistrySourcePolicy) -> None:
        self.source_id = str(source_id or "").strip()
        self.policy = policy
        super().__init__(f"registry source blocked by policy: {self.source_id}")


class PackRegistryDiscoveryService:
    def __init__(
        self,
        *,
        pack_store: Any,
        storage_root: str,
        sources_path: str | None = None,
        policy_path: str | None = None,
        opener: Any | None = None,
        lock: threading.RLock | None = None,
    ) -> None:
        self.pack_store = pack_store
        self.storage_root = Path(storage_root).expanduser().resolve()
        self.sources_path = (
            Path(sources_path).expanduser().resolve()
            if str(sources_path or "").strip()
            else (self.storage_root / "registry_sources.json").resolve()
        )
        self.policy_path = (
            Path(policy_path).expanduser().resolve()
            if str(policy_path or "").strip()
            else (self.storage_root / "registry_source_policy.json").resolve()
        )
        self._opener = opener or urllib.request.build_opener(_HttpsOnlyRedirectHandler())
        self._lock = lock or threading.RLock()

    def get_catalog(self) -> dict[str, Any]:
        document, persisted_exists = self._read_sources_document()
        policy = self._load_policy()
        return {
            "path": str(self.sources_path),
            "persisted_exists": bool(persisted_exists),
            "persisted_catalog": document,
            "normalized_sources": [
                self._catalog_source_view(source, policy=policy)
                for source in self._load_sources()
            ],
        }

    def get_catalog_source(self, source_id: str) -> dict[str, Any]:
        source = self._source_by_id(source_id)
        document, persisted_exists = self._read_sources_document()
        policy = self._load_policy()
        return {
            "path": str(self.sources_path),
            "persisted_exists": bool(persisted_exists),
            "persisted_source": self._raw_source_for_id(document, source.id),
            "source": source.to_dict(),
            "effective_policy": self._effective_policy_for_source(source, policy=policy).to_dict(),
            "meta": dict(document.get("meta")) if isinstance(document.get("meta"), dict) else {},
        }

    def create_catalog_source(
        self,
        payload: dict[str, Any],
        *,
        changed_by: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            normalized_row = self._normalize_source_write(payload, create=True)
            existing_ids = {source.id for source in self._load_sources()}
            if str(normalized_row.get("id") or "").strip() in existing_ids:
                raise ValueError("duplicate_source_id")
            document, _ = self._read_sources_document()
            next_sources = self._raw_sources(document)
            next_sources.append(normalized_row)
            next_document = self._with_catalog_audit_update(
                current_document=document,
                next_sources=next_sources,
                changed_by=changed_by,
                changed_fields=sorted(str(key) for key in payload.keys()),
                source_id=str(normalized_row.get("id") or ""),
                operation="create",
            )
            self._write_sources_document(next_document)
            return self.get_catalog_source(str(normalized_row.get("id") or ""))

    def update_catalog_source(
        self,
        source_id: str,
        payload: dict[str, Any],
        *,
        changed_by: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            source = self._source_by_id(source_id)
            normalized = str(source_id or "").strip()
            document, _ = self._read_sources_document()
            current_row = self._raw_source_for_id(document, normalized) or self._source_to_persisted_row(source)
            next_row = self._normalize_source_write(
                payload,
                create=False,
                current_row=current_row,
                source_id=normalized,
            )
            next_sources = [row for row in self._raw_sources(document) if str(row.get("id") or "").strip() != normalized]
            next_sources.append(next_row)
            next_document = self._with_catalog_audit_update(
                current_document=document,
                next_sources=next_sources,
                changed_by=changed_by,
                changed_fields=sorted(str(key) for key in payload.keys()),
                source_id=normalized,
                operation="update",
            )
            self._write_sources_document(next_document)
            return self.get_catalog_source(normalized)

    def delete_catalog_source(self, source_id: str, *, changed_by: str | None = None) -> dict[str, Any]:
        with self._lock:
            source = self._source_by_id(source_id)
            normalized = str(source_id or "").strip()
            document, _ = self._read_sources_document()
            current_row = self._raw_source_for_id(document, normalized) or self._source_to_persisted_row(source)
            next_sources = [row for row in self._raw_sources(document) if str(row.get("id") or "").strip() != normalized]
            next_document = self._with_catalog_audit_update(
                current_document=document,
                next_sources=next_sources,
                changed_by=changed_by,
                changed_fields=sorted({"source_id", *[str(key) for key in current_row.keys()]}),
                source_id=normalized,
                operation="delete",
            )
            self._write_sources_document(next_document)
            policy_cleanup = self._remove_policy_override_for_deleted_source(normalized, changed_by=changed_by)
            return {
                "path": str(self.sources_path),
                "deleted_source_id": normalized,
                "deleted_source": source.to_dict(),
                "persisted_catalog": next_document,
                "policy_override_removed": bool(policy_cleanup.get("removed", False)),
                "policy_cleanup": policy_cleanup,
            }

    def get_policy(self) -> dict[str, Any]:
        policy = self._load_policy()
        document, persisted_exists = self._read_policy_document()
        return {
            "path": str(self.policy_path),
            "persisted_exists": bool(persisted_exists),
            "persisted_policy": document,
            "normalized_policy": self._normalized_policy_dict(policy),
            "effective_sources": [
                {
                    "source": source.to_dict(),
                    "effective_policy": self._effective_policy_for_source(source, policy=policy).to_dict(),
                }
                for source in self._load_sources()
            ],
        }

    def get_source_policy(self, source_id: str) -> dict[str, Any]:
        source = self._source_by_id(source_id)
        policy = self._load_policy()
        document, persisted_exists = self._read_policy_document()
        persisted_override = self._raw_override_for_source(document, source_id)
        effective_policy = self._effective_policy_for_source(source, policy=policy)
        return {
            "path": str(self.policy_path),
            "persisted_exists": bool(persisted_exists),
            "source": source.to_dict(),
            "persisted_override": persisted_override,
            "defaults": policy.defaults.to_dict(),
            "effective_policy": effective_policy.to_dict(),
            "meta": dict(document.get("meta")) if isinstance(document.get("meta"), dict) else {},
        }

    def update_global_policy(self, payload: dict[str, Any], *, changed_by: str | None = None) -> dict[str, Any]:
        with self._lock:
            changed_fields = self._validate_policy_patch(payload)
            document, _ = self._read_policy_document()
            next_defaults = dict(document.get("defaults")) if isinstance(document.get("defaults"), dict) else {}
            for field in changed_fields:
                next_defaults[field] = payload.get(field)
            next_document = self._with_audit_update(
                current_document=document,
                next_defaults=next_defaults,
                next_overrides=self._raw_overrides(document),
                changed_by=changed_by,
                changed_fields=changed_fields,
                scope="defaults",
            )
            self._write_policy_document(next_document)
            return self.get_policy()

    def update_source_policy(
        self,
        source_id: str,
        payload: dict[str, Any],
        *,
        changed_by: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            source = self._source_by_id(source_id)
            changed_fields = self._validate_policy_patch(payload)
            document, _ = self._read_policy_document()
            overrides = self._raw_overrides(document)
            existing = self._raw_override_for_source(document, source.id) or {"source_id": source.id}
            next_override = dict(existing)
            next_override["source_id"] = source.id
            for field in changed_fields:
                next_override[field] = payload.get(field)
            next_overrides = [row for row in overrides if str(row.get("source_id") or "").strip() != source.id]
            next_overrides.append(next_override)
            next_document = self._with_audit_update(
                current_document=document,
                next_defaults=dict(document.get("defaults")) if isinstance(document.get("defaults"), dict) else {},
                next_overrides=next_overrides,
                changed_by=changed_by,
                changed_fields=changed_fields,
                scope=f"source:{source.id}",
            )
            self._write_policy_document(next_document)
            return self.get_source_policy(source.id)

    def list_sources(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        policy = self._load_policy()
        for source in self._load_sources():
            cache_row = self.pack_store.get_registry_source_cache(source.id)
            effective_policy = self._effective_policy_for_source(source, policy=policy)
            out.append(
                {
                    **source.to_dict(),
                    "allowed_by_policy": bool(effective_policy.allowed_by_policy),
                    "queryable": bool(effective_policy.allowed_by_policy),
                    "blocked_reason": effective_policy.blocked_reason,
                    "cache_ttl_seconds": int(effective_policy.cache_ttl_seconds),
                    "max_results": int(effective_policy.max_results),
                    "allowlisted": bool(effective_policy.allowlisted),
                    "denied": bool(effective_policy.denied),
                    "policy": effective_policy.to_dict(),
                    "cache": {
                        "fetched_at": int(cache_row.get("fetched_at") or 0) if isinstance(cache_row, dict) else None,
                        "expires_at": int(cache_row.get("expires_at") or 0) if isinstance(cache_row, dict) else None,
                    },
                }
            )
        return out

    def list_packs(self, source_id: str) -> dict[str, Any]:
        source, effective_policy = self._queryable_source(source_id)
        listings, meta = self._load_listings(source, effective_policy=effective_policy)
        limited = listings[: int(effective_policy.max_results)]
        return {
            "source": source.to_dict(),
            "policy": effective_policy.to_dict(),
            "packs": limited,
            "count": len(limited),
            "from_cache": bool(meta.get("from_cache", False)),
            "stale": bool(meta.get("stale", False)),
            "fetched_at": meta.get("fetched_at"),
        }

    def search(self, source_id: str, query: str) -> dict[str, Any]:
        source, effective_policy = self._queryable_source(source_id)
        normalized_query = str(query or "").strip().lower()
        listings, meta = self._load_listings(source, effective_policy=effective_policy)
        if not normalized_query:
            results = listings
        else:
            results = [
                listing
                for listing in listings
                if normalized_query in str(listing.get("name") or "").lower()
                or normalized_query in str(listing.get("summary") or "").lower()
                or normalized_query in str(listing.get("author") or "").lower()
                or any(normalized_query in str(tag).lower() for tag in (listing.get("tags") if isinstance(listing.get("tags"), list) else []))
                or normalized_query in str(listing.get("remote_id") or "").lower()
            ]
        results = results[: int(effective_policy.max_results)]
        result = RegistrySearchResult(
            registry_source_id=source.id,
            query=query,
            results=tuple(results),
            count=len(results),
            from_cache=bool(meta.get("from_cache", False)),
            stale=bool(meta.get("stale", False)),
            fetched_at=str(meta.get("fetched_at") or "").strip() or None,
        )
        return {
            "source": source.to_dict(),
            "policy": effective_policy.to_dict(),
            "search": result.to_dict(),
        }

    def preview(self, source_id: str, remote_id: str) -> dict[str, Any]:
        source, effective_policy = self._queryable_source(source_id)
        listings, meta = self._load_listings(source, effective_policy=effective_policy, persist_cache=False)
        listing = next((row for row in listings if str(row.get("remote_id") or "") == remote_id), None)
        if listing is None:
            raise KeyError(f"registry listing not found: {source_id}/{remote_id}")
        preview = self._build_preview(source=source, listing=listing)
        return {
            "source": source.to_dict(),
            "policy": effective_policy.to_dict(),
            "preview": preview.to_dict(),
            "from_cache": bool(meta.get("from_cache", False)),
            "stale": bool(meta.get("stale", False)),
            "fetched_at": meta.get("fetched_at"),
        }

    def _load_sources(self) -> list[RegistrySource]:
        document, persisted_exists = self._read_sources_document()
        if persisted_exists:
            return [self._normalize_source(row) for row in self._raw_sources(document) if isinstance(row, dict)]
        local_catalog_path = self.storage_root / "registry_catalog.json"
        if local_catalog_path.exists():
            return [
                RegistrySource(
                    id="local_catalog",
                    kind=REGISTRY_KIND_LOCAL_CATALOG,
                    name="Local Catalog",
                    base_url=str(local_catalog_path),
                    enabled=True,
                    discovery_only=True,
                    supports_search=True,
                    supports_preview=True,
                    supports_compare_hint=True,
                    notes="Local discovery catalog for development and tests.",
                )
            ]
        return []

    def _read_sources_document(self) -> tuple[dict[str, Any], bool]:
        with self._lock:
            if not self.sources_path.exists():
                return {"sources": [], "meta": {}}, False
            try:
                raw = json.loads(self.sources_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            if isinstance(raw, list):
                return {"sources": [dict(row) for row in raw if isinstance(row, dict)], "meta": {}}, True
            if not isinstance(raw, dict):
                raw = {}
            return {
                "sources": self._raw_sources(raw),
                "meta": dict(raw.get("meta")) if isinstance(raw.get("meta"), dict) else {},
            }, True

    @staticmethod
    def _raw_sources(document: dict[str, Any]) -> list[dict[str, Any]]:
        rows = document.get("sources") if isinstance(document, dict) else None
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
        return []

    @staticmethod
    def _raw_source_for_id(document: dict[str, Any], source_id: str) -> dict[str, Any] | None:
        normalized = str(source_id or "").strip()
        for row in PackRegistryDiscoveryService._raw_sources(document):
            if str(row.get("id") or row.get("source_id") or "").strip() == normalized:
                return dict(row)
        return None

    @staticmethod
    def _source_to_persisted_row(source: RegistrySource) -> dict[str, Any]:
        return {
            "id": source.id,
            "name": source.name,
            "kind": source.kind,
            "base_url": source.base_url,
            "enabled": bool(source.enabled),
            "discovery_only": bool(source.discovery_only),
            "supports_search": bool(source.supports_search),
            "supports_preview": bool(source.supports_preview),
            "supports_compare_hint": bool(source.supports_compare_hint),
            "notes": source.notes,
        }

    def _catalog_source_view(
        self,
        source: RegistrySource,
        *,
        policy: RegistryDiscoveryPolicy,
    ) -> dict[str, Any]:
        effective_policy = self._effective_policy_for_source(source, policy=policy)
        return {
            "source": source.to_dict(),
            "effective_policy": effective_policy.to_dict(),
            "queryable": bool(effective_policy.allowed_by_policy),
            "blocked_reason": effective_policy.blocked_reason,
        }

    def _source_by_id(self, source_id: str) -> RegistrySource:
        for source in self._load_sources():
            if source.id == source_id:
                return source
        raise KeyError(f"registry source not found: {source_id}")

    def _queryable_source(self, source_id: str) -> tuple[RegistrySource, RegistrySourcePolicy]:
        source = self._source_by_id(source_id)
        effective_policy = self._effective_policy_for_source(source, policy=self._load_policy())
        if not effective_policy.allowed_by_policy:
            raise RegistrySourcePolicyError(source.id, effective_policy)
        return source, effective_policy

    def _normalize_source(self, row: dict[str, Any]) -> RegistrySource:
        kind = str(row.get("kind") or "").strip().lower()
        if kind not in ALLOWED_REGISTRY_SOURCE_KINDS:
            kind = REGISTRY_KIND_GENERIC_API
        source_id = str(row.get("id") or row.get("source_id") or "").strip() or _slugify(str(row.get("name") or row.get("base_url") or kind))
        return RegistrySource(
            id=source_id,
            kind=kind,
            name=str(row.get("name") or source_id).strip() or source_id,
            base_url=str(row.get("base_url") or "").strip(),
            enabled=bool(row.get("enabled", True)),
            discovery_only=True,
            supports_search=bool(row.get("supports_search", True)),
            supports_preview=bool(row.get("supports_preview", True)),
            supports_compare_hint=bool(row.get("supports_compare_hint", True)),
            notes=str(row.get("notes") or "").strip() or None,
        )

    def _stable_sources_payload(self, document: dict[str, Any]) -> dict[str, Any]:
        rows = [
            self._source_to_persisted_row(self._normalize_source(row))
            for row in self._raw_sources(document)
        ]
        return {
            "sources": sorted(rows, key=lambda row: str(row.get("id") or "")),
        }

    def _catalog_hash(self, document: dict[str, Any]) -> str:
        return _hash_text(_safe_json(self._stable_sources_payload(document)))

    def _with_catalog_audit_update(
        self,
        *,
        current_document: dict[str, Any],
        next_sources: list[dict[str, Any]],
        changed_by: str | None,
        changed_fields: list[str],
        source_id: str,
        operation: str,
    ) -> dict[str, Any]:
        previous_hash = self._catalog_hash(current_document)
        next_document = {
            "sources": sorted(
                [dict(row) for row in next_sources if str(row.get("id") or "").strip()],
                key=lambda row: str(row.get("id") or ""),
            ),
            "meta": dict(current_document.get("meta")) if isinstance(current_document.get("meta"), dict) else {},
        }
        new_hash = self._catalog_hash(next_document)
        change_entry = {
            "changed_at": _now_iso(),
            "changed_by": str(changed_by or "").strip() or "loopback_operator",
            "operation": str(operation or "update"),
            "source_id": str(source_id or "").strip(),
            "changed_fields": sorted({str(field).strip() for field in changed_fields if str(field).strip()}),
            "previous_catalog_hash": previous_hash,
            "new_catalog_hash": new_hash,
        }
        meta = dict(next_document.get("meta")) if isinstance(next_document.get("meta"), dict) else {}
        change_log = meta.get("change_log") if isinstance(meta.get("change_log"), list) else []
        normalized_log = [dict(item) for item in change_log if isinstance(item, dict)]
        normalized_log.append(change_entry)
        meta["last_change"] = change_entry
        meta["change_log"] = normalized_log[-20:]
        next_document["meta"] = meta
        return next_document

    def _write_sources_document(self, document: dict[str, Any]) -> None:
        with self._lock:
            payload = {
                "sources": self._normalize_sources_document(document),
                "meta": dict(document.get("meta")) if isinstance(document.get("meta"), dict) else {},
            }
            self.sources_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.sources_path.with_name(f"{self.sources_path.name}.tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp_path.replace(self.sources_path)

    def _normalize_sources_document(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            self._normalize_source_write(
                {
                    "source_id": str(row.get("id") or row.get("source_id") or "").strip(),
                    "name": row.get("name"),
                    "kind": row.get("kind"),
                    "base_url": row.get("base_url"),
                    "enabled": row.get("enabled", True),
                    "discovery_only": row.get("discovery_only", True),
                    "supports_search": row.get("supports_search", True),
                    "supports_preview": row.get("supports_preview", True),
                    "supports_compare_hint": row.get("supports_compare_hint", True),
                    "notes": row.get("notes"),
                },
                create=False,
                current_row={},
                source_id=str(row.get("id") or row.get("source_id") or "").strip(),
            )
            for row in self._raw_sources(document)
        ]

    def _normalize_source_write(
        self,
        payload: dict[str, Any],
        *,
        create: bool,
        current_row: dict[str, Any] | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("invalid_source_payload")
        unknown = sorted(set(str(key) for key in payload.keys()) - _SOURCE_WRITE_FIELDS)
        if unknown:
            raise ValueError("unknown_source_fields")
        if not payload:
            raise ValueError("empty_source_patch")
        existing = dict(current_row or {})
        if not create and "source_id" in payload:
            requested_source_id = str(payload.get("source_id") or "").strip()
            if requested_source_id != str(source_id or "").strip():
                raise ValueError("source_id_rename_not_supported")
        merged = dict(existing)
        merged["id"] = (
            str(payload.get("source_id") or "").strip()
            if create
            else str(source_id or existing.get("id") or existing.get("source_id") or "").strip()
        )
        for key, value in payload.items():
            if key == "source_id":
                continue
            merged[key] = value
        self._validate_source_fields(merged, create=create)
        return {
            "id": str(merged.get("id") or "").strip(),
            "name": str(merged.get("name") or "").strip(),
            "kind": str(merged.get("kind") or "").strip().lower(),
            "base_url": str(merged.get("base_url") or "").strip(),
            "enabled": bool(merged.get("enabled", True)),
            "discovery_only": True,
            "supports_search": bool(merged.get("supports_search", True)),
            "supports_preview": bool(merged.get("supports_preview", True)),
            "supports_compare_hint": bool(merged.get("supports_compare_hint", True)),
            "notes": str(merged.get("notes") or "").strip() or None,
        }

    def _validate_source_fields(self, row: dict[str, Any], *, create: bool) -> None:
        normalized_id = str(row.get("id") or row.get("source_id") or "").strip()
        if not normalized_id or not _SOURCE_ID_RE.fullmatch(normalized_id):
            raise ValueError("invalid_source_id")
        kind = str(row.get("kind") or "").strip().lower()
        if kind not in ALLOWED_REGISTRY_SOURCE_KINDS:
            raise ValueError("invalid_source_kind")
        if not str(row.get("name") or "").strip():
            raise ValueError("invalid_source_name")
        base_url = str(row.get("base_url") or "").strip()
        if not base_url:
            raise ValueError("invalid_base_url")
        self._validate_source_base_url(kind=kind, base_url=base_url)
        for key in {"enabled", "supports_search", "supports_preview", "supports_compare_hint"}:
            if key in row and not isinstance(row.get(key), bool):
                raise ValueError(f"invalid_source_field_type:{key}")
        if "discovery_only" in row and row.get("discovery_only") is not True:
            raise ValueError("source_must_be_discovery_only")
        if "notes" in row and row.get("notes") is not None and not isinstance(row.get("notes"), str):
            raise ValueError("invalid_notes")
        if create:
            for required_field in ("id", "name", "kind", "base_url"):
                if not str(row.get(required_field) or "").strip():
                    raise ValueError("missing_required_source_fields")

    @staticmethod
    def _validate_source_base_url(*, kind: str, base_url: str) -> None:
        parsed = urllib.parse.urlparse(str(base_url or "").strip())
        scheme = str(parsed.scheme or "").lower()
        if kind == REGISTRY_KIND_LOCAL_CATALOG:
            if scheme in {"http", "https", "ssh", "git+ssh", "ftp"} or parsed.netloc:
                raise ValueError("invalid_base_url")
            return
        if scheme != "https" or parsed.username or parsed.password or not parsed.netloc:
            raise ValueError("invalid_base_url")

    def _remove_policy_override_for_deleted_source(self, source_id: str, *, changed_by: str | None = None) -> dict[str, Any]:
        document, persisted_exists = self._read_policy_document()
        if not persisted_exists:
            return {"removed": False}
        overrides = self._raw_overrides(document)
        next_overrides = [row for row in overrides if str(row.get("source_id") or "").strip() != str(source_id or "").strip()]
        if len(next_overrides) == len(overrides):
            return {"removed": False}
        next_document = self._with_audit_update(
            current_document=document,
            next_defaults=dict(document.get("defaults")) if isinstance(document.get("defaults"), dict) else {},
            next_overrides=next_overrides,
            changed_by=changed_by,
            changed_fields=["source_id"],
            scope=f"source:{str(source_id or '').strip()}:catalog_delete_cleanup",
        )
        self._write_policy_document(next_document)
        meta = next_document.get("meta") if isinstance(next_document.get("meta"), dict) else {}
        return {
            "removed": True,
            "last_change": dict(meta.get("last_change")) if isinstance(meta.get("last_change"), dict) else None,
        }

    def _read_policy_document(self) -> tuple[dict[str, Any], bool]:
        with self._lock:
            if not self.policy_path.exists():
                return {"defaults": {}, "overrides": [], "meta": {}}, False
            try:
                raw = json.loads(self.policy_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            if not isinstance(raw, dict):
                raw = {}
            return {
                "defaults": dict(raw.get("defaults")) if isinstance(raw.get("defaults"), dict) else {},
                "overrides": self._raw_overrides(raw),
                "meta": dict(raw.get("meta")) if isinstance(raw.get("meta"), dict) else {},
            }, True

    @staticmethod
    def _raw_overrides(document: dict[str, Any]) -> list[dict[str, Any]]:
        overrides = document.get("overrides")
        if isinstance(overrides, dict):
            return [
                {"source_id": key, **value}
                for key, value in sorted(overrides.items(), key=lambda item: str(item[0]))
                if isinstance(value, dict)
            ]
        if isinstance(overrides, list):
            return [dict(row) for row in overrides if isinstance(row, dict)]
        return []

    @staticmethod
    def _raw_override_for_source(document: dict[str, Any], source_id: str) -> dict[str, Any] | None:
        normalized = str(source_id or "").strip()
        for row in PackRegistryDiscoveryService._raw_overrides(document):
            if str(row.get("source_id") or "").strip() == normalized:
                return dict(row)
        return None

    def _normalized_policy_dict(self, policy: RegistryDiscoveryPolicy) -> dict[str, Any]:
        return {
            "defaults": policy.defaults.to_dict(),
            "overrides": [row.to_dict() for row in policy.overrides if row.source_id],
        }

    def _stable_policy_payload(self, document: dict[str, Any]) -> dict[str, Any]:
        defaults = self._normalize_policy_defaults(document.get("defaults") if isinstance(document, dict) else None)
        overrides = [
            self._normalize_policy_override(row, defaults=defaults).to_dict()
            for row in self._raw_overrides(document if isinstance(document, dict) else {})
            if str(row.get("source_id") or "").strip()
        ]
        for row in overrides:
            row.pop("allowed_by_policy", None)
            row.pop("blocked_reason", None)
        defaults_payload = defaults.to_dict()
        defaults_payload.pop("allowed_by_policy", None)
        defaults_payload.pop("blocked_reason", None)
        return {
            "defaults": defaults_payload,
            "overrides": sorted(overrides, key=lambda row: str(row.get("source_id") or "")),
        }

    def _policy_hash(self, document: dict[str, Any]) -> str:
        return _hash_text(_safe_json(self._stable_policy_payload(document)))

    def _with_audit_update(
        self,
        *,
        current_document: dict[str, Any],
        next_defaults: dict[str, Any],
        next_overrides: list[dict[str, Any]],
        changed_by: str | None,
        changed_fields: list[str],
        scope: str,
    ) -> dict[str, Any]:
        previous_hash = self._policy_hash(current_document)
        next_document = {
            "defaults": dict(next_defaults),
            "overrides": sorted(
                [
                    dict(row)
                    for row in next_overrides
                    if str(row.get("source_id") or "").strip()
                ],
                key=lambda row: str(row.get("source_id") or ""),
            ),
            "meta": dict(current_document.get("meta")) if isinstance(current_document.get("meta"), dict) else {},
        }
        new_hash = self._policy_hash(next_document)
        change_entry = {
            "changed_at": _now_iso(),
            "changed_by": str(changed_by or "").strip() or "loopback_operator",
            "scope": scope,
            "changed_fields": sorted({str(field).strip() for field in changed_fields if str(field).strip()}),
            "previous_policy_hash": previous_hash,
            "new_policy_hash": new_hash,
        }
        meta = dict(next_document.get("meta")) if isinstance(next_document.get("meta"), dict) else {}
        change_log = meta.get("change_log") if isinstance(meta.get("change_log"), list) else []
        normalized_log = [dict(item) for item in change_log if isinstance(item, dict)]
        normalized_log.append(change_entry)
        meta["last_change"] = change_entry
        meta["change_log"] = normalized_log[-20:]
        next_document["meta"] = meta
        return next_document

    def _write_policy_document(self, document: dict[str, Any]) -> None:
        with self._lock:
            payload = {
                "defaults": dict(document.get("defaults")) if isinstance(document.get("defaults"), dict) else {},
                "overrides": self._raw_overrides(document),
                "meta": dict(document.get("meta")) if isinstance(document.get("meta"), dict) else {},
            }
            self._load_policy_from_document(payload)
            self.policy_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.policy_path.with_name(f"{self.policy_path.name}.tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp_path.replace(self.policy_path)

    def _validate_policy_patch(self, payload: dict[str, Any]) -> list[str]:
        if not isinstance(payload, dict):
            raise ValueError("invalid_policy_payload")
        unknown = sorted(set(str(key) for key in payload.keys()) - _POLICY_WRITE_FIELDS)
        if unknown:
            raise ValueError("unknown_policy_fields")
        if not payload:
            raise ValueError("empty_policy_patch")
        changed_fields = sorted(str(key) for key in payload.keys())
        self._validate_policy_fields(payload)
        return changed_fields

    def _validate_policy_fields(self, payload: dict[str, Any]) -> None:
        for key, value in payload.items():
            if key in {"enabled", "allowlisted", "denied"} and not isinstance(value, bool):
                raise ValueError(f"invalid_policy_field_type:{key}")
            if key == "allowed_source_kinds":
                if not isinstance(value, list) or not value:
                    raise ValueError("invalid_allowed_source_kinds")
                normalized = self._normalize_allowed_source_kinds(value)
                if len(normalized) != len([item for item in value if isinstance(item, str)]):
                    raise ValueError("invalid_allowed_source_kinds")
            if key == "cache_ttl_seconds":
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError("invalid_cache_ttl_seconds")
                if value < 0 or value > MAX_DISCOVERY_CACHE_TTL_SECONDS:
                    raise ValueError("invalid_cache_ttl_seconds")
            if key == "max_results":
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError("invalid_max_results")
                if value < MIN_DISCOVERY_MAX_RESULTS or value > MAX_DISCOVERY_MAX_RESULTS:
                    raise ValueError("invalid_max_results")
            if key == "notes" and value is not None and not isinstance(value, str):
                raise ValueError("invalid_notes")

    def _load_policy(self) -> RegistryDiscoveryPolicy:
        raw, _ = self._read_policy_document()
        return self._load_policy_from_document(raw)

    def _load_policy_from_document(self, raw: dict[str, Any]) -> RegistryDiscoveryPolicy:
        defaults = self._normalize_policy_defaults(raw.get("defaults") if isinstance(raw, dict) else None)
        overrides = tuple(
            self._normalize_policy_override(row, defaults=defaults)
            for row in self._raw_overrides(raw)
            if str(row.get("source_id") or "").strip()
        )
        return RegistryDiscoveryPolicy(defaults=defaults, overrides=overrides)

    def _normalize_policy_defaults(self, raw: Any) -> RegistrySourcePolicy:
        row = raw if isinstance(raw, dict) else {}
        allowed_source_kinds = self._normalize_allowed_source_kinds(row.get("allowed_source_kinds"))
        cache_ttl_seconds = self._normalize_policy_int(row.get("cache_ttl_seconds"), DEFAULT_DISCOVERY_CACHE_TTL_SECONDS)
        max_results = self._normalize_policy_int(row.get("max_results"), DEFAULT_DISCOVERY_MAX_RESULTS)
        return RegistrySourcePolicy(
            source_id="*",
            enabled=bool(row.get("enabled", True)),
            allowlisted=bool(row.get("allowlisted", True)),
            denied=bool(row.get("denied", False)),
            allowed_source_kinds=allowed_source_kinds,
            cache_ttl_seconds=cache_ttl_seconds,
            max_results=max_results,
            notes=str(row.get("notes") or "").strip() or None,
            allowed_by_policy=True,
            blocked_reason=None,
        )

    def _normalize_policy_override(
        self,
        row: dict[str, Any],
        *,
        defaults: RegistrySourcePolicy,
    ) -> RegistrySourcePolicy:
        source_id = str(row.get("source_id") or row.get("id") or "").strip()
        if not source_id:
            return RegistrySourcePolicy(
                source_id="",
                enabled=defaults.enabled,
                allowlisted=defaults.allowlisted,
                denied=defaults.denied,
                allowed_source_kinds=defaults.allowed_source_kinds,
                cache_ttl_seconds=defaults.cache_ttl_seconds,
                max_results=defaults.max_results,
                notes=defaults.notes,
                allowed_by_policy=True,
                blocked_reason=None,
            )
        allowed_source_kinds = (
            self._normalize_allowed_source_kinds(row.get("allowed_source_kinds"))
            if "allowed_source_kinds" in row
            else defaults.allowed_source_kinds
        )
        return RegistrySourcePolicy(
            source_id=source_id,
            enabled=bool(row.get("enabled", defaults.enabled)),
            allowlisted=bool(row.get("allowlisted", defaults.allowlisted)),
            denied=bool(row.get("denied", defaults.denied)),
            allowed_source_kinds=allowed_source_kinds,
            cache_ttl_seconds=(
                self._normalize_policy_int(row.get("cache_ttl_seconds"), defaults.cache_ttl_seconds)
                if "cache_ttl_seconds" in row
                else defaults.cache_ttl_seconds
            ),
            max_results=(
                self._normalize_policy_int(row.get("max_results"), defaults.max_results)
                if "max_results" in row
                else defaults.max_results
            ),
            notes=str(row.get("notes") or defaults.notes or "").strip() or None,
            allowed_by_policy=True,
            blocked_reason=None,
        )

    @staticmethod
    def _normalize_allowed_source_kinds(value: Any) -> tuple[str, ...]:
        if value is None:
            return tuple(sorted(ALLOWED_REGISTRY_SOURCE_KINDS))
        normalized = tuple(
            item
            for item in _normalize_string_list(value)
            if item in ALLOWED_REGISTRY_SOURCE_KINDS
        )
        return normalized

    @staticmethod
    def _normalize_policy_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        return max(0, parsed)

    def _effective_policy_for_source(
        self,
        source: RegistrySource,
        *,
        policy: RegistryDiscoveryPolicy,
    ) -> RegistrySourcePolicy:
        override = policy.override_for_source(source.id)
        base = override or policy.defaults
        blocked_reason: str | None = None
        allowed_by_policy = True
        if not bool(source.enabled):
            allowed_by_policy = False
            blocked_reason = "source_disabled"
        elif not bool(base.enabled):
            allowed_by_policy = False
            blocked_reason = "source_disabled_by_policy"
        elif bool(base.denied):
            allowed_by_policy = False
            blocked_reason = "source_denied_by_policy"
        elif not bool(base.allowlisted):
            allowed_by_policy = False
            blocked_reason = "source_not_allowlisted"
        elif source.kind not in set(base.allowed_source_kinds):
            allowed_by_policy = False
            blocked_reason = "source_kind_not_allowed"
        return RegistrySourcePolicy(
            source_id=source.id,
            enabled=bool(base.enabled),
            allowlisted=bool(base.allowlisted),
            denied=bool(base.denied),
            allowed_source_kinds=tuple(base.allowed_source_kinds),
            cache_ttl_seconds=max(0, int(base.cache_ttl_seconds)),
            max_results=max(1, int(base.max_results or DEFAULT_DISCOVERY_MAX_RESULTS)),
            notes=base.notes,
            allowed_by_policy=allowed_by_policy,
            blocked_reason=blocked_reason,
        )

    def _load_listings(
        self,
        source: RegistrySource,
        *,
        effective_policy: RegistrySourcePolicy,
        persist_cache: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        now_ts = int(time.time())
        cached = self._get_registry_source_cache_with_retry(source.id)
        if isinstance(cached, dict) and int(cached.get("expires_at") or 0) >= now_ts:
            return list(cached.get("listings") if isinstance(cached.get("listings"), list) else []), {
                "from_cache": True,
                "stale": False,
                "fetched_at": self._ts_to_iso(int(cached.get("fetched_at") or 0)),
            }
        try:
            raw_catalog = self._fetch_catalog(source)
            listings = self._normalize_catalog(source, raw_catalog)
            if persist_cache:
                cached = self.pack_store.set_registry_source_cache(
                    source_id=source.id,
                    source_payload=source.to_dict(),
                    listings_payload=listings,
                    ttl_seconds=int(effective_policy.cache_ttl_seconds),
                )
            else:
                cached = {
                    "listings": list(listings),
                    "fetched_at": now_ts,
                    "expires_at": now_ts + max(int(effective_policy.cache_ttl_seconds), 0),
                }
            return list(cached.get("listings") if isinstance(cached.get("listings"), list) else []), {
                "from_cache": False,
                "stale": False,
                "fetched_at": self._ts_to_iso(int(cached.get("fetched_at") or 0)),
            }
        except Exception:
            if isinstance(cached, dict):
                return list(cached.get("listings") if isinstance(cached.get("listings"), list) else []), {
                    "from_cache": True,
                    "stale": True,
                    "fetched_at": self._ts_to_iso(int(cached.get("fetched_at") or 0)),
                }
            raise

    def _get_registry_source_cache_with_retry(self, source_id: str) -> dict[str, Any] | None:
        delay_seconds = 0.05
        for attempt in range(3):
            try:
                return self.pack_store.get_registry_source_cache(source_id)
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "locked" not in message and "busy" not in message:
                    raise
                if attempt >= 2:
                    return None
                time.sleep(delay_seconds * (attempt + 1))
        return None

    def _fetch_catalog(self, source: RegistrySource) -> Any:
        if source.kind == REGISTRY_KIND_LOCAL_CATALOG:
            payload = json.loads(Path(source.base_url).read_text(encoding="utf-8"))
            return payload
        parsed = urllib.parse.urlparse(source.base_url)
        if str(parsed.scheme or "").lower() != "https":
            raise ValueError("registry_source_requires_https")
        if parsed.username or parsed.password:
            raise ValueError("registry_source_auth_not_supported")
        request = urllib.request.Request(
            source.base_url,
            headers={
                "User-Agent": "Personal-Agent/pack-discovery",
                "Accept": "application/json",
            },
        )
        with self._opener.open(request, timeout=15) as response:
            final_url = str(response.geturl() or source.base_url)
            if str(urllib.parse.urlparse(final_url).scheme or "").lower() != "https":
                raise ValueError("registry_source_redirected_to_non_https")
            body = response.read(512 * 1024)
        return json.loads(body.decode("utf-8"))

    def _normalize_catalog(self, source: RegistrySource, raw_catalog: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]]
        if isinstance(raw_catalog, list):
            rows = [row for row in raw_catalog if isinstance(row, dict)]
        elif isinstance(raw_catalog, dict):
            candidates = raw_catalog.get("packs") or raw_catalog.get("results") or raw_catalog.get("items") or []
            rows = [row for row in candidates if isinstance(row, dict)]
        else:
            rows = []
        return [self._normalize_listing(source, row).to_dict() for row in rows]

    def _normalize_listing(self, source: RegistrySource, row: dict[str, Any]) -> RegistryPackListing:
        source_url = (
            str(row.get("source_url") or row.get("repo_url") or row.get("artifact_url") or row.get("url") or "").strip() or None
        )
        latest_ref_hint = str(row.get("latest_ref_hint") or row.get("ref") or row.get("latest_ref") or row.get("commit") or "").strip() or None
        artifact_type_hint = self._artifact_type_hint(row)
        source_kind_hint = (
            str(row.get("source_kind_hint") or "").strip().lower()
            or self._source_kind_hint(source_url)
        )
        installable_by_current_policy = bool(
            artifact_type_hint == "portable_text_skill"
            and source_url
            and (
                source.kind == REGISTRY_KIND_LOCAL_CATALOG
                or str(urllib.parse.urlparse(source_url).scheme or "").lower() == "https"
            )
        )
        install_block_reason: str | None = None
        if not installable_by_current_policy:
            if artifact_type_hint == "native_code_pack":
                install_block_reason = "likely_native_code_pack"
            elif artifact_type_hint == "experience_pack":
                install_block_reason = "experience_packs_not_supported_yet"
            elif not source_url:
                install_block_reason = "missing_source_url"
            else:
                install_block_reason = "unknown_artifact_type"
        listing = RegistryPackListing(
            registry_source_id=source.id,
            remote_id=str(row.get("remote_id") or row.get("id") or row.get("slug") or _slugify(str(source_url or row.get("name") or "pack"))),
            name=str(row.get("name") or row.get("title") or row.get("remote_id") or "Unnamed Pack").strip() or "Unnamed Pack",
            summary=str(row.get("summary") or row.get("description") or "").strip(),
            author=str(row.get("author") or row.get("publisher") or "").strip() or None,
            homepage_url=str(row.get("homepage_url") or row.get("homepage") or "").strip() or None,
            source_url=source_url,
            source_kind_hint=source_kind_hint,
            latest_ref_hint=latest_ref_hint,
            artifact_type_hint=artifact_type_hint,
            tags=tuple(_normalize_string_list(row.get("tags"))),
            badges=(),
            last_updated=str(row.get("last_updated") or row.get("updated_at") or "").strip() or None,
            preview_available=True,
            installable_by_current_policy=installable_by_current_policy,
            install_block_reason_if_known=install_block_reason,
        )
        badge_set = self._badge_set(listing.to_dict(), related_local_pack=self._find_related_local_pack(listing.to_dict()))
        return RegistryPackListing(
            registry_source_id=listing.registry_source_id,
            remote_id=listing.remote_id,
            name=listing.name,
            summary=listing.summary,
            author=listing.author,
            homepage_url=listing.homepage_url,
            source_url=listing.source_url,
            source_kind_hint=listing.source_kind_hint,
            latest_ref_hint=listing.latest_ref_hint,
            artifact_type_hint=listing.artifact_type_hint,
            tags=listing.tags,
            badges=badge_set.badges,
            last_updated=listing.last_updated,
            preview_available=listing.preview_available,
            installable_by_current_policy=listing.installable_by_current_policy,
            install_block_reason_if_known=listing.install_block_reason_if_known,
        )

    def _artifact_type_hint(self, row: dict[str, Any]) -> str:
        explicit = str(row.get("artifact_type_hint") or row.get("artifact_type") or "").strip().lower()
        if explicit in ALLOWED_ARTIFACT_TYPE_HINTS:
            return explicit
        if bool(row.get("requires_execution")) or bool(row.get("has_package_manifest")) or bool(row.get("is_plugin")):
            return "native_code_pack"
        if bool(row.get("is_experience_pack")) or str(row.get("format") or "").strip().lower() == "experience":
            return "experience_pack"
        if bool(row.get("has_skill_md")) and not bool(row.get("requires_execution")):
            return "portable_text_skill"
        return "unknown"

    @staticmethod
    def _source_kind_hint(source_url: str | None) -> str | None:
        url = str(source_url or "").strip()
        if not url:
            return None
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc.lower() == "github.com" and re.search(r"/archive/[^/]+\.(zip|tar|tar\.gz|tgz)$", parsed.path, flags=re.IGNORECASE):
            return "github_archive"
        if parsed.netloc.lower() == "github.com":
            return "github_repo"
        if str(parsed.scheme or "").lower() == "https":
            return "generic_archive_url"
        return None

    def _find_related_local_pack(self, listing: dict[str, Any]) -> dict[str, Any] | None:
        source_url = str(listing.get("source_url") or "").strip()
        if not source_url:
            return None
        related: dict[str, Any] | None = None
        for row in self.pack_store.list_external_packs():
            row_source = row.get("source") if isinstance(row.get("source"), dict) else {}
            if str(row_source.get("url") or "").strip() == source_url:
                related = row
            for entry in row.get("source_history") if isinstance(row.get("source_history"), list) else []:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("url") or "").strip() == source_url:
                    related = row
            if related is not None and int(row.get("updated_at") or 0) > int(related.get("updated_at") or 0):
                related = row
        return related

    def _badge_set(self, listing: dict[str, Any], *, related_local_pack: dict[str, Any] | None) -> RegistryBadgeSet:
        badges: list[str] = []
        artifact_type = str(listing.get("artifact_type_hint") or "unknown")
        if artifact_type == "portable_text_skill":
            badges.append("portable_text_skill")
        elif artifact_type == "experience_pack":
            badges.append("experience_pack")
        elif artifact_type == "native_code_pack":
            badges.append("native_code_pack")
        else:
            badges.append("unknown_type")
        if bool(listing.get("installable_by_current_policy", False)):
            badges.append("installable_by_current_policy")
        else:
            badges.append("blocked_by_current_policy")
        if not _is_commit_like(str(listing.get("latest_ref_hint") or "").strip() or None):
            badges.append("remote_unpinned")
        if related_local_pack is not None:
            badges.append("previously_imported")
            badges.append("compare_available")
            trust_anchor = related_local_pack.get("trust_anchor") if isinstance(related_local_pack.get("trust_anchor"), dict) else {}
            if str(trust_anchor.get("local_review_status") or "unreviewed").strip() not in {"", "unreviewed"}:
                badges.append("reviewed_locally")
            if any(str(item).strip() for item in (trust_anchor.get("user_approved_hashes") if isinstance(trust_anchor.get("user_approved_hashes"), list) else [])):
                badges.append("approval_tied_to_content")
            if self._listing_indicates_changed_upstream(listing, related_local_pack):
                badges.append("changed_upstream")
        return RegistryBadgeSet(tuple(sorted(dict.fromkeys(badges))))

    @staticmethod
    def _listing_indicates_changed_upstream(listing: dict[str, Any], related_local_pack: dict[str, Any]) -> bool:
        latest_ref_hint = str(listing.get("latest_ref_hint") or "").strip()
        if not latest_ref_hint:
            return False
        source = related_local_pack.get("source") if isinstance(related_local_pack.get("source"), dict) else {}
        local_commit = str(source.get("commit_hash") or "").strip()
        local_ref = str(source.get("ref") or "").strip()
        if local_commit and latest_ref_hint and latest_ref_hint != local_commit:
            return True
        if local_ref and latest_ref_hint and latest_ref_hint != local_ref:
            return True
        return False

    def _build_preview(self, *, source: RegistrySource, listing: dict[str, Any]) -> RegistryPackPreview:
        related_local_pack = self._find_related_local_pack(listing)
        badges = self._badge_set(listing, related_local_pack=related_local_pack).badges
        artifact_type = str(listing.get("artifact_type_hint") or "unknown")
        if artifact_type == "portable_text_skill":
            policy_hint = "This looks like a text-based skill pack and is likely compatible with safe import."
        elif artifact_type == "native_code_pack":
            policy_hint = "This appears to be a code or plugin package and would be blocked by current policy."
        elif artifact_type == "experience_pack":
            policy_hint = "This looks like an experience pack and is not currently importable as a safe local pack."
        else:
            policy_hint = "This entry is incomplete or ambiguous, so I would treat it as unknown until fetched and classified."
        appears_to_do = str(listing.get("summary") or f"{listing.get('name')} from a discovery registry.").strip()
        source_hints = [
            "Registry metadata is untrusted.",
            "Nothing has been fetched yet.",
        ]
        if not _is_commit_like(str(listing.get("latest_ref_hint") or "").strip() or None):
            source_hints.append("This listing is not pinned to a stable commit.")
        compare_hint: dict[str, Any] | None = None
        if related_local_pack is not None:
            compare_hint = {
                "available": True,
                "related_canonical_id": str(related_local_pack.get("canonical_id") or related_local_pack.get("pack_id") or ""),
                "likely_changed_upstream": self._listing_indicates_changed_upstream(listing, related_local_pack),
                "message": (
                    "A local version of this pack already exists. To inspect whether it changed, "
                    "fetch a new snapshot and compare it safely."
                ),
            }
        install_handoff: dict[str, Any] | None = None
        if listing.get("source_url"):
            source_kind_hint = str(listing.get("source_kind_hint") or "").strip().lower() or "generic_archive_url"
            install_handoff = {
                "source": str(listing.get("source_url") or ""),
                "source_kind": source_kind_hint,
            }
            latest_ref_hint = str(listing.get("latest_ref_hint") or "").strip()
            if latest_ref_hint:
                install_handoff["ref"] = latest_ref_hint
        summary = (
            "Read-only preview: metadata only for now. If you install it, I will fetch it into quarantine, "
            "scan it, and normalize the snapshot before anything becomes usable. "
        )
        summary += policy_hint + " "
        if related_local_pack is not None:
            summary += "A local version exists, but identity is tied to content, not this listing. "
        summary += "The fetched snapshot is what would be scanned and normalized."
        preview = RegistryPackPreview(
            source=source.to_dict(),
            listing=listing,
            fetched=False,
            summary=summary.strip(),
            appears_to_do=appears_to_do,
            artifact_type_hint=artifact_type,
            policy_hint=policy_hint,
            badges=badges,
            source_hints=tuple(source_hints),
            related_local_pack=(
                {
                    "canonical_id": str(related_local_pack.get("canonical_id") or related_local_pack.get("pack_id") or ""),
                    "name": str(related_local_pack.get("name") or ""),
                    "content_hash": str(related_local_pack.get("content_hash") or "").strip() or None,
                    "local_review_status": str(
                        ((related_local_pack.get("trust_anchor") or {}).get("local_review_status") or "unreviewed")
                    ),
                }
                if related_local_pack is not None
                else None
            ),
            compare_hint=compare_hint,
            install_handoff=install_handoff,
            choices=(
                "preview details",
                "fetch for inspection",
                "compare with local version if available",
                "ignore",
            ),
        )
        return preview

    @staticmethod
    def _ts_to_iso(value: int) -> str | None:
        if value <= 0:
            return None
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

from agent.packs.registry_discovery import REGISTRY_KIND_GENERIC_API, REGISTRY_KIND_GITHUB_INDEX
from agent.packs.remote_fetch import (
    REMOTE_KIND_GENERIC_ARCHIVE_URL,
    REMOTE_KIND_GITHUB_ARCHIVE,
    REMOTE_KIND_GITHUB_REPO,
)
from agent.packs.source_leads import SourceLead, infer_suspected_source_kind, sanitize_lead_url

SOURCE_KIND_GENERIC_WEB_RESULT = "generic_web_result"
APPROVABLE_SOURCE_KINDS = {
    REMOTE_KIND_GITHUB_REPO,
    REMOTE_KIND_GITHUB_ARCHIVE,
    REMOTE_KIND_GENERIC_ARCHIVE_URL,
}


@dataclass(frozen=True)
class SourceApprovalPreview:
    ok: bool
    source_id: str | None
    source_kind: str
    registry_kind: str | None
    title: str
    url: str
    base_url: str | None
    untrusted: bool = True
    explicit_user_trust_required: bool = True
    content_remains_hostile: bool = True
    fetch_allowed_after_approval: bool = False
    blocked_reason: str | None = None
    next_step: str = "confirm_source_approval"
    user_message: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceApprovalResult:
    ok: bool
    approved: bool
    source_id: str | None
    source_kind: str | None
    registry_kind: str | None
    policy: dict[str, Any] | None
    catalog_source: dict[str, Any] | None
    did_fetch: bool = False
    did_import: bool = False
    did_install: bool = False
    blocked_reason: str | None = None
    user_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SourceApprovalController:
    """Records explicit source trust without fetching or importing content."""

    def __init__(self, *, pack_registry_discovery: Any) -> None:
        self.pack_registry_discovery = pack_registry_discovery

    def preview(self, lead: SourceLead | dict[str, Any]) -> SourceApprovalPreview:
        row = lead.to_dict() if isinstance(lead, SourceLead) else dict(lead or {})
        display_url = sanitize_lead_url(str(row.get("url") or ""))
        title = _short_text(row.get("title"), default="Untrusted source lead", limit=160)
        source_kind = str(row.get("suspected_source_kind") or "").strip().lower()
        if not source_kind:
            source_kind = infer_suspected_source_kind(display_url or "")
        provenance = _provenance(row)
        if not display_url:
            return SourceApprovalPreview(
                ok=False,
                source_id=None,
                source_kind=source_kind or SOURCE_KIND_GENERIC_WEB_RESULT,
                registry_kind=None,
                title=title,
                url="",
                base_url=None,
                blocked_reason="invalid_source_lead_url",
                next_step="cancel",
                user_message="I cannot approve that source lead because it does not have a valid http/https URL. No content was fetched.",
                provenance=provenance,
            )
        if source_kind == SOURCE_KIND_GENERIC_WEB_RESULT:
            return SourceApprovalPreview(
                ok=False,
                source_id=None,
                source_kind=source_kind,
                registry_kind=None,
                title=title,
                url=display_url,
                base_url=None,
                blocked_reason="generic_web_result_not_directly_fetchable",
                next_step="manual_source_configuration_required",
                user_message=(
                    "That search result is an untrusted generic web result, not a directly fetchable pack source. "
                    "A manual source configuration or catalog entry is required before approval. No content was fetched."
                ),
                provenance=provenance,
            )
        if source_kind not in APPROVABLE_SOURCE_KINDS:
            return SourceApprovalPreview(
                ok=False,
                source_id=None,
                source_kind=source_kind,
                registry_kind=None,
                title=title,
                url=display_url,
                base_url=None,
                blocked_reason="unsupported_source_lead_kind",
                next_step="cancel",
                user_message=f"That source lead kind is not supported for source approval: {source_kind}. No content was fetched.",
                provenance=provenance,
            )
        base_url = _base_url_for_kind(source_kind, display_url)
        source_id = _source_id_for_lead(kind=source_kind, url=base_url)
        registry_kind = _registry_kind_for_source_kind(source_kind)
        message = (
            f"Source approval preview for {title}. The source is still untrusted. "
            "Approval only records explicit trust for this source id and allows a future fetch/preview into quarantine. "
            "GitHub or any other domain does not make the content safe. No pages were fetched, no archives were downloaded, and no pack was imported. "
            "If you confirm, the next safe step is preview/fetch into quarantine."
        )
        return SourceApprovalPreview(
            ok=True,
            source_id=source_id,
            source_kind=source_kind,
            registry_kind=registry_kind,
            title=title,
            url=display_url,
            base_url=base_url,
            fetch_allowed_after_approval=True,
            user_message=message,
            provenance=provenance,
        )

    def approve(self, preview: SourceApprovalPreview | dict[str, Any], *, changed_by: str | None = None) -> SourceApprovalResult:
        row = preview.to_dict() if isinstance(preview, SourceApprovalPreview) else dict(preview or {})
        if not bool(row.get("ok")):
            return SourceApprovalResult(
                ok=False,
                approved=False,
                source_id=None,
                source_kind=str(row.get("source_kind") or "") or None,
                registry_kind=str(row.get("registry_kind") or "") or None,
                policy=None,
                catalog_source=None,
                blocked_reason=str(row.get("blocked_reason") or "source_approval_preview_blocked"),
                user_message="I did not approve that source because the approval preview was blocked. No content was fetched or imported.",
            )
        source_id = str(row.get("source_id") or "").strip()
        source_kind = str(row.get("source_kind") or "").strip().lower()
        registry_kind = str(row.get("registry_kind") or "").strip().lower()
        base_url = str(row.get("base_url") or row.get("url") or "").strip()
        if not source_id or source_kind not in APPROVABLE_SOURCE_KINDS or not registry_kind or not base_url:
            return SourceApprovalResult(
                ok=False,
                approved=False,
                source_id=source_id or None,
                source_kind=source_kind or None,
                registry_kind=registry_kind or None,
                policy=None,
                catalog_source=None,
                blocked_reason="invalid_source_approval_preview",
                user_message="I did not approve that source because the approval preview was incomplete. No content was fetched or imported.",
            )
        notes = _approval_notes(source_kind=source_kind, title=str(row.get("title") or ""), provenance=row.get("provenance"))
        source_payload = {
            "source_id": source_id,
            "name": _short_text(row.get("title"), default=source_id, limit=80),
            "kind": registry_kind,
            "base_url": base_url,
            "enabled": True,
            "discovery_only": True,
            "supports_search": True,
            "supports_preview": True,
            "supports_compare_hint": False,
            "notes": notes,
        }
        try:
            try:
                catalog_source = self.pack_registry_discovery.create_catalog_source(
                    source_payload,
                    changed_by=changed_by or "assistant_source_approval",
                )
            except ValueError as exc:
                if str(exc) != "duplicate_source_id":
                    raise
                catalog_source = self.pack_registry_discovery.update_catalog_source(
                    source_id,
                    {key: value for key, value in source_payload.items() if key != "source_id"},
                    changed_by=changed_by or "assistant_source_approval",
                )
            policy = self.pack_registry_discovery.update_source_policy(
                source_id,
                {
                    "enabled": True,
                    "allowlisted": True,
                    "denied": False,
                    "notes": notes,
                    "approved_by_user": True,
                    "approved_at": _now_iso(),
                    "approval_provenance": _compact_provenance(row.get("provenance")),
                },
                changed_by=changed_by or "assistant_source_approval",
            )
        except Exception as exc:
            return SourceApprovalResult(
                ok=False,
                approved=False,
                source_id=source_id,
                source_kind=source_kind,
                registry_kind=registry_kind,
                policy=None,
                catalog_source=None,
                blocked_reason=str(exc),
                user_message="I could not record source approval. No content was fetched, imported, installed, enabled, or granted permissions.",
            )
        message = (
            f"I recorded source approval for {source_id}. No pack was fetched, imported, installed, approved, enabled, or granted permissions. "
            "The source content remains hostile and must still go through preview/fetch into quarantine, normalization, review, approval, enablement, and any required permissions before use. "
            "Next safe step: preview/fetch into quarantine."
        )
        return SourceApprovalResult(
            ok=True,
            approved=True,
            source_id=source_id,
            source_kind=source_kind,
            registry_kind=registry_kind,
            policy=policy,
            catalog_source=catalog_source,
            user_message=message,
        )


def _registry_kind_for_source_kind(kind: str) -> str:
    if kind == REMOTE_KIND_GITHUB_REPO:
        return REGISTRY_KIND_GITHUB_INDEX
    return REGISTRY_KIND_GENERIC_API


def _base_url_for_kind(kind: str, url: str) -> str:
    parsed = urlparse(url)
    if kind == REMOTE_KIND_GITHUB_REPO:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return urlunparse((parsed.scheme, parsed.netloc, f"/{parts[0]}/{parts[1]}", "", "", ""))
    return url


def _source_id_for_lead(*, kind: str, url: str) -> str:
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part][:3]
    stem = "-".join(["approved", kind.replace("_", "-"), parsed.netloc.replace(".", "-"), *path_parts])
    slug = re.sub(r"[^a-z0-9_-]+", "-", stem.lower()).strip("-")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    max_stem = max(8, 63 - len(digest) - 1)
    return f"{slug[:max_stem].strip('-')}-{digest}"[:64]


def _provenance(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "safe_web_search",
        "title": _short_text(row.get("title"), default="", limit=120),
        "snippet_excerpt": _short_text(row.get("snippet"), default="", limit=160),
        "source_engine": _short_text(row.get("source_engine"), default="", limit=80) or None,
        "lead_reason": _short_text(row.get("reason"), default="", limit=120) or None,
    }


def _compact_provenance(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    return {
        "source": "safe_web_search",
        "title": _short_text(row.get("title"), default="", limit=80),
        "source_engine": _short_text(row.get("source_engine"), default="", limit=50) or None,
    }


def _approval_notes(*, source_kind: str, title: str, provenance: Any) -> str:
    prov = _compact_provenance(provenance)
    parts = [
        "approved_by_user=true",
        f"remote_source_kind={source_kind}",
        "content_remains_hostile=true",
        "approval permits future fetch/preview into quarantine only",
        "no content was fetched during source approval",
    ]
    if title:
        parts.append(f"lead_title={_short_text(title, default='', limit=80)}")
    if prov.get("source_engine"):
        parts.append(f"source_engine={prov['source_engine']}")
    return "; ".join(parts)


def _short_text(value: Any, *, default: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        text = default
    return text[: max(1, limit)]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


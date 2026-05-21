from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agent.packs.external_ingestion import (
    STATUS_BLOCKED,
    STATUS_NORMALIZED,
    STATUS_PARTIAL_SAFE_IMPORT,
    ExternalPackIngestor,
)
from agent.packs.lifecycle import PackLifecycleService
from agent.packs.registry_discovery import REGISTRY_KIND_GITHUB_INDEX
from agent.packs.remote_fetch import (
    REMOTE_KIND_GENERIC_ARCHIVE_URL,
    REMOTE_KIND_GITHUB_ARCHIVE,
    REMOTE_KIND_GITHUB_REPO,
    RemotePackFetcher,
)
from agent.packs.source_leads import infer_suspected_source_kind, sanitize_lead_url

FETCHABLE_SOURCE_KINDS = {
    REMOTE_KIND_GITHUB_REPO,
    REMOTE_KIND_GITHUB_ARCHIVE,
    REMOTE_KIND_GENERIC_ARCHIVE_URL,
}


@dataclass(frozen=True)
class SourceFetchPreview:
    ok: bool
    source_id: str
    source_kind: str | None
    url: str | None
    source_name: str | None = None
    blocked_reason: str | None = None
    content_remains_hostile: bool = True
    fetched_to_quarantine: bool = False
    imported_for_review: bool = False
    did_approve: bool = False
    did_enable: bool = False
    did_grant_permissions: bool = False
    did_use_pack: bool = False
    user_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceFetchResult:
    ok: bool
    source_id: str
    source_kind: str | None
    fetched_to_quarantine: bool
    imported_for_review: bool
    pack_id: str | None = None
    canonical_id: str | None = None
    lifecycle_state: str | None = None
    next_step: str | None = None
    blocked_reason: str | None = None
    did_approve: bool = False
    did_enable: bool = False
    did_grant_permissions: bool = False
    did_use_pack: bool = False
    pack: dict[str, Any] | None = None
    normalization_result: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    user_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SourceFetchController:
    """Fetches an approved source into quarantine and imports for review only."""

    def __init__(
        self,
        *,
        pack_store: Any,
        pack_registry_discovery: Any,
        remote_fetcher: RemotePackFetcher | None = None,
        lifecycle_service: PackLifecycleService | None = None,
    ) -> None:
        self.pack_store = pack_store
        self.pack_registry_discovery = pack_registry_discovery
        self.remote_fetcher = remote_fetcher
        self.lifecycle_service = lifecycle_service or PackLifecycleService()

    def preview(self, source_id: str) -> SourceFetchPreview:
        source_id = str(source_id or "").strip()
        source_payload, blocked = self._approved_source_payload(source_id)
        if blocked is not None:
            return SourceFetchPreview(
                ok=False,
                source_id=source_id,
                source_kind=None,
                url=None,
                blocked_reason=blocked,
                user_message=f"I cannot fetch that source yet: {blocked}. No content was fetched or imported.",
            )
        source = source_payload["source"]
        source_kind = self._fetch_kind(source)
        url = sanitize_lead_url(str(source.get("base_url") or ""))
        if source_kind not in FETCHABLE_SOURCE_KINDS or not url:
            return SourceFetchPreview(
                ok=False,
                source_id=source_id,
                source_kind=source_kind,
                url=url,
                source_name=str(source.get("name") or source_id),
                blocked_reason="source_not_fetchable",
                user_message=(
                    "That approved source is not a directly fetchable pack source. "
                    "No content was fetched or imported."
                ),
            )
        message = (
            f"Fetch preview for {source.get('name') or source_id}. This source is approved only as a source; its content remains hostile. "
            "If you confirm, I will fetch only into quarantine and run the existing hostile archive/manifest/SKILL.md validation and ingestion path. "
            "No pack will be approved, enabled, configured, granted permissions, used, or invoked. Say yes to fetch into quarantine only."
        )
        return SourceFetchPreview(
            ok=True,
            source_id=source_id,
            source_kind=source_kind,
            url=url,
            source_name=str(source.get("name") or source_id),
            user_message=message,
        )

    def fetch_import_for_review(self, preview: SourceFetchPreview | dict[str, Any]) -> SourceFetchResult:
        row = preview.to_dict() if isinstance(preview, SourceFetchPreview) else dict(preview or {})
        source_id = str(row.get("source_id") or "").strip()
        if not bool(row.get("ok")):
            reason = str(row.get("blocked_reason") or "source_fetch_preview_blocked")
            return SourceFetchResult(
                ok=False,
                source_id=source_id,
                source_kind=str(row.get("source_kind") or "") or None,
                fetched_to_quarantine=False,
                imported_for_review=False,
                blocked_reason=reason,
                user_message=f"I did not fetch that source because the fetch preview was blocked: {reason}.",
            )
        source_payload, blocked = self._approved_source_payload(source_id)
        if blocked is not None:
            return SourceFetchResult(
                ok=False,
                source_id=source_id,
                source_kind=str(row.get("source_kind") or "") or None,
                fetched_to_quarantine=False,
                imported_for_review=False,
                blocked_reason=blocked,
                user_message=f"I did not fetch that source because its approval gate is not complete: {blocked}.",
            )
        source_kind = str(row.get("source_kind") or "").strip().lower()
        url = str(row.get("url") or "").strip()
        if source_kind not in FETCHABLE_SOURCE_KINDS or not url:
            return SourceFetchResult(
                ok=False,
                source_id=source_id,
                source_kind=source_kind or None,
                fetched_to_quarantine=False,
                imported_for_review=False,
                blocked_reason="source_not_fetchable",
                user_message="I did not fetch that source because it is not a directly fetchable pack source.",
            )
        ingestor = ExternalPackIngestor(
            self.pack_store.external_storage_root(),
            remote_fetcher=self.remote_fetcher,
        )
        remote_source = RemotePackFetcher.build_source(kind=source_kind, url=url)
        normalization_result, review_envelope = ingestor.ingest_from_remote_source(
            remote_source,
            created_by="source_fetch_preview",
        )
        pack_row = self.pack_store.record_external_pack(
            canonical_pack=normalization_result.pack.to_dict(),
            classification=normalization_result.classification,
            status=normalization_result.status,
            risk_report=normalization_result.risk_report.to_dict(),
            review_envelope=review_envelope.to_dict(),
            quarantine_path=normalization_result.quarantine_path,
            normalized_path=normalization_result.normalized_path,
        )
        pack_id = str(pack_row.get("pack_id") or pack_row.get("canonical_id") or "").strip() or None
        imported_for_review = normalization_result.status in {STATUS_NORMALIZED, STATUS_PARTIAL_SAFE_IMPORT}
        fetched_to_quarantine = bool(normalization_result.quarantine_path) and normalization_result.status != STATUS_BLOCKED
        lifecycle = self.lifecycle_service.evaluate(imported_pack=pack_row, permission_grants=[]).to_dict()
        next_step = ((lifecycle.get("next_step") if isinstance(lifecycle.get("next_step"), dict) else {}) or {}).get("action")
        if imported_for_review:
            message = (
                f"I fetched the approved source into quarantine and imported {normalization_result.pack.name} for review only. "
                "No pack was approved, enabled, configured, granted permissions, used, or invoked. "
                "Next safe step: review/approval."
            )
        else:
            message = (
                "The quarantine fetch/import was blocked by the hostile intake gates. "
                "No pack was approved, enabled, configured, granted permissions, used, or invoked."
            )
        return SourceFetchResult(
            ok=imported_for_review,
            source_id=source_id,
            source_kind=source_kind,
            fetched_to_quarantine=fetched_to_quarantine,
            imported_for_review=imported_for_review,
            pack_id=pack_id,
            canonical_id=pack_id,
            lifecycle_state=str(lifecycle.get("state") or "") or None,
            next_step=str(next_step or "") or None,
            blocked_reason=None if imported_for_review else normalization_result.status,
            pack=pack_row,
            normalization_result={
                **normalization_result.to_dict(),
                "pack": normalization_result.pack.to_dict(),
            },
            review=review_envelope.to_dict(),
            user_message=message,
        )

    def _approved_source_payload(self, source_id: str) -> tuple[dict[str, Any], str | None]:
        if not source_id:
            return {}, "missing_source_id"
        try:
            payload = self.pack_registry_discovery.get_source_policy(source_id)
        except KeyError:
            return {}, "source_not_found"
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        effective = payload.get("effective_policy") if isinstance(payload.get("effective_policy"), dict) else {}
        override = payload.get("persisted_override") if isinstance(payload.get("persisted_override"), dict) else {}
        if not source:
            return {}, "source_not_found"
        if not bool(effective.get("enabled", False)):
            return {}, "source_disabled"
        if not bool(effective.get("allowlisted", False)):
            return {}, "source_not_allowlisted"
        if bool(effective.get("denied", False)):
            return {}, "source_denied"
        if not bool(effective.get("allowed_by_policy", False)):
            return {}, str(effective.get("blocked_reason") or "source_policy_blocked")
        if not bool(override.get("approved_by_user", False)):
            return {}, "source_not_approved_by_user"
        return {"source": source, "effective_policy": effective, "persisted_override": override}, None

    @staticmethod
    def _fetch_kind(source: dict[str, Any]) -> str:
        kind = str(source.get("kind") or "").strip().lower()
        base_url = str(source.get("base_url") or "").strip()
        if kind == REGISTRY_KIND_GITHUB_INDEX:
            return REMOTE_KIND_GITHUB_REPO
        inferred = infer_suspected_source_kind(base_url)
        if inferred in FETCHABLE_SOURCE_KINDS:
            return inferred
        return inferred or "generic_web_result"


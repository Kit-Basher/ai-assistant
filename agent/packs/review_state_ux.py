from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.packs.lifecycle import PackLifecycleService


@dataclass(frozen=True)
class PackReviewStateSummary:
    pack_name: str
    pack_id: str
    canonical_id: str
    lifecycle_state: str
    local_review_status: str
    enabled: bool
    permissions_granted: tuple[str, ...]
    managed_adapters: tuple[str, ...]
    risk_level: str
    risk_score: float
    risk_flags: tuple[str, ...]
    classification: str
    import_status: str
    source_summary: str | None
    manual_review_required: bool
    next_safe_step: str
    usable: bool
    lines: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["permissions_granted"] = list(self.permissions_granted)
        payload["managed_adapters"] = list(self.managed_adapters)
        payload["risk_flags"] = list(self.risk_flags)
        payload["lines"] = list(self.lines)
        return payload


def build_pack_review_state_summary(
    pack_row: dict[str, Any],
    *,
    permission_grants: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    lifecycle: dict[str, Any] | None = None,
) -> PackReviewStateSummary:
    canonical = pack_row.get("canonical_pack") if isinstance(pack_row.get("canonical_pack"), dict) else {}
    trust_anchor = canonical.get("trust_anchor") if isinstance(canonical.get("trust_anchor"), dict) else {}
    runtime = canonical.get("runtime") if isinstance(canonical.get("runtime"), dict) else {}
    permissions = canonical.get("permissions") if isinstance(canonical.get("permissions"), dict) else {}
    risk = pack_row.get("risk_report") if isinstance(pack_row.get("risk_report"), dict) else {}
    source = canonical.get("source") if isinstance(canonical.get("source"), dict) else {}
    lifecycle_payload = lifecycle or PackLifecycleService().evaluate(
        imported_pack=pack_row,
        permission_grants=list(permission_grants),
    ).to_dict()
    adapter_rows = _managed_adapter_rows(canonical)
    managed_adapters = tuple(_safe_token(row.get("kind"), limit=80) for row in adapter_rows if _safe_token(row.get("kind"), limit=80))
    grant_labels = tuple(
        _safe_token(row.get("adapter_kind") or row.get("kind"), limit=80)
        for row in permission_grants
        if isinstance(row, dict) and str(row.get("state") or "").strip().lower() == "granted"
    )
    next_step = lifecycle_payload.get("next_step") if isinstance(lifecycle_payload.get("next_step"), dict) else {}
    lifecycle_state = _safe_token(lifecycle_payload.get("state"), default="unknown", limit=80)
    next_step_label = _safe_phrase(next_step.get("label") or "review/approval", limit=160)
    if lifecycle_state in {"generated_quarantined", "imported_for_review"}:
        next_step_label = "review/approval"
    enabled = bool(runtime.get("enabled", False))
    local_review_status = _safe_token(trust_anchor.get("local_review_status"), default="unreviewed", limit=80)
    pack_id = _safe_token(pack_row.get("pack_id") or pack_row.get("canonical_id") or canonical.get("id"), default="unknown", limit=96)
    canonical_id = _safe_token(pack_row.get("canonical_id") or canonical.get("id") or pack_id, default=pack_id, limit=96)
    pack_name = _safe_phrase(
        pack_row.get("name") or canonical.get("display_name") or canonical.get("name") or "Imported pack",
        limit=120,
    )
    risk_flags = tuple(_safe_token(flag, limit=80) for flag in _as_list(risk.get("flags"))[:8])
    source_summary = _source_summary(source)
    summary = PackReviewStateSummary(
        pack_name=pack_name,
        pack_id=pack_id,
        canonical_id=canonical_id,
        lifecycle_state=lifecycle_state,
        local_review_status=local_review_status,
        enabled=enabled,
        permissions_granted=grant_labels,
        managed_adapters=managed_adapters,
        risk_level=_safe_token(pack_row.get("risk_level") or risk.get("level"), default="unknown", limit=80),
        risk_score=float(pack_row.get("risk_score") or risk.get("score") or 0.0),
        risk_flags=risk_flags,
        classification=_safe_token(pack_row.get("classification"), default="unknown_pack", limit=80),
        import_status=_safe_token(pack_row.get("status"), default="unknown", limit=80),
        source_summary=source_summary,
        manual_review_required=bool(pack_row.get("review_required", True)),
        next_safe_step=next_step_label,
        usable=bool(lifecycle_payload.get("usable", False)),
    )
    lines = _render_lines(summary)
    return PackReviewStateSummary(**{**summary.to_dict(), "lines": tuple(lines)})


def render_pack_review_state(
    pack_row: dict[str, Any],
    *,
    permission_grants: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    lifecycle: dict[str, Any] | None = None,
) -> str:
    summary = build_pack_review_state_summary(
        pack_row,
        permission_grants=permission_grants,
        lifecycle=lifecycle,
    )
    return "\n".join(summary.lines)


def _render_lines(summary: PackReviewStateSummary) -> list[str]:
    permission_text = "No permissions granted" if not summary.permissions_granted else "Permissions granted: " + ", ".join(summary.permissions_granted)
    adapter_text = "Managed adapters requested: none" if not summary.managed_adapters else "Managed adapters requested: " + ", ".join(summary.managed_adapters)
    flags = ", ".join(summary.risk_flags[:5]) if summary.risk_flags else "none"
    review_text = "Not approved" if summary.local_review_status != "approved" else "Approved for review only"
    enabled_text = "Enabled" if summary.enabled else "Not enabled"
    lines = [
        f"Imported for review only: {summary.pack_name}",
        f"Pack id: {summary.pack_id}",
        f"Canonical id: {summary.canonical_id}",
        f"Lifecycle state: {summary.lifecycle_state}; usable: {'yes' if summary.usable else 'no'}",
        f"Review status: {summary.local_review_status}; {review_text}",
        f"Enabled: {'true' if summary.enabled else 'false'}; {enabled_text}",
        permission_text,
        adapter_text,
        f"Import status: {summary.import_status}; classification: {summary.classification}",
        f"Risk: {summary.risk_level} ({summary.risk_score:.2f}); flags: {flags}",
    ]
    if summary.source_summary:
        lines.append(f"Source: {summary.source_summary}")
    lines.extend(
        [
            f"Manual review required: {'yes' if summary.manual_review_required else 'no'}",
            "Not usable yet.",
            f"Next safe step: {summary.next_safe_step or 'review/approval'}",
        ]
    )
    return lines


def _managed_adapter_rows(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in (
        canonical.get("managed_adapters"),
        (canonical.get("runtime") if isinstance(canonical.get("runtime"), dict) else {}).get("managed_adapters"),
        (canonical.get("permissions") if isinstance(canonical.get("permissions"), dict) else {}).get("managed_adapters"),
    ):
        if isinstance(value, list):
            rows.extend(dict(row) for row in value if isinstance(row, dict))
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        kind = _safe_token(row.get("kind"), limit=80)
        if kind:
            deduped[kind] = row
    return list(deduped.values())


def _source_summary(source: dict[str, Any]) -> str | None:
    origin = _safe_token(source.get("origin"), limit=80)
    url = _safe_url(source.get("url"))
    if origin and url:
        return f"{origin} {url}"
    return origin or url or None


def _safe_url(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"([?&](?:token|key|api_key|access_token|auth|signature|sig)=)[^&\\s]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(r"://[^/@\\s]+@", "://[REDACTED]@", text)
    return _safe_phrase(text, limit=180)


def _safe_phrase(value: Any, *, default: str = "", limit: int) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"(?i)ignore (?:all )?(?:previous|system|developer)[^.\\n]*", "[REDACTED instruction]", text)
    text = re.sub(r"(?i)(?:secret|token|api[_-]?key|password)\\s*[:=]\\s*\\S+", "[REDACTED secret]", text)
    text = text[: max(1, limit)].strip()
    return text or default


def _safe_token(value: Any, *, default: str = "", limit: int) -> str:
    text = _safe_phrase(value, default=default, limit=limit)
    return re.sub(r"[^A-Za-z0-9_.:/@%?=&\\[\\]-]+", " ", text).strip()[:limit] or default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.packs.managed_adapters import GRANT_GRANTED


class PackLifecycleState:
    MISSING = "missing"
    DISCOVERED = "discovered"
    PREVIEWED = "previewed"
    SCAFFOLD_PREVIEWED = "scaffold_previewed"
    GENERATED_QUARANTINED = "generated_quarantined"
    IMPORTED_FOR_REVIEW = "imported_for_review"
    APPROVED = "approved"
    ENABLED = "enabled"
    NEEDS_CONFIGURATION = "needs_configuration"
    NEEDS_PERMISSION = "needs_permission"
    USABLE = "usable"
    BLOCKED = "blocked"
    DISABLED = "disabled"
    REMOVED = "removed"


class PackLifecycleGate:
    DISCOVERY = "discovery"
    PREVIEW = "preview"
    SCAFFOLD_PREVIEW = "scaffold_preview"
    QUARANTINE = "quarantine"
    INSPECTION = "inspection"
    APPROVAL = "approval"
    CONFIGURATION = "configuration"
    PERMISSION = "permission"
    ENABLEMENT = "enablement"
    USE = "use"
    SAFETY_REVIEW = "safety_review"


@dataclass(frozen=True)
class PackLifecycleNextStep:
    action: str
    label: str
    gate: str | None = None
    required_confirmation: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "label": self.label,
            "gate": self.gate,
            "required_confirmation": self.required_confirmation,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PackLifecycleResult:
    capability: str | None
    state: str
    usable: bool
    blocked: bool = False
    missing_gate: str | None = None
    next_step: PackLifecycleNextStep | None = None
    user_message_summary: str = ""
    required_confirmation: bool = False
    required_permissions: tuple[str, ...] = ()
    required_configuration: tuple[str, ...] = ()
    source: str = "unknown"
    pack_id: str | None = None
    canonical_id: str | None = None
    pack_name: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "pack_id": self.pack_id,
            "canonical_id": self.canonical_id,
            "pack_name": self.pack_name,
            "state": self.state,
            "usable": self.usable,
            "blocked": self.blocked,
            "missing_gate": self.missing_gate,
            "next_step": self.next_step.to_dict() if self.next_step is not None else None,
            "user_message_summary": self.user_message_summary,
            "required_confirmation": self.required_confirmation,
            "required_permissions": list(self.required_permissions),
            "required_configuration": list(self.required_configuration),
            "source": self.source,
            "evidence": dict(self.evidence),
        }


class PackLifecycleService:
    """Authoritative lifecycle evaluation for external/generated pack usability.

    This service does not install, enable, execute, or grant anything. It only
    maps observed pack/catalog/scaffold facts into a lifecycle state and the
    next safe assistant step.
    """

    def evaluate(
        self,
        *,
        capability: str | None = None,
        installed_pack: dict[str, Any] | None = None,
        catalog_pack: dict[str, Any] | None = None,
        catalog_preview: dict[str, Any] | None = None,
        scaffold_preview: dict[str, Any] | None = None,
        scaffold_preview_shown: bool = False,
        generated_candidate: dict[str, Any] | None = None,
        imported_pack: dict[str, Any] | None = None,
        removed_pack: dict[str, Any] | None = None,
        permission_grants: list[dict[str, Any]] | None = None,
        required_configuration: list[str] | tuple[str, ...] | None = None,
    ) -> PackLifecycleResult:
        cap = _clean(capability)
        if isinstance(removed_pack, dict) and removed_pack:
            return self._removed(cap, removed_pack)
        if isinstance(generated_candidate, dict) and generated_candidate:
            return self._generated_quarantined(cap, generated_candidate)
        pack = installed_pack if isinstance(installed_pack, dict) and installed_pack else imported_pack
        if isinstance(pack, dict) and pack:
            return self._installed_or_imported(
                cap,
                pack,
                permission_grants=permission_grants or [],
                required_configuration=tuple(_clean_list(required_configuration)),
            )
        if isinstance(catalog_preview, dict) and catalog_preview:
            return self._catalog_previewed(cap, catalog_preview)
        if isinstance(catalog_pack, dict) and catalog_pack:
            return self._catalog_discovered(cap, catalog_pack)
        if isinstance(scaffold_preview, dict) and scaffold_preview:
            if scaffold_preview_shown:
                return self._scaffold_previewed(cap, scaffold_preview)
            return self._missing(
                cap,
                source="scaffold",
                next_step=PackLifecycleNextStep(
                    action="scaffold_preview",
                    label="Show the scaffold preview before creating anything.",
                    gate=PackLifecycleGate.SCAFFOLD_PREVIEW,
                    required_confirmation=True,
                ),
                summary="No usable pack is installed yet. A scaffold preview is available as the next safe step.",
            )
        return self._missing(
            cap,
            source="unknown",
            next_step=PackLifecycleNextStep(
                action="discover_or_scaffold",
                label="Search approved pack sources or define a scaffold preview.",
                gate=PackLifecycleGate.DISCOVERY,
                required_confirmation=True,
            ),
            summary="No installed, approved, enabled pack is usable for this capability yet.",
        )

    def _removed(self, capability: str | None, row: dict[str, Any]) -> PackLifecycleResult:
        name = _pack_name(row)
        return PackLifecycleResult(
            capability=capability,
            pack_id=_pack_id(row),
            canonical_id=_canonical_id(row),
            pack_name=name,
            state=PackLifecycleState.REMOVED,
            usable=False,
            blocked=True,
            missing_gate=PackLifecycleGate.DISCOVERY,
            source="installed",
            user_message_summary=f"{name} was removed and is not usable.",
            next_step=PackLifecycleNextStep(
                action="discover_or_reinstall",
                label="Search approved sources again or reinstall a reviewed pack.",
                gate=PackLifecycleGate.DISCOVERY,
                required_confirmation=True,
            ),
        )

    def _generated_quarantined(self, capability: str | None, row: dict[str, Any]) -> PackLifecycleResult:
        name = _pack_name(row)
        return PackLifecycleResult(
            capability=capability,
            pack_id=_pack_id(row),
            canonical_id=_canonical_id(row),
            pack_name=name,
            state=PackLifecycleState.GENERATED_QUARANTINED,
            usable=False,
            missing_gate=PackLifecycleGate.INSPECTION,
            source="generated",
            user_message_summary=f"{name} is a generated candidate in quarantine and cannot be used yet.",
            required_confirmation=True,
            next_step=PackLifecycleNextStep(
                action="inspect",
                label="Inspect the quarantined candidate before approval.",
                gate=PackLifecycleGate.INSPECTION,
                required_confirmation=True,
            ),
        )

    def _catalog_discovered(self, capability: str | None, row: dict[str, Any]) -> PackLifecycleResult:
        name = _catalog_name(row)
        return PackLifecycleResult(
            capability=capability,
            pack_id=_clean(row.get("remote_id") or row.get("id")),
            pack_name=name,
            state=PackLifecycleState.DISCOVERED,
            usable=False,
            missing_gate=PackLifecycleGate.PREVIEW,
            source="catalog",
            user_message_summary=f"{name} was discovered in an approved source, but it is not installed or usable yet.",
            required_confirmation=True,
            next_step=PackLifecycleNextStep(
                action="preview",
                label="Show the pack preview before import.",
                gate=PackLifecycleGate.PREVIEW,
                required_confirmation=True,
            ),
        )

    def _catalog_previewed(self, capability: str | None, row: dict[str, Any]) -> PackLifecycleResult:
        listing = row.get("listing") if isinstance(row.get("listing"), dict) else row
        name = _catalog_name(listing if isinstance(listing, dict) else row)
        return PackLifecycleResult(
            capability=capability,
            pack_id=_clean((listing if isinstance(listing, dict) else row).get("remote_id") or row.get("id")),
            pack_name=name,
            state=PackLifecycleState.PREVIEWED,
            usable=False,
            missing_gate=PackLifecycleGate.QUARANTINE,
            source="catalog",
            user_message_summary=f"{name} has been previewed, but must be imported into quarantine for review before use.",
            required_confirmation=True,
            next_step=PackLifecycleNextStep(
                action="import_for_review",
                label="Import the previewed pack into quarantine for review.",
                gate=PackLifecycleGate.QUARANTINE,
                required_confirmation=True,
            ),
        )

    def _scaffold_previewed(self, capability: str | None, preview: dict[str, Any]) -> PackLifecycleResult:
        name = _clean(preview.get("title")) or "Scaffolded pack"
        return PackLifecycleResult(
            capability=capability or _clean(preview.get("capability")),
            pack_id=_clean(preview.get("scaffold_id")),
            pack_name=name,
            state=PackLifecycleState.SCAFFOLD_PREVIEWED,
            usable=False,
            missing_gate=PackLifecycleGate.QUARANTINE,
            source="scaffold",
            user_message_summary=f"{name} has only been previewed. Creating it would make a review-only candidate.",
            required_confirmation=True,
            next_step=PackLifecycleNextStep(
                action="create_review_candidate",
                label="Create a text-only candidate in quarantine for review.",
                gate=PackLifecycleGate.QUARANTINE,
                required_confirmation=True,
            ),
        )

    def _installed_or_imported(
        self,
        capability: str | None,
        row: dict[str, Any],
        *,
        permission_grants: list[dict[str, Any]],
        required_configuration: tuple[str, ...],
    ) -> PackLifecycleResult:
        name = _pack_name(row)
        pack_id = _pack_id(row)
        canonical_id = _canonical_id(row)
        source = "generated" if _is_generated(row) else "installed"
        status = _clean(row.get("status")).lower()
        if status in {"removed", "tombstoned"} or row.get("removed") is True:
            return self._removed(capability, row)
        if _is_blocked(row):
            return PackLifecycleResult(
                capability=capability,
                pack_id=pack_id,
                canonical_id=canonical_id,
                pack_name=name,
                state=PackLifecycleState.BLOCKED,
                usable=False,
                blocked=True,
                missing_gate=PackLifecycleGate.SAFETY_REVIEW,
                source=source,
                user_message_summary=f"{name} is blocked by safety review and cannot be enabled or used.",
                next_step=PackLifecycleNextStep(
                    action="inspect_blocker",
                    label="Inspect the blocked import details.",
                    gate=PackLifecycleGate.SAFETY_REVIEW,
                    required_confirmation=False,
                ),
            )
        if row.get("disabled") is True or status == "disabled":
            return PackLifecycleResult(
                capability=capability,
                pack_id=pack_id,
                canonical_id=canonical_id,
                pack_name=name,
                state=PackLifecycleState.DISABLED,
                usable=False,
                missing_gate=PackLifecycleGate.ENABLEMENT,
                source=source,
                user_message_summary=f"{name} is disabled and cannot be used.",
                required_confirmation=True,
                next_step=PackLifecycleNextStep(
                    action="enable",
                    label="Enable the pack after review.",
                    gate=PackLifecycleGate.ENABLEMENT,
                    required_confirmation=True,
                ),
            )
        if not _is_approved(row):
            state = PackLifecycleState.IMPORTED_FOR_REVIEW
            missing_gate = PackLifecycleGate.APPROVAL
            if _is_quarantine_only(row):
                state = PackLifecycleState.GENERATED_QUARANTINED
                missing_gate = PackLifecycleGate.INSPECTION
            return PackLifecycleResult(
                capability=capability,
                pack_id=pack_id,
                canonical_id=canonical_id,
                pack_name=name,
                state=state,
                usable=False,
                missing_gate=missing_gate,
                source=source,
                user_message_summary=f"{name} is imported for review only. It is not approved, enabled, or usable yet.",
                required_confirmation=True,
                next_step=PackLifecycleNextStep(
                    action="review_approve",
                    label="Review and approve the pack before enabling it.",
                    gate=PackLifecycleGate.APPROVAL,
                    required_confirmation=True,
                ),
                required_permissions=tuple(_adapter_kinds(row)),
                required_configuration=required_configuration,
            )
        enabled = _enabled_state(row)
        if enabled is False or enabled is None:
            return PackLifecycleResult(
                capability=capability,
                pack_id=pack_id,
                canonical_id=canonical_id,
                pack_name=name,
                state=PackLifecycleState.APPROVED,
                usable=False,
                missing_gate=PackLifecycleGate.ENABLEMENT,
                source=source,
                user_message_summary=f"{name} is approved, but it is not enabled as a live capability.",
                required_confirmation=True,
                next_step=PackLifecycleNextStep(
                    action="enable",
                    label="Enable the approved pack before use.",
                    gate=PackLifecycleGate.ENABLEMENT,
                    required_confirmation=True,
                ),
                required_permissions=tuple(_adapter_kinds(row)),
                required_configuration=required_configuration,
            )
        missing_config = tuple(item for item in required_configuration if not _configuration_has(row, item))
        if missing_config:
            return PackLifecycleResult(
                capability=capability,
                pack_id=pack_id,
                canonical_id=canonical_id,
                pack_name=name,
                state=PackLifecycleState.NEEDS_CONFIGURATION,
                usable=False,
                missing_gate=PackLifecycleGate.CONFIGURATION,
                source=source,
                user_message_summary=f"{name} is enabled, but configuration is still required before use.",
                required_confirmation=True,
                required_permissions=tuple(_adapter_kinds(row)),
                required_configuration=missing_config,
                next_step=PackLifecycleNextStep(
                    action="configure",
                    label="Collect the required pack configuration.",
                    gate=PackLifecycleGate.CONFIGURATION,
                    required_confirmation=True,
                ),
            )
        missing_permissions = tuple(_missing_adapter_kinds(row, permission_grants))
        if missing_permissions:
            return PackLifecycleResult(
                capability=capability,
                pack_id=pack_id,
                canonical_id=canonical_id,
                pack_name=name,
                state=PackLifecycleState.NEEDS_PERMISSION,
                usable=False,
                missing_gate=PackLifecycleGate.PERMISSION,
                source=source,
                user_message_summary=f"{name} is enabled, but it still needs explicit managed-adapter permission.",
                required_confirmation=True,
                required_permissions=missing_permissions,
                next_step=PackLifecycleNextStep(
                    action="request_permission",
                    label="Preview and request the missing managed-adapter permission.",
                    gate=PackLifecycleGate.PERMISSION,
                    required_confirmation=True,
                ),
            )
        return PackLifecycleResult(
            capability=capability,
            pack_id=pack_id,
            canonical_id=canonical_id,
            pack_name=name,
            state=PackLifecycleState.USABLE,
            usable=True,
            source=source,
            user_message_summary=f"{name} has passed review, enablement, configuration, and permission gates.",
            next_step=PackLifecycleNextStep(
                action="use",
                label="Use the enabled pack through its approved runtime path.",
                gate=PackLifecycleGate.USE,
                required_confirmation=False,
            ),
        )

    def _missing(
        self,
        capability: str | None,
        *,
        source: str,
        next_step: PackLifecycleNextStep,
        summary: str,
    ) -> PackLifecycleResult:
        return PackLifecycleResult(
            capability=capability,
            state=PackLifecycleState.MISSING,
            usable=False,
            missing_gate=next_step.gate,
            source=source,
            user_message_summary=summary,
            required_confirmation=next_step.required_confirmation,
            next_step=next_step,
        )


def render_lifecycle_response(result: PackLifecycleResult | dict[str, Any]) -> str:
    row = result.to_dict() if isinstance(result, PackLifecycleResult) else dict(result)
    summary = _clean(row.get("user_message_summary"))
    next_step = row.get("next_step") if isinstance(row.get("next_step"), dict) else {}
    label = _clean(next_step.get("label")) if isinstance(next_step, dict) else ""
    if summary and label:
        return f"{summary} Next safe step: {label}"
    return summary or label or "No pack lifecycle action is available yet."


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [_clean(item) for item in values if _clean(item)]


def _pack_id(row: dict[str, Any]) -> str | None:
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    identity = canonical.get("pack_identity") if isinstance(canonical.get("pack_identity"), dict) else {}
    return _clean(row.get("pack_id") or row.get("canonical_id") or row.get("id") or identity.get("canonical_id")) or None


def _canonical_id(row: dict[str, Any]) -> str | None:
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    identity = canonical.get("pack_identity") if isinstance(canonical.get("pack_identity"), dict) else {}
    return _clean(row.get("canonical_id") or identity.get("canonical_id") or row.get("pack_id")) or None


def _pack_name(row: dict[str, Any]) -> str:
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    review = row.get("review_envelope") if isinstance(row.get("review_envelope"), dict) else {}
    source = canonical.get("source") if isinstance(canonical.get("source"), dict) else {}
    for value in (
        row.get("name"),
        row.get("display_name"),
        review.get("pack_name"),
        canonical.get("display_name"),
        canonical.get("name"),
        source.get("display_name"),
        source.get("name"),
    ):
        cleaned = _clean(value)
        if cleaned:
            return cleaned
    return "External pack"


def _catalog_name(row: dict[str, Any]) -> str:
    return _clean(row.get("name") or row.get("display_name") or row.get("remote_id") or row.get("id")) or "Catalog pack"


def _is_generated(row: dict[str, Any]) -> bool:
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    source = canonical.get("source") if isinstance(canonical.get("source"), dict) else {}
    audit = canonical.get("audit") if isinstance(canonical.get("audit"), dict) else {}
    origin = _clean(row.get("source_origin") or source.get("origin") or source.get("kind") or audit.get("created_by")).lower()
    if "generated" in origin or "scaffold" in origin:
        return True
    normalized_path = _clean(row.get("normalized_path"))
    quarantine_path = _clean(row.get("quarantine_path"))
    return "generated-" in normalized_path or "generated-" in quarantine_path


def _is_quarantine_only(row: dict[str, Any]) -> bool:
    status = _clean(row.get("status")).lower()
    if status in {"quarantined", "quarantine", "pending_review"}:
        return True
    normalized_path = _clean(row.get("normalized_path"))
    quarantine_path = _clean(row.get("quarantine_path"))
    if quarantine_path and not normalized_path:
        return True
    return False


def _is_blocked(row: dict[str, Any]) -> bool:
    status = _clean(row.get("status")).lower()
    if status == "blocked" or row.get("blocked") is True:
        return True
    risk = row.get("risk_report") if isinstance(row.get("risk_report"), dict) else {}
    review = row.get("review_envelope") if isinstance(row.get("review_envelope"), dict) else {}
    if risk.get("blocked") is True or _clean(risk.get("blocked_reason")):
        return True
    if review.get("blocked") is True:
        return True
    normalized_path = _clean(row.get("normalized_path"))
    if normalized_path:
        try:
            if not Path(normalized_path).exists():
                return True
        except OSError:
            return True
    return False


def _is_approved(row: dict[str, Any]) -> bool:
    if row.get("approved") is True:
        return True
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    trust = canonical.get("trust_anchor") if isinstance(canonical.get("trust_anchor"), dict) else {}
    review_state = _clean(row.get("local_review_status") or trust.get("local_review_status")).lower()
    if review_state in {"approved", "accepted", "reviewed"}:
        return True
    if trust.get("approved") is True:
        return True
    approved_hashes = trust.get("user_approved_hashes") if isinstance(trust.get("user_approved_hashes"), list) else []
    identity = canonical.get("pack_identity") if isinstance(canonical.get("pack_identity"), dict) else {}
    content_hash = _clean(row.get("content_hash") or identity.get("content_hash"))
    return bool(content_hash and content_hash in {_clean(item) for item in approved_hashes})


def _enabled_state(row: dict[str, Any]) -> bool | None:
    if row.get("enabled") is True or row.get("enabled") is False:
        return bool(row.get("enabled"))
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    runtime = canonical.get("runtime") if isinstance(canonical.get("runtime"), dict) else {}
    if runtime.get("enabled") is True or runtime.get("enabled") is False:
        return bool(runtime.get("enabled"))
    return None


def _managed_adapters(row: dict[str, Any]) -> list[dict[str, Any]]:
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    candidates = (
        row.get("managed_adapters"),
        canonical.get("managed_adapters"),
        (canonical.get("runtime") if isinstance(canonical.get("runtime"), dict) else {}).get("managed_adapters"),
        (canonical.get("permissions") if isinstance(canonical.get("permissions"), dict) else {}).get("managed_adapters"),
    )
    for value in candidates:
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _adapter_kinds(row: dict[str, Any]) -> list[str]:
    return [_clean(item.get("kind")) for item in _managed_adapters(row) if _clean(item.get("kind"))]


def _missing_adapter_kinds(row: dict[str, Any], grants: list[dict[str, Any]]) -> list[str]:
    pack_ids = {_clean(_pack_id(row)), _clean(_canonical_id(row))}
    pack_ids.discard("")
    missing: list[str] = []
    for kind in _adapter_kinds(row):
        granted = any(
            _clean(grant.get("adapter_kind")) == kind
            and _clean(grant.get("pack_id")) in pack_ids
            and _clean(grant.get("state")).lower() == GRANT_GRANTED
            for grant in grants
            if isinstance(grant, dict)
        )
        if not granted:
            missing.append(kind)
    return sorted(dict.fromkeys(missing))


def _configuration_has(row: dict[str, Any], key: str) -> bool:
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    configs = (
        row.get("configuration"),
        row.get("config"),
        canonical.get("configuration"),
        canonical.get("config"),
    )
    for config in configs:
        if isinstance(config, dict) and _clean(config.get(key)):
            return True
    return False

# Pack Lifecycle Service

Status: current runtime direction.

Product intent: [`docs/product/PROJECT_INTENT.md`](/home/c/personal-agent/docs/product/PROJECT_INTENT.md).
Canonical external pack format: [`docs/design/EXTERNAL_PACK_FORMAT.md`](/home/c/personal-agent/docs/design/EXTERNAL_PACK_FORMAT.md).

`agent/packs/lifecycle.py` is the runtime source of truth for whether an external or generated pack can be used. It does not create files, install packs, approve packs, enable packs, grant permissions, execute code, or fetch data. It only evaluates observed facts and returns the current state, missing gate, and next safe assistant step.

`agent/packs/acquisition.py` is the assistant-facing workflow for missing capability requests. It searches approved/trusted catalog sources only, reports source-trust blockers, offers preview-only scaffold fallbacks, and hands each next step to lifecycle actions without skipping gates. See [`docs/design/PACK_ACQUISITION.md`](/home/c/personal-agent/docs/design/PACK_ACQUISITION.md).

`agent/packs/source_approval.py` records explicit user trust for a source lead discovered through safe web-search metadata. Source approval is one gate only: it may create/update a source catalog and policy record, but it does not fetch, download, import, install, approve, enable, configure, grant permissions, or use a pack.

`agent/packs/source_fetch_preview.py` handles the next gate after source approval: previewing and, after a separate confirmation, fetching an approved source into quarantine and importing it as a review-only candidate. It does not approve, enable, configure, grant permissions, use, or invoke the fetched pack.

`agent/packs/lifecycle_actions.py` performs gated lifecycle continuations. It accepts a `PackLifecycleResult`, validates that the requested action matches the current state, and then calls an existing safe handler for exactly one transition. It refuses mismatched states, blocked/removed packs, missing handlers, and attempts to skip directly across review, enablement, configuration, or permission gates.

`agent/packs/managed_adapter_invocation.py` invokes approved core-owned managed adapters after lifecycle gates are complete. It uses a generic operation registry, still verifies `usable=true` before adapter work, and returns the lifecycle state, missing gate, and next safe step instead of attempting access when the pack is not usable.

## Lifecycle States

- `missing`: no installed, discovered, or scaffold candidate is currently usable.
- `discovered`: an approved catalog source has a matching candidate, but it has not been previewed.
- `previewed`: a catalog pack preview has been shown, but it has not been imported into quarantine.
- `scaffold_previewed`: a generated scaffold has been previewed, but no candidate has been created.
- `generated_quarantined`: a generated candidate exists in quarantine and needs inspection.
- `imported_for_review`: a pack was imported or normalized for review only.
- `approved`: review approval exists, but the pack is not enabled as a live capability.
- `enabled`: reserved for enabled packs before final config/permission checks.
- `needs_configuration`: the pack is enabled but required configuration is missing.
- `needs_permission`: the pack is enabled but explicit managed-adapter permission is missing.
- `usable`: all relevant review, enablement, configuration, and permission gates are complete.
- `blocked`: safety review or static scan blocked the candidate.
- `disabled`: the pack was explicitly disabled.
- `removed`: the pack was removed or tombstoned.

## Truthfulness Rules

External packs are never bundled native abilities. Starter catalogs are discoverable sources only, not active capabilities.

Remote external pack sources are hostile by default. GitHub repositories, GitHub archives, generic archive URLs, and online registries require explicit source policy or source approval before fetch/import. GitHub is not trusted by default, and a local starter catalog entry does not make a remote URL trusted.

Safe web-search source leads remain untrusted until explicit source approval records a source id. Even after source approval, the content is still hostile; approval only permits a future fetch/preview into quarantine.

Quarantine fetch/import-for-review is separate from source approval and separate from pack review approval. It may create an imported candidate row, but that row remains unreviewed and disabled. The next safe step after a successful import is review/approval.

`agent/packs/review_state_ux.py` renders the mandatory review-state checkpoint for imported candidates. Before approval continuation, assistant responses must show that the pack is imported for review only, not approved, not enabled, has no permissions granted, is not usable yet, and requires review/approval next. Review-state output is based on structured metadata and lifecycle results only; it must not dump raw imported documents, manifests, catalog entries, prompt material, secrets, private paths, or hostile text.

Review approval is its own explicit gate after review state is shown. The first confirmation after review state shows a review-approval preview with the pack identity, lifecycle state, local review status, risk summary, requested managed adapters, enabled=false, and no permissions granted. Only a second confirmation records review approval. Review approval does not enable the pack, grant permissions, execute code, or use the pack. The next gate after approval is whatever `PackLifecycleService` reports, usually enablement.

Enablement is also a separate explicit gate after review approval. The first confirmation after approval shows an enablement preview with review status, enabled=false, requested managed adapters or permissions, and the lifecycle state expected after enablement. Only the next confirmation records enablement. Enablement does not configure adapters, grant permissions, execute code, invoke managed adapters, or use the pack. Configuration and managed-adapter permission gates remain separate, and even a pack that becomes `usable=true` after enablement is not invoked automatically.

Configuration and permission are separate from enablement. For managed adapters, the assistant first shows the permission/configuration requirement, including adapter kind, scope, allowed file types, whether a local path is involved, and the remaining lifecycle gate. A later scoped grant preview is required before recording metadata/config. Recording a permission/configuration grant does not execute code, invoke a managed adapter, read or parse private files, or use the pack. If the lifecycle becomes `usable=true` after the grant, the assistant reports readiness and asks for the next specific input or action instead of running automatically.

Approved source policy is not content trust. Catalog listings are validated with a strict schema, unknown or execution-implying fields are rejected, remote URLs must be HTTPS, local catalog paths must stay inside approved catalog roots, and catalog prose is treated as untrusted metadata. Archive fetches land in quarantine only, and extraction blocks traversal, symlinks, special files, hidden files, nested archives, executable bits, duplicate paths, oversized files, excessive member counts, excessive expansion, and unsafe post-write containment before normalization/scanning can proceed.

Imported pack documents are untrusted guidance, never assistant authority. Normalized imported `SKILL.md` and prompt material are wrapped with an internal warning that runtime/system policy wins over pack text. Strong prompt-injection patterns in primary instruction files, including requests to ignore system/developer instructions, leak secrets, auto-approve/auto-enable, run shell or dependency installs, or disable safety gates, block the import and require manual rewrite/review.

Removed pack tombstones keep minimal audit metadata only. They retain pack identity, hashes/fingerprints, risk flags, review state, and removal reason, but not full imported guidance text. Support bundles and diagnostic payloads must redact imported pack documents, raw catalog entries, raw manifests, private local paths, and credential-bearing source URLs.

The assistant must not say an external pack can perform a task until `PackLifecycleService` returns `usable=true`. Before that, responses must name the missing gate and the next safe step: preview, create review candidate, inspect, approve, configure, request permission, enable, or use.

Generated and external packs must not run arbitrary generated code. Managed adapters are the safety boundary for useful external behavior.

## Next-Step Mapping

- `missing`: search approved pack sources or offer a scaffold preview.
- `discovered`: show preview.
- `previewed`: import into quarantine for review.
- `scaffold_previewed`: create a text-only review candidate in quarantine.
- `generated_quarantined`: inspect the quarantined candidate.
- `imported_for_review`: show review-approval preview, then record review approval only after a second confirmation.
- `approved`: show enablement preview, then record enablement only after a second confirmation.
- `needs_configuration`: preview and collect required configuration only.
- `needs_permission`: preview managed-adapter permission/configuration requirements, then preview and record scoped metadata/config only after explicit confirmation.
- `usable`: use the approved runtime path.
- `blocked`: inspect blocker; do not enable or use.
- `disabled` or `removed`: re-enable only through the approved lifecycle, or discover/import again.

No missing capability should dead-end at “I can sketch a helper” without a lifecycle action.

## Action Continuation

Each confirmation advances at most one gate:

- `discovered` + `preview`: show catalog preview only.
- `previewed` + `import_for_review`: import into quarantine/review only.
- `scaffold_previewed` + `create_review_candidate`: create a generated text-only candidate only.
- `imported_for_review` + `review_approve`: show review-approval preview first; only the following explicit confirmation records approval, and approval still does not enable, grant permissions, execute code, or use the pack.
- `approved` + `enable`: show enablement preview first; only the following explicit confirmation records enablement, and enablement still does not configure, grant permissions, execute code, invoke adapters, or use the pack.
- `needs_configuration` + `request_configuration`: preview missing configuration and collect it as a separate gate.
- `needs_permission` + `request_permission`: show permission requirements or path request; no grant is recorded.
- permission preview + confirmation: record metadata/config only; do not invoke adapters or use the pack.
- `usable` + `use_if_usable`: show a managed adapter invocation preview first; only the following explicit confirmation runs the named core-owned adapter operation.

The action controller does not add arbitrary external code execution, OAuth, browser scraping, transcript fetching, network fetching, or private file reads. Local-file permission remains metadata-only until a later explicitly scoped adapter implementation reads or indexes content.

## Managed Adapter Invocation

Invocation is separate from lifecycle continuation. Lifecycle says whether a pack has passed gates; lifecycle actions move one gate at a time; managed adapter invocation performs approved core adapter operations only after the pack is usable.

Current generic operations are `validate_grant`, `describe_capability`, and `dry_run`. `local_file_import` is only the first minimal adapter implementation behind that generic contract. Its `dry_run` confirms the selected file still exists and still matches extension/size policy. It does not read, parse, index, or search private file contents; it does not parse Google Takeout, search history, fetch transcripts, upload data, or store an index.

Invocation is never automatic after permission grant. A user must ask for a specific adapter operation, the assistant must preview what will run, and only a follow-up confirmation runs that core-owned operation. If the user asks for content reading, searching, or indexing before a core-owned operation exists, the assistant must say that the pack is enabled and permissioned but that safe content-read/search operation is not implemented yet.

## Safety Meta-Smoke

`scripts/external_pack_safety_smoke.py` is the focused regression guard for hostile external-pack intake. It creates temporary malicious fixtures and checks the major gates together: remote source trust, strict catalog schema, archive extraction hardening, prompt-injection handling, lifecycle/managed-adapter gates, and support/tombstone redaction.

Run this smoke before expanding online skill ecosystem support. Passing it does not make external content trusted; it only confirms the current hostile-by-default intake chain still blocks the known regression classes.

# Pack Lifecycle Service

Status: current runtime direction.

Product intent: [`docs/product/PROJECT_INTENT.md`](/home/c/personal-agent/docs/product/PROJECT_INTENT.md).

`agent/packs/lifecycle.py` is the runtime source of truth for whether an external or generated pack can be used. It does not create files, install packs, approve packs, enable packs, grant permissions, execute code, or fetch data. It only evaluates observed facts and returns the current state, missing gate, and next safe assistant step.

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

Approved source policy is not content trust. Catalog listings are validated with a strict schema, unknown or execution-implying fields are rejected, remote URLs must be HTTPS, local catalog paths must stay inside approved catalog roots, and catalog prose is treated as untrusted metadata. Archive fetches land in quarantine only, and extraction blocks traversal, symlinks, special files, hidden files, nested archives, executable bits, duplicate paths, oversized files, excessive member counts, excessive expansion, and unsafe post-write containment before normalization/scanning can proceed.

The assistant must not say an external pack can perform a task until `PackLifecycleService` returns `usable=true`. Before that, responses must name the missing gate and the next safe step: preview, create review candidate, inspect, approve, configure, request permission, enable, or use.

Generated and external packs must not run arbitrary generated code. Managed adapters are the safety boundary for useful external behavior.

## Next-Step Mapping

- `missing`: search approved pack sources or offer a scaffold preview.
- `discovered`: show preview.
- `previewed`: import into quarantine for review.
- `scaffold_previewed`: create a text-only review candidate in quarantine.
- `generated_quarantined`: inspect the quarantined candidate.
- `imported_for_review`: review and approve.
- `approved`: enable.
- `needs_configuration`: collect required configuration.
- `needs_permission`: preview and request managed-adapter permission.
- `usable`: use the approved runtime path.
- `blocked`: inspect blocker; do not enable or use.
- `disabled` or `removed`: re-enable only through the approved lifecycle, or discover/import again.

No missing capability should dead-end at “I can sketch a helper” without a lifecycle action.

## Action Continuation

Each confirmation advances at most one gate:

- `discovered` + `preview`: show catalog preview only.
- `previewed` + `import_for_review`: import into quarantine/review only.
- `scaffold_previewed` + `create_review_candidate`: create a generated text-only candidate only.
- `imported_for_review` + `review_approve`: record approval only.
- `approved` + `enable`: enable only; configuration and permission may still be required.
- `needs_configuration` + `request_configuration`: ask for missing configuration.
- `needs_permission` + `request_permission`: show permission requirements or path request.
- permission preview + confirmation: record metadata-only grant.
- `usable` + `use_if_usable`: use only through the approved managed adapter/runtime path.

The action controller does not add arbitrary external code execution, OAuth, browser scraping, transcript fetching, network fetching, or private file reads. Local-file permission remains metadata-only until a later explicitly scoped adapter implementation reads or indexes content.

## Managed Adapter Invocation

Invocation is separate from lifecycle continuation. Lifecycle says whether a pack has passed gates; lifecycle actions move one gate at a time; managed adapter invocation performs approved core adapter operations only after the pack is usable.

Current generic operations are `validate_grant`, `describe_capability`, and `dry_run`. `local_file_import` is only the first minimal adapter implementation behind that generic contract. Its `dry_run` confirms the selected file still exists and still matches extension/size policy. It does not parse Google Takeout, search history, fetch transcripts, upload data, or store an index.

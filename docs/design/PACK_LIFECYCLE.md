# Pack Lifecycle Service

Status: current runtime direction.

Product intent: [`docs/product/PROJECT_INTENT.md`](/home/c/personal-agent/docs/product/PROJECT_INTENT.md).

`agent/packs/lifecycle.py` is the runtime source of truth for whether an external or generated pack can be used. It does not create files, install packs, approve packs, enable packs, grant permissions, execute code, or fetch data. It only evaluates observed facts and returns the current state, missing gate, and next safe assistant step.

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

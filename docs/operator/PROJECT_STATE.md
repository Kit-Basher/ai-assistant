# Project State

This is the operator truth snapshot for the current local checkpoint. It is not
marketing copy and it is not a final release claim.

## Current Checkpoint

- Tag: `v0.2.1-daily-driver-maturity-audit-clean`
- Commit: `7410cf9`
- Fresh Debian VM proof: not run
- Release status: ready for VM proof, not finished

Current confirmed proof:

- `python scripts/installed_product_abuse.py`: `PASS=42 WARN=0 FAIL=0`
- `python scripts/operator_lifecycle_smoke.py`: installed operator lifecycle
  preview lane passes
- `python scripts/memory_lifecycle_smoke.py`: installed memory lifecycle
  preview lane passes
- `python scripts/plan_mode_v2_smoke.py`: installed Plan Mode v2 proof lane
  passes
- `python scripts/executor_registry_smoke.py`: installed Executor Registry v1
  proof lane for preview-only refusal and the safe support-bundle executor
- `python scripts/support_bundle_v2_smoke.py`: installed Support Bundle v2
  diagnostics packaging proof
- `python scripts/backup_v1_smoke.py`: installed Backup v1 additive backup
  proof for redacted bounded summaries; live restore remains dry-run-only
- `python scripts/restore_validator_smoke.py`: installed Restore v1 Validator
  proof for read-only backup listing, Backup v1 validation, unsafe path
  rejection, malformed backup handling, and restore preview-only refusal
- `python scripts/cleanup_preview_smoke.py`: installed cleanup preview proof
  for old/oversized backup, support bundle, and runtime-release candidates;
  deletion remains disabled
- `python scripts/first_run_smoke.py`: isolated first-run/fresh-state proof
  for a temporary API with empty HOME/XDG/state paths; this is not the fresh
  Debian VM proof
- `docs/operator/VM_PROOF.md`: documented clean-host Debian VM proof plan;
  the VM proof itself has not been run
- `docs/operator/DAILY_DRIVER_MATURITY.md` and
  `python scripts/daily_driver_maturity_audit.py`: recurring installed-product
  maturity audit for daily-driver blockers, irritants, state growth, and
  performance drift. It includes stale diagnostic-context checks so rewrite/edit
  prompts and ambiguous correction prompts do not replay old doctor/status
  output.
- `docs/operator/TEST_SUITE_RATIONALIZATION.md`: current inventory and
  consolidation plan for pytest modules, smoke/proof scripts, release blockers,
  daily-driver irritant checks, historical/manual proofs, and recommended
  command groups.
- `python scripts/prove_daily_driver_product.py`: `PASS`
- `python scripts/daily_driver_smoke.py --timeout 90`: `PASS=9 BLOCKED=0 FAIL=0`
- `python scripts/prove_pre_vm_complete.py`: `PRE_VM_COMPLETE=yes`, `BLOCKERS=0`, `UNKNOWN_AREAS=0`, `WARNINGS=7`
- `python scripts/prove_ready.py`: `READY_FOR_VM_PROOF=yes`, `RELEASE_BLOCKERS=0`, `WARNINGS=2`
- `git status`: clean at the checkpoint

New restart/browser survival lane:

- `python scripts/restart_survival_smoke.py`: automated stable API service
  restart proof. It is not a full PC reboot proof.
- `docs/operator/REBOOT_PROOF.md`: manual reboot and browser survival
  checklist.

Earlier readiness gates passed before the first real web UI search request
failed. That failure proved the repo needed an installed-product gate, not only
internal and mock-heavy tests. `installed_product_abuse.py` and
`prove_daily_driver_product.py` are now the stronger product-facing gates.

## Meaning Of Gates

- `scripts/prove_ready.py`: canonical pre-VM readiness gate. It separates
  release-blocking failures from runtime-state warnings and expected isolated
  proof limits.
- `scripts/prove_pre_vm_complete.py`: broader local subsystem gate. It asks
  whether the local runtime is hardened enough that the fresh VM proof should
  be confirmation, not discovery.
- `scripts/installed_product_abuse.py`: strict installed-runtime abuse harness.
  It talks to the promoted API surface, verifies runtime freshness, checks
  endpoint wiring, and drives confused user flows for search, Telegram, memory,
  and Plan Mode.
- `scripts/prove_daily_driver_product.py`: wrapper around the installed-product
  abuse harness for the current daily-driver product proof.
- `scripts/daily_driver_smoke.py`: user-facing smoke for ready/state/search,
  pack use, package preview, ordinary chat, and doctor. It can repair
  configured-stopped managed search through the assistant Plan Mode flow.
- `scripts/release_smoke.py`: deterministic release smoke for core behavior.
- `scripts/perf_smoke.py`: read-only latency and no-LLM deterministic route
  check. Small latency warnings are not release blockers by themselves.
- `scripts/restart_survival_smoke.py`: installed stable service stop/start
  proof. It verifies runtime freshness, status surfaces, managed search repair,
  metadata-only search after restart, Telegram optional wording, and stale
  confirmation rejection across service restart.
- `scripts/operator_lifecycle_smoke.py`: installed-runtime proof for health,
  broken-status, storage, repair, backup, restore, update, cleanup, uninstall,
  and support-bundle user-facing lifecycle prompts. It proves safe previews and
  cancellation, not full destructive execution.
- `scripts/memory_lifecycle_smoke.py`: installed-runtime proof for memory
  inspection, status, current-turn opt-out, thread/global memory controls,
  forget/delete/export/redact/dedupe previews, cancellation, and stale
  confirmation rejection. It proves safe previews, not full destructive memory
  execution.
- `scripts/plan_mode_v2_smoke.py`: installed-runtime proof for canonical Plan
  objects, current-plan inspection, cancellation/revision, preview-only
  executor blocking, stale confirmation rejection after service restart, and
  thread/session confirmation binding.
- `scripts/executor_registry_smoke.py`: installed-runtime proof for the shared
  executor registry result shape, preview-only refusals, support-bundle
  execution, registry journal ids, stale confirmation rejection, and
  thread/session binding.
- `scripts/support_bundle_v2_smoke.py`: installed-runtime proof for the useful
  Support Bundle v2 diagnostics package: manifest, bounded summaries,
  redaction, registry result fields, and scoped rollback hint.
- `scripts/backup_v1_smoke.py`: installed-runtime proof for Backup v1:
  Plan Mode preview, Executor Registry confirmation, timestamped local backup
  directory, manifest, bounded redacted summary files, scoped rollback hint,
  and restore dry-run/no-mutation behavior.
- `scripts/restore_validator_smoke.py`: installed-runtime proof for Restore v1
  Validator. It lists recent backups, identifies the latest valid backup,
  validates Backup v1 manifests and required summaries, rejects unsafe outside
  paths, detects missing manifests, and keeps live restore disabled.
- `scripts/cleanup_preview_smoke.py`: installed-runtime proof for cleanup
  preview. It classifies old/oversized backup artifacts, old support bundles,
  and old runtime releases, protects current runtime/latest backup/secrets, and
  proves confirmation remains `preview_only` with `mutated=false`.
- `scripts/first_run_smoke.py`: isolated first-run/fresh-state proof. It starts
  a temporary API on a random loopback port with isolated HOME/XDG/state/config
  paths. It does not overwrite the promoted stable runtime and it is not the
  full clean Debian VM install proof.
- `scripts/vm_proof_smoke.py`: post-install clean-VM API smoke. It runs against
  `http://127.0.0.1:8765` after the VM install and verifies status surfaces,
  optional Telegram, unconfigured search guidance, Plan Mode previews, support
  preview, backup preview, and restore preview-only behavior. It does not run
  the installer by itself.
- `scripts/daily_driver_maturity_audit.py`: recurring installed-product audit
  for startup/search/Telegram/memory/operator/backup/user-friction/performance
  and state-growth maturity. It reports blockers separately from irritants and
  does not confirm enabled mutating actions. It also checks that stale
  status/doctor context does not hijack unrelated provided-text transforms or
  ambiguous corrections.
- `docs/operator/TEST_SUITE_RATIONALIZATION.md`: test/proof-suite map. It
  defines the fast local dev check, behavior gate, installed daily-driver check,
  operator safety check, full local release proof, and historical/manual proof
  groups so future work does not create another overlapping proof lane by
  default.

## Proven Now

- The promoted stable runtime reports build metadata and can be compared with
  the checkout commit.
- The installed API surface exists for the routes used by the web UI and
  operator proof scripts.
- Managed local SearXNG setup and recovery are confirmation-gated and
  product-facing.
- `configured_stopped` SearXNG state repairs through assistant Plan Mode and
  then continues the original lookup.
- Search remains metadata-only: no page fetch, browser automation, downloads,
  JavaScript, or pack install from search.
- Podman detection works from the stable service, including approved absolute
  paths such as `/usr/bin/podman`.
- Telegram configured-but-inactive is reported as optional, not whole-agent
  failure.
- Telegram start/restart/stop prompts route to bounded Plan Mode previews.
- Common prompt-injection and ambiguous action prompts do not bypass Plan Mode.
- Stale confirmations and missing confirmations are rejected cleanly.
- External pack safety smoke still passes and packs remain text-only unless
  separately approved through explicit lifecycle gates.
- Daily-driver smoke passes with installed managed search running.
- The stable API service can survive a controlled service stop/start in the
  automated restart smoke. Actual PC reboot remains a separate manual proof.
- Operator lifecycle prompts now route deterministically through the installed
  `/chat` API. Read-only health/storage prompts answer directly; repair,
  backup, restore, update, cleanup, uninstall, and support-bundle prompts show
  confirmation-gated previews.
- Memory lifecycle prompts now route deterministically through the installed
  `/chat` API. Memory inspection/status and current-turn opt-out are
  deterministic; destructive or broad controls show confirmation-gated previews.
- Plan Mode previews expose canonical v2 fields including plan id, action type,
  target, scope, mutation level, affected resources, risk, rollback scope,
  executor status, allowed confirmations, and expiry. Users can inspect, cancel,
  revise, or confirm the current pending plan.
- Executor Registry v1 sits behind Plan Mode for the first wired lifecycle
  actions. It refuses preview-only memory/delete/uninstall/cleanup actions with
  `mutated=false`, records a redacted journal row, and can execute the safe
  additive support-bundle and backup executors.
- Support Bundle v2 creates a temporary redacted diagnostics package with
  doctor/version/ready/state/search/Telegram/pack/journal/git summaries. It
  does not include raw logs, raw secrets, arbitrary home data, or destructive
  cleanup.
- Backup v1 creates an approved local timestamped backup directory with a
  manifest, redacted state/preferences/memory/pack/runtime/journal summaries,
  and explicit exclusions for raw secret stores, logs, arbitrary home data,
  model caches, and untrusted pack/source text. Restore remains
  dry-run/preview-only and live restore is not enabled.
- Restore v1 Validator can inspect Backup v1 artifacts read-only and explain
  validity, missing files, warnings, and restore-disabled state without writing
  or restoring anything.
- Cleanup preview identifies old or oversized Personal Agent artifacts and
  estimates recoverable space without deleting anything. Cleanup remains
  preview-only.
- A fresh isolated user state can start a temporary API, return coherent
  `/ready`, `/state`, and `/version` JSON, serve the web UI entrypoint, report
  missing Telegram as optional, report unconfigured search honestly, start with
  empty memory wording, and keep package/support/backup/restore/cleanup prompts
  safe.

## Still Partial

These are not unknowns, but they are not finished:

- Installer/update/uninstall: install, promotion, bundle, and package paths have
  coverage; user-facing update/uninstall previews exist; full execution and
  fresh-host partial-failure recovery remain partial.
- Storage/log growth: read-only installed storage estimate and cleanup preview
  exist; rotation/enforced cleanup policy remains partial.
- Web UI robustness: static/component smoke exists; browser automation and
  manual refresh/large-transcript checks remain. `REBOOT_PROOF.md` carries the
  manual UI checklist.
- Telegram runtime behavior: optional-service semantics and status UX are
  covered; full start/stop/restart execution proof remains partial.
- Memory completion: audits, deterministic status/inspection/current-turn
  opt-out, and preview UX exist; full delete/export/redact/dedupe executors and
  richer explainability remain partial.
- Plan Mode execution: canonical plan previews, inspection, cancellation,
  thread/session binding, stale-confirmation rejection, and Executor Registry
  v1 exist. Most mutators are not yet migrated to the registry; destructive
  lifecycle and memory actions remain preview-only.
- Release/CI automation: CI-safe and live-runtime gates are split; broader CI
  adoption remains future work.
- Model/provider management: deterministic guidance and switching paths are
  covered; opt-in real local LLM soak remains future work.
- Fresh Debian VM proof: intentionally not run yet.
- First-run/fresh-state proof: isolated state is covered by
  `first_run_smoke.py`; full clean Debian VM install remains intentionally not
  run yet.
- VM proof plan: `docs/operator/VM_PROOF.md` and `scripts/vm_proof_smoke.py`
  define the clean-host install proof, but the proof has not been executed on a
  disposable VM yet.

## What Not To Claim

- Do not claim the agent is finished.
- Do not claim final release readiness.
- Do not claim the fresh Debian VM proof is complete.
- Do not claim full memory completion.
- Do not claim full installer/update/uninstall lifecycle completion.
- Do not claim bug-free behavior.
- Do not claim all external skills execute or are complete.
- Do not claim full web browsing; current search is metadata-only.

Safe wording:

- “ready for VM proof”
- “installed daily-driver product gate passes”
- “not finished”
- “release candidate pending VM proof and lifecycle completion”

## Next Release Lanes

1. Manual reboot proof.
2. Automated browser/UI survival proof.
3. Operator lifecycle execution: Backup v1 is now additive and executable;
   restore/update/cleanup/uninstall remain preview-only until bounded
   executors exist.
4. Memory lifecycle execution: implement bounded executors for thread/global
   toggles, forget-topic, delete-all, export, redaction, and dedupe.
5. Executor Registry expansion: move remaining per-flow mutators behind the
   canonical apply/recovery interface when each bounded executor is ready.
6. Skill pack lifecycle hardening.
7. Model/provider management and real local LLM soak.
8. Clean VM proof.
9. Release candidate.

## Operator Quick Commands

```bash
bash scripts/promote_local_stable.sh
python scripts/installed_product_abuse.py
python scripts/prove_daily_driver_product.py
python scripts/operator_lifecycle_smoke.py
python scripts/memory_lifecycle_smoke.py
python scripts/plan_mode_v2_smoke.py
python scripts/executor_registry_smoke.py
python scripts/support_bundle_v2_smoke.py
python scripts/backup_v1_smoke.py
python scripts/restore_validator_smoke.py
python scripts/cleanup_preview_smoke.py
python scripts/first_run_smoke.py
python scripts/daily_driver_maturity_audit.py
python scripts/vm_proof_smoke.py --expected-commit da6c71e
python scripts/daily_driver_smoke.py --timeout 90
python scripts/restart_survival_smoke.py
python scripts/prove_pre_vm_complete.py
python scripts/prove_ready.py
git status
```

## Search Lifecycle Truth

- `never_configured`: no trusted endpoint is configured. First public lookup
  offers local SearXNG setup and requires confirmation.
- `configured_running`: trusted endpoint is configured and JSON metadata search
  works. Public lookups search immediately.
- `configured_stopped`: trusted endpoint is configured but unreachable. Public
  lookups offer inline managed start/repair confirmation. After approval,
  Personal Agent repairs only the trusted managed endpoint, rechecks status,
  and continues the original lookup if search becomes available.
- `invalid_or_untrusted_config`: persisted or configured endpoint failed trust
  checks. The runtime refuses to use it and offers safe reconfiguration.

Search is not a browser. It returns untrusted SearXNG metadata only.

## Telegram Truth

- Telegram is optional.
- Configured but inactive Telegram is not a whole-agent failure.
- Start/restart/stop are bounded Plan Mode actions.
- Telegram tokens must remain redacted from status, chat, logs, docs, and
  support output.

# Project State

This is the operator truth snapshot for the current local checkpoint. It is not
marketing copy and it is not a final release claim.

## Current Checkpoint

- Tag: `v0.2.2-runtime-latency-closure-v1`
- Commit: `34632188bcd90ad41e74ba7e188db905dfa710dc`
- Fresh Debian VM proof: not run
- Release status: final release audit is active; final tag has not been created

## Active Phase: v0.2.2 Final Release Audit and Version Decision

Authorization, bypass hardening, adversarial proof, and latency closure are
complete.

This phase verifies release truth, compatibility, installation behavior,
documentation, reproducibility, rollback guidance, and semantic-versioning
classification before creating the final release tag.

Current version decision:

- Recommended final product version: `v0.2.2`.
- Rationale: the work completes the intended v0.2 authorization and release
  foundation while preserving normal user/API/state compatibility. It hardens
  mutation behavior and skill-pack permissions, but unsupported or unsafe
  paths are denied rather than removed supported workflows.
- Rejected option: `v0.3.0`; it would imply a new schema/minor line in current
  doctor/version checks and would overstate migration burden for this release.

Current latency closure evidence:

- `python scripts/runtime_latency_investigation.py`: `PASS=11 WARN=0 FAIL=0`,
  warm `/ready` p95 1 ms, direct `htop` package-state p95 0 ms, package Plan
  preview p95 1193 ms, pending confirmation lookup p95 2932 ms.
- `python scripts/runtime_latency_closure_smoke.py`: `PASS=18 WARN=0 FAIL=0`.
- Remaining latency risks are accepted runtime-state warnings with checked-in
  records and revisit triggers in `docs/operator/RUNTIME_LATENCY_ACCEPTANCE_V1.json`.

## Completed Checkpoint: Full Adversarial Authorization Proof v1

The central authorization architecture and generic mutation boundary are
implemented.

This phase attacks the full authorization chain end-to-end using forged
identities, replay attempts, target drift, cross-scope reuse, direct primitive
access, callback forgery, and partial or uncertain execution outcomes.

The goal is proof, not new capability expansion.

The current guarantee remains a platform API permission boundary. It does not
claim process isolation for arbitrary malicious in-process Python code.

Confirmed proof:

- `python scripts/full_adversarial_authorization_proof.py`: 55 cases,
  10/10 properties proven, `FAIL=0`, `SKIP=0`, `RELEASE_BLOCKERS=0`.

## Completed Checkpoint: Generic Mutation Bypass Hardening v1

First-party and skill-pack mutation paths now use central capability policy,
Universal Mutation Plans, trusted invocation context, and Executor Registry
bindings.

This phase performs a repository-wide bypass review across shell, filesystem,
network, provider, database, API, background, maintenance, and internal helper
paths.

The goal is to prove that supported mutation cannot occur outside the central
authorization architecture.

## Completed Checkpoint: Skill-Pack Permission Boundary v1

First-party mutation paths now use central capability policy and Universal
Mutation Plans. Skill-Pack Permission Boundary v1 establishes a hard permission
boundary for installed skill packs. Skill-pack identity, declared permissions,
granted permissions, capability bindings, and runtime authorization are
enforced before privileged platform API actions.

## Completed Checkpoint: Communications Mutation Migration v1

Files, Git, and service-control mutation lanes now use central capability
policy and Universal Mutation Plans.

This phase migrates implemented communications mutations, including supported
notification delivery and notification-history mutation. No email or calendar
provider is currently implemented in this repository, so those providers remain
unsupported rather than silently mapped to a generic transport.

## Completed Checkpoint: Files, Git, and Service Mutation Migration v1

Executor Authorization Migration v1 brought backup, restore, support bundles,
and memory lifecycle under central authorization. Files, Git, and Service
Mutation Migration v1 migrated the remaining local-host mutation areas:

- file creation, modification, and deletion;
- Git commit and push policy boundaries;
- service restart fixture control.

## Completed Checkpoint: Executor Authorization Migration v1

Universal Plan Mode v1 established one mutation protocol. Executor
Authorization Migration v1 migrated the next legacy executor group:

- backup creation;
- restore execution;
- support-bundle creation;
- memory lifecycle mutation.

## Completed Checkpoint: Universal Plan Mode Enforcement v1

Capability Policy Schema v1 established the authorization vocabulary and central
gate.

This phase standardizes the complete mutation journey:

1. inspect and validate;
2. produce a versioned mutation Plan;
3. bind confirmation to the Plan;
4. revalidate runtime truth;
5. authorize execution;
6. mutate through the trusted executor;
7. record durable outcome and receipt.

The first migrated actions are package install, cleanup, update, and uninstall.
Legacy mutation paths remain audit-visible and are not claimed as universally
protected yet.

## Completed Checkpoint: Capability Policy Schema and Central Authorization Gate v1

The v0.2.1 lifecycle roadmap is complete. Capability Policy Schema v1 added the
central capability registry, structured authorization decisions, trusted
invocation context, capability metadata in Plan Mode and receipts, and central
enforcement for cleanup, update, uninstall, and package-install bypass blocking.

## Completed Phase: v0.2.1 Release Closure and Installed-Product Hardening

The lifecycle roadmap is complete through:

- primary non-no-op update;
- verified rollback;
- primary preserve-data uninstall wiring;
- strict local uninstall activation policy.

Next work:

1. Clean minor repository and host-policy issues.
2. Diagnose remaining latency warnings.
3. Run a full sequential installed-product proof from the current checkpoint.
4. Audit Debian setup, recovery, backup, update, and uninstall documentation.
5. Cut a v0.2.1 release candidate or final v0.2.1 checkpoint.

After release closure, return to broader tool authorization and Plan Mode
policy maturity.

## RC1 Latency Closure

The remaining RC1 warnings were performance-related. Current measured closure:

- `python scripts/rc1_latency_closure_smoke.py`: `PASS=6 WARN=0 FAIL=0`,
  warm `/ready` median/p90 1 ms, `htop` preview median 843 ms and p90 976 ms
  in the full sequential closure run.
- `python scripts/daily_driver_maturity_audit.py`: `PASS=35 WARN=0 FAIL=0`;
  `install htop` preview distribution median 1088 ms and p90 1152 ms after
  same-user pending-state bounding.
- `python scripts/perf_smoke.py`: `PASS=10 WARN=0 FAIL=0`.
- `python scripts/prove_ready.py`: `PASS=14 WARN=0 FAIL=0 NOTES=1`.
- `python scripts/v0_2_1_release_closure.py`: `PASS=29 WARN=0 FAIL=0 SKIP=0`.

The note is expected isolated-proof classification, not a release warning.
No destructive active primary uninstall was run, and the primary uninstall
activation marker remains absent.

## Current Confirmed Proof

- `python scripts/installed_product_abuse.py`: `PASS=40 WARN=0 FAIL=0`
  when search starts `configured_running`; `PASS=42 WARN=0 FAIL=0` when search
  starts `configured_stopped` and the repair preview plus approval branch is
  exercised successfully.
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
  proof for redacted bounded summaries and restore preview cancellation
- `python scripts/restore_validator_smoke.py`: installed Restore v1 Validator
  proof for read-only backup listing, Backup v1 validation, unsafe path
  rejection, malformed backup handling, and restore preview cancellation
- `python scripts/restore_execution_smoke.py`: isolated Restore Executor v1
  proof for validated Backup v1 restore, staging, pre-restore safety snapshot,
  allowlisted preference apply, duplicate confirmation safety, and rollback on
  forced post-apply verification failure. It does not restore real personal
  daily-driver state.
- `python scripts/update_execution_smoke.py`: isolated Update Executor v1 proof
  for staged release promotion, rollback checkpoint, forced post-promotion
  rollback, dirty-tree refusal, target-drift refusal, and verified live no-op.
  It does not update the real daily-driver runtime to an unknown commit.
- `python scripts/uninstall_execution_smoke.py`: isolated Uninstall Executor v1
  proof for preserve-data fixture uninstall, final safety backup, receipt,
  idempotency, live daily-driver guard, symlink-escape rejection, and truthful
  partial-failure reporting. It does not uninstall the real daily-driver
  runtime.
- `python scripts/host_lifecycle_runner_smoke.py`: shared Host Lifecycle Runner
  v1 proof for fixture update promotion, forced rollback, preserve-data
  uninstall, duplicate/idempotent execution, tamper rejection, and arbitrary
  command-field rejection.
- `python scripts/host_lifecycle_systemd_smoke.py`: installed-host systemd
  handoff proof for a fixture update unit. It uses fixture roots and fixture
  unit names only; it does not stop or remove the active Personal Agent
  service.
- `python scripts/active_host_enablement_smoke.py`: installed-host proof using
  a real alternate Personal Agent instance with separate runtime/state roots,
  a proof-prefixed user-systemd API service, and a non-primary loopback port. It
  proves alternate Release A readiness, real A -> B host-runner update, HTTP
  reconnect, forced rollback to A, interrupted runner resume, preserve-data
  uninstall after API shutdown, post-uninstall host status, reinstall sanity,
  and primary-installation protection. It does not update or uninstall the
  primary daily-driver installation.
- `python scripts/primary_update_enablement_smoke.py --allow-primary-update-proof
  --expected-commit <current-serving-commit>`: explicit installed-host proof
  for primary daily-driver update handoff. It builds a second release from the
  currently serving trusted commit, drives the normal Plan Mode `/chat`
  confirmation path, launches Host Lifecycle Runner v1 through user systemd,
  verifies API disconnect/recovery and runtime path switch, injects one
  proof-only post-promotion failure, verifies rollback against the serving API,
  and confirms primary uninstall still returns
  `uninstall_live_execution_not_enabled`. This proof is intentionally not run
  automatically by `prove_ready.py`.
- `python scripts/primary_uninstall_enablement_smoke.py
  --allow-primary-uninstall-shaped-proof --expected-commit
  <current-serving-commit>`: explicit production-shaped preserve-data uninstall
  proof. It creates an isolated install-shaped Personal Agent tree, confirms
  uninstall through `operator.uninstall.v1`, runs the shared Host Lifecycle
  Runner path, verifies final backup/receipt/preserved data, proves duplicate
  idempotency and truthful partial failure, and verifies the active primary
  installation remains unchanged. It does not uninstall the active primary
  daily-driver runtime and is intentionally not run automatically by
  `prove_ready.py`.
- `python scripts/primary_uninstall_policy_smoke.py`: strict activation-policy
  proof for primary preserve-data uninstall. It validates the v1 marker schema,
  installation binding, permissions, expiry, integrity, local enable/disable
  helpers, update-shaped marker survival, marker consumption, reinstall default
  disabled state, and actual-host read-only status. It never enables or
  confirms uninstall against the active primary installation.
- `python scripts/skill_pack_permission_boundary_smoke.py`: skill-pack
  platform API permission-boundary proof for v1 manifest validation,
  declared/granted/effective permission handling, target-scope enforcement,
  identity/version/fingerprint binding, brokered Universal Plan dispatch,
  direct helper blocking, shell/HTTP/secret platform API denial, grant
  revocation, update permission diffs, receipt metadata, and honest
  in-process Python isolation limitation reporting. It uses fixture skill packs
  only and does not load untrusted external code into the primary runtime.
- `python scripts/generic_mutation_bypass_audit.py`: static reviewed-inventory
  audit for supported mutation surfaces across subprocess, filesystem, SQL,
  HTTP/provider, Git, systemctl, secret, registry, and trusted-context sites.
- `python scripts/generic_mutation_bypass_smoke.py`: dynamic denial proof for
  direct primitive access, copied/stale/expired/consumed contexts, registry
  freeze, API override rejection, and the documented process-isolation
  limitation.
- `python scripts/full_adversarial_authorization_proof.py`: comprehensive
  fixture-based attack matrix for fixed authority, exact target binding,
  single-use authorization, scope isolation, runtime truth, primitive
  enforcement, durable mutation truth, failure truth, fail-closed behavior, and
  fixture isolation. It writes machine-readable evidence to
  `/tmp/full_adversarial_authorization_proof_evidence.json` and reports the
  in-process Python process-isolation limitation as an explicit non-blocking
  warning.
- `python scripts/cleanup_preview_smoke.py`: installed cleanup preview proof
  for old/oversized backup, support bundle, and runtime-release candidates;
  the installed daily-driver plan is cancelled during this smoke.
- `python scripts/cleanup_execution_smoke.py`: cleanup execution proof. It
  deletes only an isolated generated fixture through the Executor Registry and
  verifies the installed daily-driver cleanup plan is enabled but not executed
  by the proof.
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
- `docs/operator/RELIABILITY_COVERAGE_GAP_AUDIT.md`: subsystem reliability map
  for startup/runtime, search, Telegram, memory, Plan Mode, executor registry,
  backup/restore/cleanup, packs, secrets, UI/API behavior, and response
  routing. It names the highest-risk missing fault-injection guarantees and the
  next five implementation items. Search Lifecycle Fault Injection Batch 1 now
  covers invalid endpoints, HTML/non-JSON responses, malformed JSON, missing
  `results`, timeouts, unsafe setup URLs, local repo/planning prompt
  suppression, and correction back to search in focused tests. Telegram
  Reliability Batch 2 now covers stale dead-PID lock cleanup, live lock
  preservation, duplicate poller reporting through `/telegram/status` and
  doctor, process-list inspection failure, token redaction from status state,
  and chat wording for stale locks/duplicate pollers. Secret-store Reliability
  Batch 3 now covers missing/corrupt/decrypt-failed store status,
  `agent.secrets get --redacted`, doctor corruption wording, `/config`,
  `/telegram/status`, deterministic raw-token chat refusal, and Telegram
  configured-status wording when the store is corrupt. Executor Reliability
  Batch 4 now covers bounded executor journal writes, oversized journal
  compaction, support/backup partial-artifact failure returns before final
  manifests, custom partial-failure exceptions, malformed executor result
  failure handling, and scoped rollback hints. Plan Mode Reliability Batch 5
  now covers stale/cancelled/expired/wrong-thread/overwritten confirmations
  across search, Telegram, support bundle, backup, restore, cleanup, memory,
  package install, update, and uninstall action families.
- `docs/operator/P0_RELIABILITY_RECAP.md`: concise checkpoint recap for the
  five completed P0 reliability batches, trusted gates, remaining P1/P2 gaps,
  and the now-fixed daily-driver rewrite irritant.
- `python scripts/real_use_journey_smoke.py`: non-destructive installed-product
  journey audit for real web chat greeting, casual Ollama/model-status wording,
  immediate `why` follow-up context, concise RAM/system-check baseline behavior,
  and Telegram service/poller truth. This closes the gap where P0 safety gates
  passed but a real Telegram message could still get no reply because the
  optional Telegram service was stopped.
- `python scripts/normal_user_acceptance_smoke.py`: installed-product
  response-quality acceptance layer for normal-user UX. It rejects verbose
  diagnostic dumps for Telegram-style RAM checks, requires baseline
  create/compare language, verifies explicit detailed mode remains available,
  and checks that system baselines stay bounded and secret-free.
- `docs/operator/PYTEST_FAILURE_TRIAGE.md`: current classification of full
  pytest inventory failures. It records the `108 failed, 2280 passed` rerun and
  separates stale expectations, environment assumptions, duplicate gates,
  obsolete tests, flaky/soak tests, and real possible regressions. The focused
  real-regression inspection fixed pack-enable validation, explicit redacted
  secret reads, open-loop routing, and local/planning search suppression; the
  remaining behavioral replay candidate is classified as a non-release fixture
  contract until it is rewritten around current deterministic routing.
- `python scripts/prove_daily_driver_product.py`: `PASS`
- `python scripts/daily_driver_smoke.py --timeout 90`: `PASS=9 BLOCKED=0 FAIL=0`
- `python scripts/prove_pre_vm_complete.py`: `PRE_VM_COMPLETE=yes`, `BLOCKERS=0`, `UNKNOWN_AREAS=0`, `WARNINGS=7`
- `python scripts/prove_ready.py`: `READY_FOR_VM_PROOF=yes`, `RELEASE_BLOCKERS=0`, `WARNINGS=0`, `NOTES=1`
- `git status`: clean at the checkpoint

Real-use journey truth:

- P0 reliability gates prove safety boundaries, fault handling, and stale
  confirmation behavior. They do not by themselves prove every natural journey
  through the installed UI or Telegram transport.
- Telegram inbound text handling has deterministic transport coverage:
  incoming text is converted to the local `/chat` payload and the returned
  assistant text is sent back to Telegram.
- Live Telegram delivery still depends on the optional
  `personal-agent-telegram.service` being active and polling. If the service is
  stopped, Telegram messages will not be received; the web/API assistant remains
  usable and `/telegram/status` should say exactly how to start Telegram.
- A real Telegram network/chat-id send-receive proof remains an optional manual
  operator check because it requires a live bot token, network access, and a
  real Telegram chat.

New restart/browser survival lane:

- `python scripts/restart_survival_smoke.py`: automated stable API service
  restart proof. It is not a full PC reboot proof.
- `python scripts/browser_ui_survival_smoke.py`: automated installed-browser
  survival proof using Playwright and the promoted UI/API at
  `http://127.0.0.1:8765`. It covers initial page load, ordinary chat,
  concise and detailed RAM/system-check rendering, browser refresh behavior,
  API interruption and restart recovery, Plan Mode stale-confirmation safety,
  bounded long transcript behavior, special-character rendering, duplicate-send
  protection, and browser console/network diagnostics.
- `docs/operator/REBOOT_PROOF.md`: manual PC reboot checklist plus remaining
  browser checks that still require human observation.

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
  executor registry result shape, preview-only refusals, enabled cleanup plan
  cancellation, support-bundle execution, registry journal ids, stale
  confirmation rejection, and thread/session binding.
- `scripts/support_bundle_v2_smoke.py`: installed-runtime proof for the useful
  Support Bundle v2 diagnostics package: manifest, bounded summaries,
  redaction, registry result fields, and scoped rollback hint.
- `scripts/backup_v1_smoke.py`: installed-runtime proof for Backup v1:
  Plan Mode preview, Executor Registry confirmation, timestamped local backup
  directory, manifest, bounded redacted summary files, scoped rollback hint,
  and restore preview cancellation behavior.
- `scripts/restore_validator_smoke.py`: installed-runtime proof for Restore v1
  Validator. It lists recent backups, identifies the latest valid backup,
  validates Backup v1 manifests and required summaries, rejects unsafe outside
  paths, detects missing manifests, and keeps validation read-only.
- `scripts/restore_execution_smoke.py`: isolated Restore Executor v1 proof. It
  restores only fixture state, validates Backup v1, stages supported content,
  creates a pre-restore safety snapshot, applies allowlisted preferences,
  verifies live fixture state, and proves rollback on forced post-apply
  verification failure.
- `scripts/cleanup_preview_smoke.py`: installed-runtime proof for cleanup
  preview. It classifies old/oversized backup artifacts, old support bundles,
  and old runtime releases, protects current runtime/latest backup/secrets, and
  proves the installed plan can be cancelled without mutation.
- `scripts/cleanup_execution_smoke.py`: isolated cleanup execution proof. It
  deletes a generated owned support-bundle fixture through the Executor
  Registry and journals the result, while avoiding deletion of real
  daily-driver artifacts.
- `scripts/first_run_smoke.py`: isolated first-run/fresh-state proof. It starts
  a temporary API on a random loopback port with isolated HOME/XDG/state/config
  paths. It does not overwrite the promoted stable runtime and it is not the
  full clean Debian VM install proof.
- `scripts/vm_proof_smoke.py`: post-install clean-VM API smoke. It runs against
  `http://127.0.0.1:8765` after the VM install and verifies status surfaces,
  optional Telegram, unconfigured search guidance, Plan Mode previews, support
  preview, backup preview, and restore confirmation-gated behavior. It does not run
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
- `docs/operator/PYTEST_FAILURE_TRIAGE.md`: full-pytest failure map. It explains
  why full pytest is currently an inventory/triage run rather than the canonical
  release blocker, and identifies the focused real-regression tests to inspect
  first.

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
- Executor Registry v1 sits behind Plan Mode for wired lifecycle actions. It
  refuses preview-only memory/delete actions with `mutated=false`, records a
  redacted journal row, executes safe support-bundle/backup/cleanup/restore/
  update paths, and guards live daily-driver uninstall with `mutated=false`
  unless the target is an approved isolated fixture.
- Support Bundle v2 creates a temporary redacted diagnostics package with
  doctor/version/ready/state/search/Telegram/pack/journal/git summaries. It
  does not include raw logs, raw secrets, arbitrary home data, or destructive
  cleanup.
- Backup v1 creates an approved local timestamped backup directory with a
  manifest, redacted state/preferences/memory/pack/runtime/journal summaries,
  and explicit exclusions for raw secret stores, logs, arbitrary home data,
  model caches, and untrusted pack/source text.
- Restore v1 Validator can inspect Backup v1 artifacts read-only and explain
  validity, missing files, and warnings without writing or restoring anything.
- Restore Executor v1 can apply only allowlisted non-secret Backup v1
  preference values for system-resource baselines/context. It uses staging, a
  pre-restore safety snapshot, post-apply verification, and rollback on
  verification failure. It does not restore raw secrets, raw logs, arbitrary
  personal files, model caches, runtime releases, or untrusted executable/pack
  content.
- Update Executor v1 is enabled behind Plan Mode for bounded update outcomes:
  isolated staged-release promotion/rollback proof through Host Lifecycle
  Runner v1, verified live no-op, and structured live-promotion blockers when
  rollback-safe active-install handoff is unavailable.
  It rejects arbitrary repositories, branches, commits, scripts, URLs, dirty
  working trees, and target drift after preview.
- Uninstall Executor v1 is enabled behind Plan Mode for preserve-data outcomes:
  isolated fixture removal through Host Lifecycle Runner v1 with a final safety
  backup and receipt, plus a structured no-mutation blocker for live
  daily-driver uninstall. It preserves backups, memory, preferences, secret
  store, repository, model caches, and external packs by default.
- Cleanup preview identifies old or oversized Personal Agent artifacts and
  estimates recoverable space. Cleanup execution deletes only revalidated
  approved fixture/owned artifacts and remains guarded by Plan Mode.
- A fresh isolated user state can start a temporary API, return coherent
  `/ready`, `/state`, and `/version` JSON, serve the web UI entrypoint, report
  missing Telegram as optional, report unconfigured search honestly, start with
  empty memory wording, and keep package/support/backup/restore/cleanup prompts
  safe.

## Still Partial

These are not unknowns, but they are not finished:

- Installer/update/uninstall: install, promotion, bundle, and package paths have
  coverage; Update Executor v1 has isolated staged-release execution and
  rollback proof through the shared host runner plus live no-op/blocker
  behavior; Uninstall Executor v1 has isolated preserve-data fixture execution
  through the shared host runner plus live daily-driver no-mutation guard. Full
  live remote self-update, live daily-driver self-uninstall, and fresh-host
  partial-failure recovery remain partial.
- Storage/log growth: read-only installed storage estimate, cleanup preview,
  and bounded cleanup execution for approved old Personal Agent artifacts
  exist; broader rotation/enforced cleanup policy remains partial.
- Web UI robustness: static/component smoke and installed-browser automation
  exist. `browser_ui_survival_smoke.py` proves the promoted web UI through a
  real headless Chrome journey. Manual checks remain for actual PC reboot,
  visual polish across different display sizes, export-download inspection, and
  broad browser compatibility beyond installed Chrome.
- Telegram runtime behavior: optional-service semantics and status UX are
  covered; stale-lock and duplicate-poller fault injection is covered at the
  deterministic runtime/chat layer; full installed-service start/stop/restart
  failure proof remains partial.
- Secret-store behavior: missing/corrupt/decrypt-failed store status is covered
  at the deterministic CLI/status/chat layer; promoted-runtime corrupt-store
  fixture proof for support bundle and backup remains partial.
- Executor failure behavior: bounded journal writes, recent/summary reads,
  secret redaction, recursive-summary avoidance, malformed executor result
  handling, and support/backup partial-artifact failures are covered at the
  deterministic registry/unit layer. Journal rotation and broad migration of
  remaining mutators into the registry remain partial.
- Memory completion: audits, deterministic status/inspection/current-turn
  opt-out, and preview UX exist; full delete/export/redact/dedupe executors and
  richer explainability remain partial.
- Plan Mode execution: canonical plan previews, inspection, cancellation,
  thread/session binding, stale-confirmation rejection, and Executor Registry
  v1 exist. Batch 5 covers stale/cancelled/expired/wrong-thread/overwritten
  confirmations across the major action families. Cleanup now has a bounded
  executor for approved old Personal Agent artifacts. Restore now has a bounded
  allowlisted Backup v1 executor. Update now has a bounded staged-release
  fixture executor and live no-op/blocker path. Uninstall now has a bounded
  preserve-data fixture executor and live no-op blocker path. Destructive memory
  actions remain preview-only.
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
- Do not claim live daily-driver uninstall has been executed.
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
2. Use the system for real daily-driver observation and capture new failures as
   focused acceptance tests.
3. Operator lifecycle execution: Backup v1 is additive and executable; cleanup,
   restore, update, and uninstall now have bounded executor paths. Host
   Lifecycle Runner v1 provides the shared fixture/systemd handoff boundary, but
   live non-no-op self-update and live daily-driver self-uninstall remain
   guarded until active-install proof is added.
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

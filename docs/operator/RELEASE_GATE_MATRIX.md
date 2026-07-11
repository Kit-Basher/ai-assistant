# Release Gate Matrix

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.

This file separates deterministic CI-safe checks from local-runtime and optional
integration checks. The split prevents GitHub Actions from requiring a personal
machine's services while keeping the local release proof strict.

For the full inventory, overlap analysis, and recommended day-to-day command
groups, see `docs/operator/TEST_SUITE_RATIONALIZATION.md`.
For subsystem guarantees and missing fault-injection coverage, see
`docs/operator/RELIABILITY_COVERAGE_GAP_AUDIT.md`.

## CI-Safe Gates

These can run from a clean checkout without Personal Agent already installed as a
local service:

- `python scripts/release_gate.py --py-compile-only`
- `python scripts/release_smoke.py`
- `python scripts/chat_eval.py`
- `python scripts/llm_behavior_eval.py`
- `python scripts/docs_truth_smoke.py`
- `python scripts/external_pack_safety_smoke.py`
- `python scripts/backup_restore_proof.py`
- `python scripts/first_run_smoke.py`
- `python scripts/release_gate_matrix_smoke.py`
- `python -m pytest -q tests/test_release_gate.py tests/test_release_smoke.py`
- `python -m pytest -q tests/test_backup_restore_proof.py tests/test_pre_vm_complete_gate.py`

The current GitHub Actions workflow intentionally stays small and CI-safe. It
does not require SearXNG, Telegram, Ollama, a desktop session, systemd user
services, Podman, or Docker.

## Live-Runtime Gates

These are the live-runtime gates. They require the local API/runtime or
machine-specific state and should run on the release machine before the fresh VM
proof.

Commands:

- `python scripts/prove_ready.py`
- `python scripts/prove_pre_vm_complete.py`
- `python scripts/prove_daily_driver_product.py`
- `python scripts/installed_product_abuse.py`
- `python scripts/operator_lifecycle_smoke.py`
- `python scripts/memory_lifecycle_smoke.py`
- `python scripts/plan_mode_v2_smoke.py`
- `python scripts/universal_plan_mode_smoke.py`
- `python scripts/universal_plan_mode_audit.py`
- `python scripts/capability_policy_smoke.py`
- `python scripts/capability_policy_audit.py`
- `python scripts/executor_registry_smoke.py`
- `python scripts/support_bundle_v2_smoke.py`
- `python scripts/backup_v1_smoke.py`
- `python scripts/restore_validator_smoke.py`
- `python scripts/restore_execution_smoke.py`
- `python scripts/host_lifecycle_runner_smoke.py`
- `python scripts/host_lifecycle_systemd_smoke.py`
- `python scripts/active_host_enablement_smoke.py`
- `python scripts/update_execution_smoke.py`
- `python scripts/uninstall_execution_smoke.py`
- `python scripts/cleanup_preview_smoke.py`
- `python scripts/cleanup_execution_smoke.py`
- `python scripts/daily_driver_maturity_audit.py`
- `python scripts/restart_survival_smoke.py`
- `python scripts/browser_ui_survival_smoke.py`
- `python scripts/perf_smoke.py`
- `python scripts/rc1_latency_closure_smoke.py`
- `python scripts/daily_driver_smoke.py --timeout 90`
- `python scripts/prove_core_workflows.py`
- `python scripts/webui_robustness_smoke.py`
- `python -m agent doctor`

Do not treat this list as the normal edit-test loop. The default daily-driver
product loop is:

```bash
bash scripts/promote_local_stable.sh
python scripts/installed_product_abuse.py
python scripts/prove_daily_driver_product.py
python scripts/daily_driver_smoke.py --timeout 90
python scripts/daily_driver_maturity_audit.py
```

Runtime-state warnings are acceptable only when the command clearly labels them,
for example search disabled because no trusted SearXNG backend is configured.
`installed_product_abuse.py` is stricter than the ordinary smoke: it talks only
to the installed API surface, verifies promoted runtime freshness, checks
documented endpoint availability, and abuses search/Telegram/memory/Plan Mode
flows like a confused web UI user.

At `v0.2.1-search-recovery-product-pass`, the installed-product gate is the
stronger daily-driver truth gate because an earlier real UI search request found
failures that internal/mock gates had missed.

`restart_survival_smoke.py` is also a live-runtime gate. It intentionally stops
and starts `personal-agent-api.service`, verifies the promoted runtime returns,
checks managed search recovery, and confirms stale approvals do not survive the
service restart as executable approvals. It is not a full machine reboot proof.

`browser_ui_survival_smoke.py` is the installed browser/UI survival gate. It
launches a real headless browser against the promoted UI/API, verifies normal
chat, RAM-system-check rendering, refresh behavior, temporary API interruption
and restart recovery, stale Plan Mode confirmation safety, bounded long
transcripts, special-character rendering, duplicate-send protection, and
console/network diagnostics. It is a live-runtime gate, not CI-safe.

`rc1_latency_closure_smoke.py` is the installed RC1 latency distribution gate.
It samples `/ready`, `/state`, `/search/status`, direct `htop` package-state
lookup, `htop` Plan Mode preview, and pending-action lookup sequentially. It
does not enable uninstall, confirm package installation, refresh apt metadata,
or perform network package-manager work.

`operator_lifecycle_smoke.py` is the installed operator-lifecycle gate. It
proves health, broken-status, storage, repair, backup, restore, update,
cleanup, uninstall, and support-bundle prompts route through the real `/chat`
API and remain preview/confirmation-gated. It does not prove destructive
execution for cleanup, restore, update, or uninstall.

`memory_lifecycle_smoke.py` is the installed memory-lifecycle gate. It proves
memory status/inspection, current-turn opt-out, thread/global memory controls,
forget/delete/export/redact/dedupe previews, cancellation, and stale
confirmation safety through the real `/chat` API. It does not prove destructive
memory execution.

`plan_mode_v2_smoke.py` is the installed Plan Mode v2 gate. It proves canonical
plan fields, plan inspection, cancellation, preview-only executor blocking,
stale confirmation rejection after service restart, ambiguous restart
clarification, safety-bypass refusal, and thread/session confirmation binding
through the real `/chat` API.

`capability_policy_smoke.py` is the Capability Policy v1 foundation gate. It
proves registry load, schema validation, read-only allow, Plan/confirmation
requirements, local activation requirement, stale/changed target blocking,
shell package-install bypass blocking, receipt metadata, status categories, and
unmigrated action reporting. It is non-destructive and does not enable primary
uninstall.

`capability_policy_audit.py` reports registered capabilities, migrated executor
bindings, receipt/revalidation/bypass requirements, and explicit warnings for
legacy unmigrated action paths. Warnings for documented unmigrated actions are
expected in this foundation checkpoint.

`universal_plan_mode_smoke.py` is the Universal Plan Mode v1 focused proof. It
proves the Mutation Plan schema, package inspection no-mutation behavior,
package Executor Registry dispatch, direct shell package bypass blocking,
cleanup/update Universal Plan metadata, uninstall activation blocking,
cancellation, expiry, duplicate handling, changed-target rejection, receipt
metadata, and legacy-action visibility. It is non-destructive, does not install
a new package, does not enable primary uninstall, and does not uninstall the
primary installation.

`universal_plan_mode_audit.py` reports migrated mutating executors, Plan schema
compliance, receipt metadata, direct package mutation blocking, shell trusted
context requirements, and explicit warnings for legacy/unmigrated mutation
areas.

`executor_registry_smoke.py` is the installed Executor Registry v1 gate. It
proves preview-only memory delete plans do not execute; live daily-driver
uninstall confirmation is guarded with `mutated=false`; cleanup plans are
enabled but can be cancelled without mutation; the enabled support-bundle
executor returns a journal id and creates only a redacted temporary artifact;
stale confirmations still do not execute; and a different thread/session cannot
apply the pending plan.

`support_bundle_v2_smoke.py` is the installed diagnostics packaging gate. It
proves the support-bundle executor creates a bounded temporary bundle with a
manifest, redacted status summaries, executor result fields, scoped rollback
hint, and no obvious raw secret samples.

`backup_v1_smoke.py` is the installed additive-backup gate. It proves the
backup prompt is Plan Mode gated, confirmation executes through Executor
Registry v1, a timestamped local backup directory is created with a manifest
and bounded redacted summaries, obvious raw secrets are absent, rollback is
scoped to that new directory, and restore preview can be cancelled without
mutation.

`restore_validator_smoke.py` is the installed restore-validator gate. It proves
backup discovery and validation are read-only, the latest valid backup can be
identified, unsafe outside paths are rejected, malformed backups are explained,
and generic restore preview is validation-gated.

`restore_execution_smoke.py` is the isolated Restore Executor v1 gate. It
restores only fixture state, proves staging, pre-restore safety snapshot,
allowlisted preference apply, duplicate confirmation safety, and rollback on a
forced post-apply verification failure.

`update_execution_smoke.py` is the isolated Update Executor v1 gate. It proves
staged release promotion, rollback checkpoint creation, forced post-promotion
rollback, dirty-tree refusal, target-drift refusal, live no-op behavior, and
unchanged git status. It does not update the real installed daily-driver
runtime to an unknown remote commit.

`host_lifecycle_runner_smoke.py` is the shared Host Lifecycle Runner v1 fixture
gate. It proves the external runner record validation, fixture update
promotion, forced rollback, preserve-data fixture uninstall, duplicate
idempotency, partial uninstall reporting, tamper rejection, and arbitrary
command-field rejection.

`host_lifecycle_systemd_smoke.py` is the installed-host user-systemd handoff
gate. It uses fixture runtime roots and fixture unit names only. It may report
`SKIP` when the environment cannot access a user systemd bus; it must pass on
the installed Debian host before active live handoff is enabled.

`active_host_enablement_smoke.py` is the narrow active-host proof. It creates a
real alternate Personal Agent instance with separate roots, a proof-prefixed
user-systemd API service, and a non-primary loopback port. It proves real A -> B
update, HTTP reconnect, rollback, interrupted runner resume, preserve-data
uninstall, post-uninstall status, reinstall sanity, and primary-installation
protection. It is an installed-host gate, not CI-safe, and it must not target
the primary `personal-agent-api.service` or port `8765`.

`primary_update_enablement_smoke.py` is the explicit primary-update proof. It
requires `--allow-primary-update-proof` and an expected serving commit. It uses
the normal `/chat` Plan Mode path, writes a host lifecycle operation record,
launches the shared runner through user systemd, verifies primary API recovery,
forces one post-promotion rollback, and confirms primary uninstall remains
guarded. It is not part of automatic release gates because it restarts the
primary daily-driver API.

`primary_uninstall_enablement_smoke.py` is the explicit production-shaped
primary-uninstall proof. It requires
`--allow-primary-uninstall-shaped-proof` and an expected serving commit. It
creates an isolated install-shaped tree, confirms preserve-data uninstall
through `operator.uninstall.v1`, verifies final backup, receipt, preservation,
idempotency, and partial-failure reporting, and verifies the real primary
installation remains unchanged. It is not part of automatic release gates
because active primary uninstall must never be confirmed by a normal smoke.

`primary_uninstall_policy_smoke.py` is the activation-policy proof. It validates
the strict marker schema, file security, expiry, binding, integrity, local
enable/disable helpers, update-shaped marker survival, marker consumption, and
reinstall disabled default using isolated policy roots. Its installed-host check
is read-only and never creates the active marker or runs uninstall.

`v0_2_1_release_closure.py` is the sequential release-closure orchestrator for
the v0.2.1 cycle. It runs the installed-product proof set without parallel HTTP
load, captures durations, and reports aggregate PASS/WARN/FAIL/SKIP counts. It
does not enable the primary uninstall marker and does not run active primary
uninstall.

`uninstall_execution_smoke.py` is the isolated Uninstall Executor v1 gate. It
removes only generated fixture runtime/service/launcher artifacts through the
Executor Registry, verifies backups, memory, preferences, secrets, repository,
models, and external packs remain, writes a final backup and uninstall receipt,
proves idempotency, blocks live daily-driver uninstall, rejects symlink escapes,
and reports partial failures truthfully.

`cleanup_preview_smoke.py` is the installed cleanup-preview gate. It proves
cleanup prompts remain read-only, classify old/oversized backup artifacts,
old support bundle artifacts, and old runtime releases, protect the latest
valid backup and active runtime, and can be cancelled without mutation.

`cleanup_execution_smoke.py` is the cleanup execution gate. It performs actual
deletion only against an isolated generated fixture through the Executor
Registry, journals the result, and verifies the installed daily-driver cleanup
plan is enabled but not executed by the smoke.

`daily_driver_maturity_audit.py` is the recurring installed-product maturity
gate. It checks startup honesty, search/Telegram/memory honesty, operator
safety, backup/restore sanity, user-facing friction, performance drift, and
state growth. It distinguishes daily-driver irritants from release-blocking
safety failures and does not confirm enabled mutating actions.

`first_run_smoke.py` is an isolated fresh-state gate. It starts a temporary API
on a random loopback port with isolated HOME/XDG/state/config paths and verifies
first-run status, optional Telegram, unconfigured search, empty memory wording,
and Plan Mode gating. It does not overwrite the promoted stable runtime and is
not the full clean Debian VM proof.

## Optional Integration Gates

These require optional services or hardware and are not CI requirements:

- managed SearXNG live setup/query proof
- Telegram bridge smoke
- local model/Ollama live model smoke
- local model/provider matrix on the target hardware
- browser compatibility beyond the installed Chrome used by
  `browser_ui_survival_smoke.py`
- manual reboot proof from `docs/operator/REBOOT_PROOF.md`
- fresh Debian VM install proof from `docs/operator/VM_PROOF.md`:
  `python scripts/vm_proof_smoke.py --expected-commit <commit>` after install

Optional gates should report `BLOCKED` when the required service is not
configured, not `PASS`.

The fresh Debian VM proof is intentionally not listed as completed here.
`vm_proof_smoke.py` is the post-install verifier for that disposable VM; it is
not a substitute for actually running the install on a clean host.

## GitHub Actions Direction

Broader GitHub Actions coverage should be added later, after the local
PRE_VM_COMPLETE gate and the fresh Debian VM proof are stable. The next likely
CI expansion is to add `chat_eval.py`, `llm_behavior_eval.py`,
`backup_restore_proof.py`, and `external_pack_safety_smoke.py` to the existing
workflow. Do not add local-runtime gates to CI unless they are explicitly
mocked or converted to CI-safe mode.

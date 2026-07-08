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
- `python scripts/executor_registry_smoke.py`
- `python scripts/support_bundle_v2_smoke.py`
- `python scripts/backup_v1_smoke.py`
- `python scripts/restore_validator_smoke.py`
- `python scripts/cleanup_preview_smoke.py`
- `python scripts/cleanup_execution_smoke.py`
- `python scripts/daily_driver_maturity_audit.py`
- `python scripts/restart_survival_smoke.py`
- `python scripts/browser_ui_survival_smoke.py`
- `python scripts/perf_smoke.py`
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

`operator_lifecycle_smoke.py` is the installed operator-lifecycle gate. It
proves health, broken-status, storage, repair, backup, restore, update,
cleanup, uninstall, and support-bundle prompts route through the real `/chat`
API and remain preview/confirmation-gated. It does not prove destructive
execution for uninstall, cleanup, restore, or update.

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

`executor_registry_smoke.py` is the installed Executor Registry v1 gate. It
proves preview-only memory delete and uninstall plans do not execute; cleanup
plans are enabled but can be cancelled without mutation; the enabled
support-bundle executor returns a journal id and creates only a redacted
temporary artifact; stale confirmations still do not execute; and a different
thread/session cannot apply the pending plan.

`support_bundle_v2_smoke.py` is the installed diagnostics packaging gate. It
proves the support-bundle executor creates a bounded temporary bundle with a
manifest, redacted status summaries, executor result fields, scoped rollback
hint, and no obvious raw secret samples.

`backup_v1_smoke.py` is the installed additive-backup gate. It proves the
backup prompt is Plan Mode gated, confirmation executes through Executor
Registry v1, a timestamped local backup directory is created with a manifest
and bounded redacted summaries, obvious raw secrets are absent, rollback is
scoped to that new directory, and restore remains dry-run/preview-only with
`mutated=false`.

`restore_validator_smoke.py` is the installed restore-validator gate. It proves
backup discovery and validation are read-only, the latest valid backup can be
identified, unsafe outside paths are rejected, malformed backups are explained,
and generic restore confirmation still returns `executor_not_enabled`,
`mutated=false`.

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

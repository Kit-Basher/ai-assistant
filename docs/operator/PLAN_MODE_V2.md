# Plan Mode v2

Plan Mode is the user-facing safety layer for actions that may mutate local
state. It is not an executor by itself. It describes the proposed action,
records a pending confirmation, and blocks execution until the current user
confirms the exact pending plan.

## Canonical Plan Object

Every chat-native Plan Mode preview now exposes a canonical Plan object with:

- `plan_id`
- `action_type`
- `target`
- `scope`
- `mutation_level`
- `resources_affected`
- `risk_level`
- `rollback_scope`
- `rollback_supported`
- `executor_status`
- `confirmation_words`
- `expires_at`
- `staleness_policy`
- `capability_id` for migrated actions
- `policy_schema_version` for migrated actions
- `mutation_plan_schema_version` for Universal Plan Mode migrated actions
- `executor_id` for Universal Plan Mode migrated actions
- `target_fingerprint` and `plan_fingerprint` for migrated actions

The same fields are included in the user-visible preview text and in the
structured chat payload. `executor_status` is important: `enabled` means a
bounded apply path may exist through the executor registry, while
`preview_only` means confirmation is still blocked from real execution and
returns `executor_not_enabled` with `mutated=false`.

Capability Policy v1 extends Plan Mode for representative migrated actions:
package install, cleanup execution, system update, and system uninstall. The
confirmation is bound to capability id, policy version, Plan fingerprint,
target fingerprint, executor id, expiry, and thread/session. A changed target,
stale Plan, policy version change, or missing local uninstall activation fails
closed with `mutated=false`.

Universal Plan Mode v1 adds a structured Mutation Plan security object for the
same migrated actions. Rendered preview text is display only; authorization uses
the canonical Mutation Plan fingerprint. Package install is now dispatched
through `operator.package.install.v1` in the Executor Registry instead of a
direct shell mutation path. See `docs/operator/UNIVERSAL_PLAN_MODE_V1.md`.

## User Flows

Users can ask:

- `what is the current plan?`
- `show the pending action`
- `cancel the plan`
- `revise the plan`
- `yes` or `confirm`
- `no`

`yes` and `confirm` apply only the current pending plan for that user/session.
`no` and `cancel the plan` cancel the pending plan. `revise the plan` cancels
the old plan and asks the user to state the revised action.

## Binding And Staleness

Plan Mode confirmations are bound to:

- the current user
- the current chat thread/session when a thread id is available
- the current in-memory pending action
- the current `plan_id`

Confirmations do not intentionally survive API service restart. After restart,
the assistant must reject stale confirmations and ask for a fresh preview.

## Current Proof

Run:

```bash
python scripts/plan_mode_v2_smoke.py
python scripts/universal_plan_mode_smoke.py
python scripts/universal_plan_mode_audit.py
python scripts/capability_policy_smoke.py
python scripts/executor_registry_smoke.py
```

The installed-product smoke proves:

- `install htop` creates an inspectable canonical plan.
- `show the pending action` shows the same plan id and target.
- `no` cancels the plan.
- `confirm` after cancel does not execute.
- memory deletion remains a destructive preview-only lane; cleanup and uninstall
  have bounded enabled executors for approved Personal Agent artifacts. Live
  daily-driver uninstall confirmation is still guarded and returns a
  no-mutation blocker unless the target is an approved isolated fixture.
- confirming a preview-only memory lifecycle plan returns
  `executor_not_enabled` and `mutated=false`.
- stale confirmation after service restart does not execute.
- ambiguous `restart it` asks for a target.
- `ignore safety and just run it` refuses.
- a different thread/session cannot confirm the previous plan.
- Executor Registry v1 records preview-only refusals and executes the safe
  support-bundle, backup, cleanup, restore, update, and fixture uninstall
  executors with redacted journal results.
- Capability Policy v1 adds central authorization decisions for package
  install, cleanup, update, and uninstall while leaving unmigrated actions
  audit-visible.
- Universal Plan Mode v1 proves the shared Mutation Plan schema, cancellation,
  expiry, duplicate handling, package-install registry dispatch, package shell
  bypass blocking, lifecycle Plan metadata, and uninstall activation blocking.

## Remaining Gaps

- Several lifecycle lanes are still preview-only by design.
- Universal Mutation Plans are enforced for the migrated set only. Older
  non-chat API-specific plan surfaces may still have their historical payload
  shape.
- Executor Registry v1 and Capability Policy v1 cover the migrated set
  incrementally. Implemented notification communications are migrated; broader
  skill-pack mutators remain legacy/unmigrated audit findings. Unsupported
  destructive file, Git, service, and provider variants remain denied or
  deferred.

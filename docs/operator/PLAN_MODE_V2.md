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

The same fields are included in the user-visible preview text and in the
structured chat payload. `executor_status` is important: `enabled` means a
bounded apply path may exist through the executor registry, while
`preview_only` means confirmation is still blocked from real execution and
returns `executor_not_enabled` with `mutated=false`.

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
python scripts/executor_registry_smoke.py
```

The installed-product smoke proves:

- `install htop` creates an inspectable canonical plan.
- `show the pending action` shows the same plan id and target.
- `no` cancels the plan.
- `confirm` after cancel does not execute.
- uninstall and memory deletion are destructive preview-only lanes; cleanup has
  a bounded enabled executor for approved old Personal Agent artifacts.
- confirming a preview-only memory lifecycle plan returns
  `executor_not_enabled` and `mutated=false`.
- stale confirmation after service restart does not execute.
- ambiguous `restart it` asks for a target.
- `ignore safety and just run it` refuses.
- a different thread/session cannot confirm the previous plan.
- Executor Registry v1 records preview-only refusals and executes the safe
  support-bundle, backup, and cleanup executors with redacted journal results.

## Remaining Gaps

- Several lifecycle lanes are still preview-only by design.
- Canonical Plan objects are now exposed for chat-native previews, but older
  non-chat API-specific plan surfaces may still have their historical payload
  shape.
- Executor Registry v1 exists, but only a small set of safe/additive actions is
  wired through it. Most existing managed-service, external-pack, model, and
  lifecycle mutators still use their established bounded apply paths.

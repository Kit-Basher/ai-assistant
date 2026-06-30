# Executor Registry v1

Executor Registry v1 is the narrow apply boundary behind Plan Mode v2. Plan
Mode still owns preview, user confirmation, expiry, and thread/session binding.
The registry owns executor lookup, refusal for non-enabled executors, canonical
result fields, and an append-only redacted mutation journal.

## Result Schema

Every registry result includes:

- `ok`
- `mutated`
- `executor_id`
- `plan_id`
- `action_type`
- `target`
- `started_at`
- `finished_at`
- `resources_touched`
- `journal_id`
- `rollback_available`
- `rollback_hint`
- `error_code`
- `user_message`

## Enforcement

- `executor_status=preview_only` returns `executor_not_enabled` and
  `mutated=false`.
- `executor_status=unavailable` returns `executor_unavailable` and
  `mutated=false`.
- `executor_status=enabled` can run only when the confirmed Plan Mode action
  reaches the registry with the exact matching `plan_id`.
- High-risk/destructive actions stay behind Plan Mode confirmation and are not
  enabled unless a bounded executor exists.
- Registry refusals are journaled too, so attempted preview-only execution is
  auditable.

## Journal

The registry writes append-only JSONL records under runtime state:

`executor_registry_journal.jsonl`

The journal redacts token, secret, password, bearer, API key, and confirmation
token fields. It should not contain raw secrets, raw support logs, or private
unredacted payloads.

## Enabled Executors

Current enabled executor:

- `operator.support_bundle` via `operator.support_bundle.v1`

This creates a small redacted support summary in a temporary directory. It is
additive and returns a `journal_id`. Rollback is limited to removing the newly
created temporary support bundle directory.

## Preview-Only Lanes

These remain preview-only in v1:

- memory delete/export/redact/dedupe/control executors
- uninstall
- cleanup
- restore/update/repair lifecycle executors unless a separate bounded executor
  is already used by an existing managed-service path

Confirming one of these plans must return `executor_not_enabled`,
`mutated=false`, and a registry `journal_id`.

## Proof

Run:

```bash
python scripts/executor_registry_smoke.py
```

The smoke talks to the installed `/chat` API and proves:

- preview-only memory delete, uninstall, and cleanup cannot execute
- support bundle has an enabled executor and returns a journal id
- support bundle creates only a redacted temporary artifact
- stale confirmation after API restart does not execute
- unrelated thread/session cannot execute the pending plan
- obvious secret markers are not present in executor results


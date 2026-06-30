# Support Bundle v2

Support Bundle v2 is the first useful additive executor behind Executor
Registry v1. It is still Plan Mode gated: the assistant previews the action,
the user confirms, then the executor creates a new temporary diagnostics bundle.

It is not destructive and it does not repair anything.

## Included Files

The bundle directory contains fixed, bounded JSON files:

- `manifest.json`
- `doctor_summary.json`
- `version.json`
- `ready.json`
- `state_summary.json`
- `search_status.json`
- `telegram_status.json`
- `packs_state_summary.json`
- `executor_registry_journal_summary.json`
- `readiness_proof_summary.json`
- `git_runtime_freshness.json`
- `support_summary.json`

`manifest.json` includes:

- `created_at`
- `runtime_commit`
- `checkout_commit`
- `runtime_instance`
- `included_files`
- `redaction_policy`
- `bundle_schema_version`

## Redaction Policy

The packager redacts or summarizes:

- Telegram tokens
- API keys
- passwords
- bearer tokens
- server secret keys
- confirmation tokens
- raw secret-file values
- raw logs
- broad private paths

The bundle should not include raw logs, raw secret-store contents, arbitrary home
directory data, browser data, downloaded pages, or external pack source text.

## Rollback Scope

Support bundle creation is additive. Rollback is limited to removing only the
newly created temporary bundle directory named in the executor result.

## Proof

Run:

```bash
python scripts/support_bundle_v2_smoke.py
```

The installed-product smoke proves:

- preview uses Plan Mode v2
- confirmation executes through Executor Registry v1
- bundle artifact and manifest exist
- expected summary files exist
- obvious raw secret samples are absent from bundle files
- executor result includes `mutated=true`, `resources_touched`, `journal_id`,
  and a scoped rollback hint
- git status remains unchanged


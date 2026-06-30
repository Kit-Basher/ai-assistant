# Backup v1

Backup v1 is the first additive backup executor behind Plan Mode v2 and
Executor Registry v1. It creates a new local backup directory with redacted
summaries only. It does not restore, overwrite, delete, or stop services.

## Flow

User-facing prompt:

```text
back up the assistant
```

The assistant must show a Plan Mode v2 preview. After explicit confirmation,
the executor writes a new timestamped directory under the approved Personal
Agent state backup path:

```text
~/.local/share/personal-agent/backups/
```

The executor result includes `mutated=true`, `resources_touched`, `journal_id`,
and a rollback hint scoped to removing only the new backup directory.

## Included Files

The backup directory contains fixed, bounded JSON files:

- `manifest.json`
- `state_database_summary.json`
- `preferences_summary.json`
- `memory_anchors_summary.json`
- `pack_metadata_summary.json`
- `runtime_config_summary.json`
- `executor_registry_journal_summary.json`
- `diagnostics_summary.json`
- `support_bundle_style_summary.json`
- `backup_summary.json`

`manifest.json` includes:

- `backup_schema_version`
- `created_at`
- `runtime_commit`
- `runtime_instance`
- `included_files`
- `excluded_files`
- `redaction/encryption policy`
- `restore_status`

## Redaction And Exclusions

Backup v1 stores redacted summaries only. It excludes:

- raw secret-store files
- raw Telegram tokens, provider API keys, passwords, and bearer tokens
- raw logs and full support bundles
- arbitrary home directory data
- model caches/downloads and GGUF/model artifacts
- raw external pack archives, `SKILL.md`, `AGENTS.md`, and untrusted source text
- browser caches, downloaded pages, and search result page contents

No encryption is applied in v1 because raw secret material is not included.
Treat the backup directory as local-sensitive.

## Restore Status

Restore remains dry-run/preview-only:

- `restore_status= dry_run_only`
- `live_restore= restore_not_enabled`

The prompt `restore from backup` must show a preview-only Plan Mode response.
Confirming that preview must return `executor_not_enabled` and `mutated=false`.
Live restore must not overwrite local state in Backup v1.

## Proof

Run:

```bash
python scripts/backup_v1_smoke.py
```

The installed-product smoke proves:

- backup preview uses Plan Mode v2
- confirmation executes through Executor Registry v1
- backup artifact and manifest exist
- expected bounded summary files exist
- obvious raw secret samples are absent from backup files
- executor journal recorded the action
- rollback hint is scoped to the new backup path only
- restore remains dry-run/preview-only and does not mutate
- git status remains unchanged

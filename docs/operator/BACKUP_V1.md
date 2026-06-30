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

Backup v1 is summary/export based. It does not copy runtime directories,
previous backups, virtual environments, caches, raw logs, support bundle
directories, model artifacts, or arbitrary home directory data.

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
- `size_caps`
- `file_sizes`
- `total_size_bytes`

Current caps:

- total backup JSON size: 2 MiB
- per-file JSON size: 256 KiB
- executor journal rows summarized: 8 recent entries

## Redaction And Exclusions

Backup v1 stores redacted summaries only. It excludes:

- raw secret-store files
- raw Telegram tokens, provider API keys, passwords, and bearer tokens
- raw logs and full support bundles
- arbitrary home directory data
- model caches/downloads and GGUF/model artifacts
- raw external pack archives, `SKILL.md`, `AGENTS.md`, and untrusted source text
- browser caches, downloaded pages, and search result page contents
- `~/.local/share/personal-agent/backups` recursively
- runtime releases and `.venv` directories

The executor records only fixed summary files. Executor journal rows are reduced
to bounded metadata such as `journal_id`, `action_type`, `executor_id`,
`mutated`, `error_code`, timestamps, and resource counts. It must not embed
full prior executor actions or prior backup contents.

Old or oversized backup artifacts are not deleted automatically. Use
`clean old backup files` or `python scripts/cleanup_preview_smoke.py` to verify
that cleanup preview identifies candidate artifacts and protects the latest
valid backup. Cleanup deletion is not implemented in Backup v1.

If a cap is exceeded, Backup v1 returns a failed executor result with
`mutated=false`, does not write a final manifest, records any partial JSON files
already created, and gives a rollback hint scoped only to that new partial
backup directory.

No encryption is applied in v1 because raw secret material is not included.
Treat the backup directory as local-sensitive.

## Restore Status

Restore remains dry-run/preview-only:

- `restore_status= dry_run_only`
- `live_restore= restore_not_enabled`

The prompt `restore from backup` must show a preview-only Plan Mode response.
Confirming that preview must return `executor_not_enabled` and `mutated=false`.
Live restore must not overwrite local state in Backup v1.

Restore v1 Validator can inspect Backup v1 artifacts without restoring them:

```bash
python scripts/restore_validator_smoke.py
```

User-facing prompts such as `show my backups` and
`validate this backup: <path>` list and validate backup artifacts read-only.
See `docs/operator/RESTORE_VALIDATOR.md`.

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
- backup artifact size stays below the documented caps
- obvious raw secret samples are absent from backup files
- executor journal recorded the action
- rollback hint is scoped to the new backup path only
- restore remains dry-run/preview-only and does not mutate
- git status remains unchanged

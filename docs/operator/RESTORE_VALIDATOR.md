# Restore v1 Validator

Restore v1 Validator inspects Backup v1 artifacts and explains whether they are
safe-looking Backup v1 summaries. It does not restore, overwrite, delete, start
services, create restore directories, or mutate live state.

Live restore remains disabled.

## User Flow

Read-only backup discovery:

```text
show my backups
```

Read-only validation:

```text
validate this backup: ~/.local/share/personal-agent/backups/personal-agent-backup-...
check this backup before restore: <path>
inspect backup <path>
```

Generic restore remains preview-only:

```text
restore from backup
```

Confirming the generic restore preview returns `executor_not_enabled`,
`mutated=false`.

## Approved Path Rules

The validator reads only explicit backup directories under approved locations:

- `~/.local/share/personal-agent/backups`
- `/tmp` for temporary validation fixtures and operator-supplied scratch
  artifacts

Paths outside approved locations are rejected before reading. The validator does
not scan arbitrary home directories.

## Validation Rules

For a supplied backup directory, the validator checks:

- `manifest.json` exists and is JSON
- `backup_schema_version == backup.v1`
- required summary files exist
- included file names are local file names, not paths
- per-file and total JSON sizes are within Backup v1 caps
- `restore_status == dry_run_only`
- `live_restore == restore_not_enabled`
- `excluded_files` documents raw secrets, raw logs, and arbitrary home data as
  excluded
- included JSON files do not contain obvious token/API-key/password markers

The response reports valid/invalid status, schema version, creation time,
runtime commit, included files, missing files, warnings, validation errors, and
the live-restore-disabled state.

## Proof

Run:

```bash
python scripts/restore_validator_smoke.py
```

The installed-product smoke proves:

- `show my backups` lists recent backup artifacts and identifies the latest
  valid backup
- the latest valid Backup v1 artifact validates
- unsafe outside paths are rejected before reading
- a malformed temp fixture with no manifest is detected
- generic restore remains preview-only
- restore confirmation returns `executor_not_enabled`, `mutated=false`
- git status remains unchanged


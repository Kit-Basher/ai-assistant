# Restore v1 Validator

Restore v1 Validator inspects Backup v1 artifacts and explains whether they are
safe-looking Backup v1 summaries. Validation itself does not restore,
overwrite, delete, start services, create restore directories, or mutate live
state.

Restore Executor v1 is now enabled for a narrow supported subset of Backup v1
data. It restores only allowlisted non-secret preference values used for
system-resource baselines/context. It does not restore raw secrets, raw logs,
arbitrary files, model caches, runtime releases, or untrusted executable/pack
content.

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

Restore remains Plan Mode gated:

```text
restore from backup
restore from backup: <path>
```

The preview validates a Backup v1 artifact, lists supported/excluded
categories, and states that a pre-restore safety snapshot will be created before
mutation. Confirmation executes through `operator.restore.v1`.

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
- Backup v1 manifest restore metadata remains present for compatibility with
  older dry-run-only artifacts
- `excluded_files` documents raw secrets, raw logs, and arbitrary home data as
  excluded
- included JSON files do not contain obvious token/API-key/password markers

The response reports valid/invalid status, schema version, creation time,
runtime commit, included files, missing files, warnings, validation errors, and
that validation is read-only.

## Restore Executor v1 Boundary

Restored:

- `system_resource_baseline_v1`
- `system_resource_baseline_context_v1`

Validated but ignored or preserved:

- runtime metadata
- diagnostics summaries
- pack metadata summaries
- executor journal summaries
- memory anchor summaries that do not include supported restore preferences

Explicitly unsupported:

- raw secret store files and tokens
- provider/API keys/passwords/bearer tokens
- raw logs and full support bundle contents
- arbitrary home-directory files
- browser data
- model caches/downloads
- runtime release bundles
- untrusted pack source text or executable content
- unknown Backup v1 schema extensions

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
- restore preview is validation-gated and enabled through Plan Mode
- restore preview can be cancelled without mutation
- git status remains unchanged

Run the isolated execution proof for actual restore mutation and rollback:

```bash
python scripts/restore_execution_smoke.py
```

# Backup and Restore

The supported portable recovery artifact is
`personal-agent.portable-backup.v2`. It is created from an explicit allowlist,
validated before extraction, and restored through a staging directory into an
empty or byte-identical explicit target. A recursive copy of the whole state
root is not the portable product contract because it includes runtimes,
quarantine, logs, caches, and other ephemeral material.

## Manual operator snapshot (advanced)

If an operator deliberately takes a stopped-machine snapshot rather than the
portable workflow, preserve these paths as sensitive material:

- `~/.config/personal-agent`
- `~/.local/share/personal-agent`
- `~/.config/systemd/user`

Also preserve the source checkout or a release artifact, but do not treat a
repo-local `memory/agent.db` or `llm_registry.json` as live canonical state.
The canonical database and registry are:

- `~/.local/share/personal-agent/agent.db`
- `~/.local/share/personal-agent/llm_registry.json`

The canonical confirmation ledger is:

- `~/.local/share/personal-agent/confirmation_transactions.sqlite3`

Operator permissions remain configuration rather than mutable runtime state:

- `~/.config/personal-agent/permissions.json`

Internal-writer receipts use bounded `*.internal-writer.sqlite3` files under
the same canonical state root. A whole-state snapshot includes them, but is
host-sensitive and is not the normal-user cross-host recovery format.

Recovery directories and archives are evidence until a human explicitly
retires them. Install, doctor, backup, and cleanup flows must not delete or
overwrite preserved recovery material outside their own versioned runtime and
temporary staging roots.

## Read-Only Recovery Audit

Before changing a recovered installation, run:

```bash
python scripts/recovery_install_audit.py \
  --expect-artifact ~/personal-agent-old-feb-2026-07-20 \
  --expect-artifact ~/personal-agent-state-recovery \
  --expect-artifact ~/personal-agent-migration-test \
  --expect-artifact ~/personal-agent-recovery-2026-07-20.tar.gz \
  --expect-artifact ~/personal-agent/.venv-before-canonical-move
```

This command is read-only. It verifies canonical path presence, SQLite
integrity, schema version, registry presence, expected recovery artifacts, and
reports repo-local legacy state as a preserve-and-migrate warning.

If you want logs for support, include:

- `~/.local/share/personal-agent/agent.jsonl`

If you want continuity and pack history preserved, back up the whole state
directory rather than only individual files.

## What Can Usually Be Rebuilt

Usually safe to rebuild from the repo and runtime defaults:

- discovery caches
- release smoke outputs
- temporary doctor bundles
- web UI build output
- other derived cache files under the state directory

Do not assume pack state, continuity state, or operator config can be rebuilt
without loss. Back those up first.

## Restore Steps

For Backup v1 artifacts created by the assistant, prefer the chat flow:

1. Ask `show my backups`.
2. Ask `validate this backup: <path>` if you need a specific artifact checked.
3. Ask `restore from backup: <path>`.
4. Confirm only after the Plan Mode preview shows a valid Backup v1 artifact,
   supported restore categories, excluded categories, and a pre-restore safety
   snapshot.

Restore Executor v1 restores only supported non-secret Backup v1 state. It does
not restore raw secrets, logs, arbitrary files, model caches, runtime releases,
or executable pack source. It does preserve authorization history: Backup v1
uses SQLite online snapshots for the confirmation ledger and internal-writer
receipts, and restore merges them append-only. Historical reserved operations
restore as failed; historical executing operations restore as indeterminate,
so uncertain work cannot be retried automatically.

Do not copy only a SQLite main file while the service is running. Either back
up the whole stopped state directory or use Backup v1, whose SQLite backup API
includes committed WAL state in a standalone database. WAL/SHM files are not
independent backup artifacts.

Executor Authorization Migration v1 binds restore execution to
`restore.execute`. Confirmation uses Universal Mutation Plan metadata, central
capability authorization, trusted invocation context, a pre-restore safety
snapshot, and receipt metadata. Direct restore helper calls without trusted
context are blocked before locks, staging directories, snapshots, or state
changes are created.

For older full-path manual backups:

1. Stop the user services.
2. Restore the saved paths above to the same locations.
3. Run `python -m agent doctor --fix`.
4. Restart `personal-agent-api.service`.
5. Verify:
   - `python -m agent status`
   - `curl -sS http://127.0.0.1:8765/ready`
   - `curl -sS http://127.0.0.1:8765/state`
   - `curl -sS http://127.0.0.1:8765/packs/state`

## Full Reset

A full reset is acceptable only when you intentionally want to discard local
state.

In that case:

- back up the paths above first
- remove the local state/config/service files
- reinstall or restore the checkout
- run `python -m agent setup`
- run `python -m agent doctor`

## Confirming a Restore

A restore is successful when:

- the service starts cleanly
- `/ready` is sane
- `/state` is sane
- `/packs/state` matches the expected pack inventory
- `python -m agent version` reports the expected build/version
- `python scripts/recovery_install_audit.py` reports no failures
- the running unit resolves to the intended stable or dev code root and only
  one API service is enabled for daily use
- `/ready` reports SAFE MODE unless Controlled Mode was explicitly confirmed
- `/telegram/status` reports disabled unless Telegram was explicitly enabled

## Pre-VM Proof

Run:

```bash
python scripts/backup_restore_proof.py
```

This proof is intentionally bounded and does not touch live
`~/.local/share/personal-agent`, live `~/.config/personal-agent`, or user
services. It creates representative Personal Agent state under a temporary
directory, creates an allowlisted v2 archive, validates every member and
digest, performs a dry-run restore, restores only into another temporary
directory, and checks expected config, state database, search config, pack
state, and the Personal Agent service unit.

The proof also checks:

- corrupt backup archives fail with `corrupt_backup`
- strict version mismatch fails with `version_mismatch`
- identical repeated restore is idempotent and conflicting targets fail closed
- machine-bound secret files are excluded and Setup re-entry is explicit
- no service is started, stopped, enabled, or restarted
- live runtime state is not mutated

## Version Mismatch

Strict restore validation refuses backups whose manifest `app_version` does not
match the expected Personal Agent version. Operators should upgrade or downgrade
the runtime intentionally, then rerun validation. Do not silently restore a
version-mismatched backup into live state.

## Secret Handling

Portable backup v2 excludes the machine-bound encrypted-file secret store and
never exports keyring contents. On a recovered host, the user re-enters optional
provider and Telegram secrets through Setup. This is intentional: the existing
file cipher derives from host identity and copying it is neither genuinely
portable nor a sound secret-recovery design. Treat the remaining archive as
local-sensitive because it contains user configuration, memory, task, and pack
state.

## Uninstall Safety Backup

Uninstall Executor v1 creates a final bounded Backup v1-style safety artifact
before fixture uninstall removes any runtime/service files. The uninstall final
backup is stored outside the removable runtime path and referenced by the
uninstall receipt.

The default uninstall mode preserves user data in place. Full user-data purge is
not part of Uninstall Executor v1.

Provider/model authorization adds no standalone durable database. It uses the
canonical confirmation transaction database and executor journal already
covered by Backup/Restore v1. Plaintext secrets are never copied into Plans,
receipts, support artifacts, or backup summaries.

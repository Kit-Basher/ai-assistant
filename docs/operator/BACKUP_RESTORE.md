# Backup and Restore

The safest backup is the full canonical runtime state plus the user-service
configuration.

## Minimal Backup Set

Back up these paths before upgrades, risky changes, or recovery work:

- `~/.config/personal-agent`
- `~/.local/share/personal-agent`
- `~/.config/systemd/user`

Also preserve the source checkout or a release artifact, but do not treat a
repo-local `memory/agent.db` or `llm_registry.json` as live canonical state.
The canonical database and registry are:

- `~/.local/share/personal-agent/agent.db`
- `~/.local/share/personal-agent/llm_registry.json`

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
or executable pack source.

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
directory, creates a backup archive, validates the archive, performs a dry-run
restore, restores only into another temporary directory, and then checks that
expected config, state, memory DB, search config, systemd unit, and secret-store
files are present.

The proof also checks:

- corrupt backup archives fail with `corrupt_backup`
- strict version mismatch fails with `version_mismatch`
- dry-run output redacts sensitive file details and never prints secret values
- no service is started, stopped, enabled, or restarted
- live runtime state is not mutated

## Version Mismatch

Strict restore validation refuses backups whose manifest `app_version` does not
match the expected Personal Agent version. Operators should upgrade or downgrade
the runtime intentionally, then rerun validation. Do not silently restore a
version-mismatched backup into live state.

## Secret Handling

Backups include the encrypted/local secret-store file so same-machine restore can
preserve provider and Telegram configuration. Proof and dry-run output must not
print raw secret values. Treat backup archives as sensitive local artifacts.

## Uninstall Safety Backup

Uninstall Executor v1 creates a final bounded Backup v1-style safety artifact
before fixture uninstall removes any runtime/service files. The uninstall final
backup is stored outside the removable runtime path and referenced by the
uninstall receipt.

The default uninstall mode preserves user data in place. Full user-data purge is
not part of Uninstall Executor v1.

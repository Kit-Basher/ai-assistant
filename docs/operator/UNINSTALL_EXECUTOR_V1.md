# Uninstall Executor v1

Uninstall Executor v1 is the bounded preserve-data uninstall path behind Plan
Mode and Executor Registry.

It is not a purge feature and not an arbitrary software uninstaller. Active
primary uninstall is wired only through the same Host Lifecycle Runner path and
requires the reviewed primary enablement marker. Automated proof still does not
uninstall the active daily-driver runtime.

## Current Checkpoint

- Checkpoint before this batch: `v0.2.1-update-executor-v1`
- Commit before this batch: `3ceb95d`
- Executor id: `operator.uninstall.v1`
- Action type: `operator.uninstall`
- Mode: `preserve_data`

## User Data Preserved By Default

The default uninstall preserves:

- Backup v1 artifacts
- pre-restore and pre-update safety snapshots
- memory database and memory summaries
- user preferences and application state
- secret store and user-provided credentials
- repository checkout
- model caches
- external skill packs and user-authored pack content
- support bundles and logs needed for troubleshooting
- final uninstall safety backup
- uninstall receipt

No raw secrets are copied into ordinary chat output, receipts, or journal
records.

## Removable Resources

Uninstall v1 may remove only exact, allowlisted Personal Agent-owned resources:

- promoted runtime release directories in an approved runtime root
- stable/current runtime symlink
- generated install metadata
- generated Personal Agent API and Telegram user service unit files
- generated launchers, desktop entries, and icon files
- generated helper metadata inside an approved fixture/install root

The executor rejects arbitrary user paths, repository paths, state/config roots,
backup roots, secret stores, model directories, external pack roots, mount
points, symlink escapes, path traversal, unknown services, unknown containers,
and resources whose ownership is not proven.

## Plan And Confirmation

`uninstall the assistant` creates a canonical Plan Mode preview. The preview
states that the app and user services would be removed, data is preserved, a
final safety backup is required, a receipt will be written, and the chat may
disconnect when uninstall begins.

Confirmation is bound to the current plan, thread/session, expiry, and target
snapshot. Stale, cancelled, expired, wrong-thread, overwritten, duplicate, or
changed-target confirmations do not launch a second uninstall.

## Final Backup

Before destructive fixture removal, the executor creates a bounded final backup
under the approved backup root. It includes:

- Backup v1-compatible manifest
- uninstall target snapshot
- preserved-resource inventory
- preferences/runtime summary
- install metadata summary

It excludes raw secrets and arbitrary home-directory data. If backup creation or
manifest validation fails, uninstall aborts before mutation.

## Host Runner And Receipt

The executor prepares a Host Lifecycle Runner v1 operation record and runs the
shared trusted runner against the validated fixture. The runner removes only
approved resources and writes a durable JSON receipt outside the removable
runtime path.

The receipt records:

- operation id and mode
- start and finish timestamps
- previous commit/version metadata
- removed resources
- skipped or failed resources
- preserved resources
- final backup path
- reinstall guidance
- rollback limitations
- verification result

The receipt is bounded and redacted.

## Primary Daily-Driver Enablement

Primary preserve-data uninstall is conditionally enabled. Confirmation can hand
off to the Host Lifecycle Runner only when the canonical primary inventory
validates and the local activation-policy marker is valid:

`~/.local/share/personal-agent/host_lifecycle/primary_uninstall_enabled.json`

The marker is schema v1, preserve-data only, time-limited, installation-bound,
integrity checked, owned by the current user, mode `0600`, and stored under a
host-lifecycle directory no broader than `0700`. Raw file existence is not
authorization.

Without a valid marker, a confirmed live install plan returns a structured
no-mutation result:

- `error_code=uninstall_live_execution_not_enabled`
- `mutated=false`
- no services are stopped
- no runtime files are removed
- no state, secrets, or backups are deleted
- no final backup, uninstall operation record, or runner handoff is created

Preview carries the marker fingerprint when enabled. Confirmation revalidates
the marker and blocks if it was removed, replaced, expired, loosened, or rebound
after preview. Accepted primary uninstall operation records contain the marker
fingerprint and preserve-data policy summary, not raw marker JSON. The marker
is consumed before runner handoff so reinstall defaults back to disabled.

The production-shaped proof exercises the same operation shape against an
isolated install-shaped tree and verifies the real primary installation remains
unchanged. Normal installed-product safety gates still do not confirm active
primary uninstall.

## Verification

The isolated proof verifies:

- fixture runtime, current symlink, service units, launcher, icon, and install
  metadata are removed
- memory/state, secret store, existing backups, repository, model cache, and
  external packs remain
- unrelated service and container fixtures remain untouched
- final backup exists and has a manifest
- uninstall receipt exists
- repeated helper execution is idempotent
- live uninstall is blocked with `mutated=false`
- changed target snapshots are blocked
- symlink escapes are rejected
- forced partial failure produces a truthful partial receipt/result

Run:

```bash
python scripts/uninstall_execution_smoke.py
python scripts/host_lifecycle_runner_smoke.py
python scripts/host_lifecycle_systemd_smoke.py
python scripts/primary_uninstall_enablement_smoke.py --allow-primary-uninstall-shaped-proof --expected-commit <current-serving-commit>
```

## Recovery

Uninstall is not automatically reversible after runtime files and services are
removed. The recovery path is:

1. Reinstall Personal Agent from the trusted preserved repository or release
   source.
2. Ask the assistant to restore supported state from the final uninstall backup.
3. Reuse the preserved secret store and configuration.
4. Verify `/ready`, `/state`, `/version`, and normal chat.

Do not claim rollback succeeded unless a reinstall and restore were actually
performed and verified.

## Known Limits

- Purge/delete-user-data mode is not implemented.
- Arbitrary path/service/container uninstall is rejected.
- Live daily-driver self-removal is guarded and not executed by the installed
  proof lanes.
- Host Lifecycle Runner v1 proves fixture resource selection, sequencing,
  receipt, and preservation through a shared runner. The active-host proof uses
  a real alternate installed instance and removes its proof-prefixed user
  service and runtime while preserving alternate state, backups, dummy secret
  store, repository, model cache, external packs, and receipt.
- Live daily-driver self-removal remains guarded. Removing that guard should be
  a separate small change that wires the same validated active-host operation
  record fields to the primary profile and re-runs the full installed-host
  proof.

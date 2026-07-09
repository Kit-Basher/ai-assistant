# Primary Update Enablement

Primary Update Guard Wiring v1 enables daily-driver non-no-op update handoff
through the same Host Lifecycle Runner v1 path proven by the alternate-host
enablement proof.

## Checkpoint

- Checkpoint before this batch: `v0.2.1-active-host-enablement-proof`
- Commit before this batch: `95beaac`
- Proof script: `scripts/primary_update_enablement_smoke.py`

## What Changed

`operator.update.v1` now has a primary staged-release handoff mode. It is used
only when an internal, bounded primary-update proof request supplies:

- the primary state root: `~/.local/share/personal-agent`
- the primary runtime root: `~/.local/share/personal-agent/runtime`
- `personal-agent-api.service`
- `http://127.0.0.1:8765`
- a proof marker under the primary Personal Agent state root
- an already-built staged release source
- a fixed approved commit

The executor writes the same tamper-detected host lifecycle operation record
used by the alternate-host proof and launches the shared host runner through
user systemd. The chat request returns an `in_progress` result; completion or
rollback truth comes from the host-runner state and receipt.

## Proof Target

The primary proof does not update to an unknown remote commit. It builds a new
runtime release from the currently serving trusted release while preserving the
true Git commit. The operation is non-no-op at the runtime-release level because
`runtime/current` switches to a new release directory.

The rollback half of the proof injects a proof-only verification failure through
the immutable operation record after promotion. The host runner then restores
the previous serving release and verifies the API before reporting rollback.

## Safety Boundaries

The primary update path still requires:

- Plan Mode confirmation
- trusted local checkout identity in the preview
- fixed target commit
- no target drift after preview
- primary path, service, and port allowlists
- host-runner operation record hashing
- user-systemd handoff
- rollback checkpoint and receipt
- serving `/ready` and `/version` verification

The primary proof script requires:

```bash
python scripts/primary_update_enablement_smoke.py \
  --allow-primary-update-proof \
  --expected-commit <current-serving-commit>
```

It is intentionally not run by `prove_ready.py`.

## Uninstall Guard

Primary daily-driver uninstall remains guarded. This batch does not confirm or
enable primary uninstall. Installed-product proof should continue to see:

```text
uninstall_live_execution_not_enabled
```

## Limits

- This is not a reboot-safe update proof.
- This does not prove updating to an arbitrary remote commit.
- This does not enable purge or primary uninstall.
- The proof may briefly restart the primary Personal Agent API.

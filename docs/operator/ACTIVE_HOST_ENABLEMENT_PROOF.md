# Active Host Enablement Proof

This proof exercises Host Lifecycle Runner v1 against a real alternate Personal
Agent installation on the current Debian host. It is deliberately separate from
the primary daily-driver installation.

## Current Checkpoint

- Checkpoint before this batch: `v0.2.1-host-lifecycle-runner-v1`
- Commit before this batch: `f571462`
- Proof script: `scripts/active_host_enablement_smoke.py`
- Profile name: `active-host-proof`
- Alternate service prefix: `personal-agent-active-host-proof-`

## Isolation Boundary

The proof creates a disposable alternate profile under a temporary root. It uses:

- a separate runtime root, release root, and `runtime/current` symlink
- a separate state root, DB, log, backups, models, external-pack directory, and
  lifecycle operation root
- a dynamically selected non-primary loopback API port
- a real user-systemd API service named
  `personal-agent-active-host-proof-api.service`
- generated proof-only launch/service files

The proof refuses to use the primary API port `8765` or the primary service
name `personal-agent-api.service`. It snapshots the primary installation before
and after the proof and fails if the primary API, runtime commit, service state,
or git status changes unexpectedly.

## What It Proves

`scripts/active_host_enablement_smoke.py` proves:

- alternate Release A starts as a real Personal Agent API under user systemd
- alternate `/ready` and `/version` respond over HTTP
- alternate deterministic chat works
- Plan Mode update and uninstall previews render through the alternate API
- Host Lifecycle Runner promotes alternate Release A to Release B through a
  real user-systemd runner unit
- the alternate API restarts and `/version` reports Release B
- forced post-promotion verification failure rolls back to Release A and
  verifies the serving runtime
- an interrupted update after promotion resumes without duplicate promotion
- preserve-data uninstall stops the alternate API, removes alternate runtime
  and proof service files, and preserves state, backups, dummy secret store,
  repository, model cache, external packs, and unrelated proof files
- post-uninstall status works from the host runner and receipt without the API
- alternate reinstall sanity starts Release A again and sees the final uninstall
  backup and preserved dummy secret store
- cleanup removes alternate proof resources and leaves the primary installation
  healthy

## Guard Decision

The proof validates the host runner against a real alternate installed instance,
and Primary Update Guard Wiring v1 uses that same runner path for the primary
daily-driver update proof:

- active daily-driver non-no-op self-update is wired through
  `operator.update.v1` -> Host Lifecycle Runner v1 for approved primary staged
  releases
- active daily-driver preserve-data uninstall remains guarded

Primary update enablement is documented in
`docs/operator/PRIMARY_UPDATE_ENABLEMENT.md`. No primary uninstall was confirmed
or executed during this proof.

## Run

```bash
python scripts/active_host_enablement_smoke.py
```

The command requires a working user systemd session. It creates and removes only
proof-prefixed services and temporary proof roots.

## Limits

- This is not a VM install proof.
- This is not a reboot-safe proof.
- Browser reconnect is represented by live HTTP/API reconnect in this narrow
  installed-host proof; the full primary browser UI gate remains
  `scripts/browser_ui_survival_smoke.py`.
- The alternate service runs in proof import mode with explicit runtime metadata
  so Release A and Release B are distinguishable without editing production
  source files.

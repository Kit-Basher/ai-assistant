# Update Executor v1

Update Executor v1 is the bounded Executor Registry path for Personal Agent
updates. It is behind Plan Mode v2 and accepts only canonical internal update
plans.

## Trust Boundary

Allowed:

- the configured local Personal Agent checkout
- the approved local branch/channel reported in the preview
- repository-owned release metadata and promotion inputs
- internally generated staged-release fixture inputs used by proof scripts
- a verified live no-op when the runtime commit already matches the approved
  checkout commit

Rejected:

- arbitrary repository URLs
- arbitrary local paths
- user-supplied branches, commits, shell commands, install scripts, archives,
  container images, or download URLs
- dirty working trees
- targets that change after preview
- runtime/current targets that change after preview
- live promotion without a rollback-safe staged-release handoff

## Plan Mode Fields

The update preview includes:

- trusted source
- trusted remote identity if available
- approved channel
- current runtime commit
- checkout/target commit
- working-tree cleanliness
- dirty-file blocker summary when present
- affected runtime/service resources
- rollback scope
- confirmation expiry and Executor Registry status

Preview is read-only. It does not fetch, build, promote, restart services, or
discard local changes.

## Executor Behavior

`operator.update.v1` supports three outcomes:

- `fixture_staged_release`: real staged release promotion in an isolated proof
  root through Host Lifecycle Runner v1, with rollback checkpoint, operation
  state, journal record, verification, and forced-failure rollback proof.
- `live_noop`: verifies the installed runtime is already at the approved target
  commit and returns `mutated=false`.
- guarded live update blocker: refuses non-no-op live promotion until the
  rollback-safe active-install handoff is available.

Dirty working trees are blocked:

```text
I can’t update yet because the Personal Agent repository has uncommitted changes. I left them untouched.
```

## Rollback

The isolated staged-release runner creates a pre-update checkpoint recording:

- previous release path
- previous runtime commit
- `runtime/current` symlink
- target commit
- operation id and creation time

If post-promotion verification fails, the executor switches `runtime/current`
back to the previous release and verifies the previous commit before saying it
rolled back.

## Proof

Run:

```bash
python scripts/update_execution_smoke.py
python scripts/host_lifecycle_runner_smoke.py
```

The smoke uses only temporary fixture roots. It proves:

- staged release A -> B promotion
- journal record and rollback checkpoint creation
- forced post-promotion verification failure rolls back to A
- dirty-tree refusal returns `mutated=false`
- target drift after preview is blocked
- live no-op returns `mutated=false`
- git status is unchanged

## Limits

Update v1 does not yet perform a live remote fetch or promote an unknown live
commit from chat. The installed product may preview an update and block when the
checkout is dirty or when live promotion is not rollback-safe. Uninstall remains
guarded for active daily-driver removal.

Host Lifecycle Runner v1 now proves the shared external runner boundary for
fixture promotion and rollback. `active_host_enablement_smoke.py` also proves
the runner against a real alternate installed instance: Release A serves HTTP,
the host runner promotes it to Release B through user systemd, rollback restores
Release A after a forced verification failure, and interrupted promotion can be
resumed.

Active daily-driver non-no-op update remains guarded. Removing that guard should
be a separate small change that wires the same validated active-host operation
record fields to the primary profile and re-runs the full installed-host proof.

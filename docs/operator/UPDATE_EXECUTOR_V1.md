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
- a primary staged release generated from the currently trusted serving runtime
  and handed to Host Lifecycle Runner v1 through user systemd
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

`operator.update.v1` supports four outcomes:

- `fixture_staged_release`: real staged release promotion in an isolated proof
  root through Host Lifecycle Runner v1, with rollback checkpoint, operation
  state, journal record, verification, and forced-failure rollback proof.
- `primary_staged_release`: primary daily-driver runtime promotion through the
  shared Host Lifecycle Runner v1. It accepts only the primary runtime root,
  `personal-agent-api.service`, `http://127.0.0.1:8765`, a proof marker under
  the Personal Agent state root, and an already-built staged release. It
  returns `in_progress` at handoff; completion or rollback truth comes from the
  host lifecycle receipt.
- `live_noop`: verifies the installed runtime is already at the approved target
  commit and returns `mutated=false`.
- guarded live update blocker: refuses requests that do not carry an approved
  internal staged-release target.

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
python scripts/primary_update_enablement_smoke.py \
  --allow-primary-update-proof \
  --expected-commit <current-serving-commit>
```

The smoke uses only temporary fixture roots. It proves:

- staged release A -> B promotion
- journal record and rollback checkpoint creation
- forced post-promotion verification failure rolls back to A
- dirty-tree refusal returns `mutated=false`
- target drift after preview is blocked
- live no-op returns `mutated=false`
- primary update handoff returns `in_progress`
- primary runtime switches to a new release path built from the trusted serving
  commit
- forced primary post-promotion failure rolls back and verifies the serving API
- git status is unchanged

## Limits

Update v1 does not yet perform a live remote fetch or promote an unknown live
commit from chat. The installed product blocks dirty-checkout and target-drift
updates. Uninstall remains guarded for active daily-driver removal.

Host Lifecycle Runner v1 now proves the shared external runner boundary for
fixture promotion and rollback. `active_host_enablement_smoke.py` also proves
the runner against a real alternate installed instance: Release A serves HTTP,
the host runner promotes it to Release B through user systemd, rollback restores
Release A after a forced verification failure, and interrupted promotion can be
resumed.

Primary daily-driver non-no-op update is wired to the same validated host-runner
operation path. The explicit primary proof is `scripts/primary_update_enablement_smoke.py`
and is not part of the automatic non-destructive release gates.

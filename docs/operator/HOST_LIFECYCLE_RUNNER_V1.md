# Host Lifecycle Runner v1

Host Lifecycle Runner v1 is the shared trusted runner for lifecycle work that
must happen outside the active Personal Agent API process.

It is intentionally narrow. It supports only internally generated operation
records for:

- `update`
- `uninstall`
- read/resume/verify-style status commands through the same operation record
  validation path

It does not accept shell command strings, arbitrary paths, arbitrary service
names, repository URLs, branches, commits, scripts, or environment payloads from
chat.

## Current Checkpoint

- Checkpoint before this batch: `v0.2.1-uninstall-executor-v1-clean`
- Commit before this batch: `6690b4e0f3948cad9a222ac10f0dc1dc0ae6c430`
- Library: `agent/host_lifecycle.py`
- CLI: `scripts/host_lifecycle_runner.py`
- Operation schema: `host_lifecycle_operation.v1`
- Runner version: `host_lifecycle_runner.v1`

## Trust Boundary

The runner loads one validated operation record. The record must:

- live under an approved operation root, currently `/tmp` for strict fixtures or
  Personal Agent-owned local state roots for installed operations
- not be a symlink
- use a supported schema and runner version
- name a supported operation type
- have a matching immutable approved-payload hash
- avoid arbitrary `command`, `shell`, or `argv` fields
- include only validated internal identifiers and allowlisted paths

The runner rejects tampered records, unknown operation types, unsupported
schemas, expired records, command fields, path escapes, and missing fixture
markers.

## Operation Record

The shared record includes:

- schema version and runner version
- operation id and operation type
- plan id
- created/expiry time when available
- fixture/live mode
- current stage and state path
- receipt path
- update target fields when applicable
- uninstall target snapshot and final backup path when applicable
- approved payload hash

Records do not contain raw chat text, secrets, raw logs, auth headers, or shell
commands. Fixture operation records may contain exact temporary paths because the
runner needs them to mutate the fixture; these records are not rendered directly
to users.

## Stage Model

The runner uses the shared stage vocabulary:

- `created`
- `validated`
- `checkpoint_ready`
- `staging`
- `promoting`
- `removing_runtime`
- `verifying`
- `rolling_back`
- `completed`
- `rollback_completed`
- `rollback_failed`
- `partial`
- `failed_before_mutation`
- `failed_after_mutation`

Stage state is written atomically to the operation state file.

## Update Flow

For strict fixture updates, the runner:

1. validates the operation record
2. verifies the current runtime symlink and staged release
3. creates a rollback checkpoint
4. copies the staged release into the approved release root
5. atomically switches `runtime/current`
6. verifies the target commit
7. writes a receipt

If verification fails, it switches `runtime/current` back to the previous
release and verifies the previous commit before reporting rollback success.

Active daily-driver non-no-op update is still guarded. Update v1 supports live
no-op and fixture proof; it does not promote an unknown live remote commit from
chat.

## Uninstall Flow

For strict fixture uninstall, the runner:

1. validates the operation record
2. verifies the final safety backup
3. removes only exact allowlisted runtime/service/launcher resources
4. verifies preserved state, secrets, backups, repository, models, and external
   packs
5. writes a durable receipt

Active daily-driver uninstall is still guarded. The runner proves the shared
control path and systemd handoff with fixture resources only.

## Systemd Handoff

`scripts/host_lifecycle_systemd_smoke.py` proves the runner can be launched by
a user transient systemd unit against fixture roots:

```bash
python scripts/host_lifecycle_systemd_smoke.py
```

The proof uses a fixture unit name and fixture runtime roots. It does not stop,
disable, or remove the real `personal-agent-api.service`.

If the current environment cannot access a user systemd bus, the script reports
`SKIP` with the exact reason. It must pass on the installed Debian host before
active live handoff is enabled.

## Status

The runner CLI supports a read-only status operation:

```bash
python scripts/host_lifecycle_runner.py status --operation-record <record>
```

Status validates the operation record, then reads the operation state and
receipt paths. It returns `mutated=false` and does not resume or rerun the
operation.

## Proof

Run:

```bash
python scripts/host_lifecycle_runner_smoke.py
python scripts/host_lifecycle_systemd_smoke.py
python scripts/update_execution_smoke.py
python scripts/uninstall_execution_smoke.py
```

The runner smoke proves:

- fixture update promotion
- forced update rollback
- fixture preserve-data uninstall
- duplicate uninstall idempotency
- partial uninstall reporting
- tampered record rejection
- arbitrary command field rejection

## Remaining Limits

- Active daily-driver non-no-op self-update remains guarded.
- Active daily-driver self-uninstall remains guarded.
- Reboot-safe completion is not claimed.
- Host-systemd proof is fixture-only; it proves handoff mechanics, not removal
  of the active assistant.

# v0.2.1 Release Closure

Checkpoint:

- Tag: `v0.2.1-primary-uninstall-activation-policy-v1`
- Commit: `226c1497db85549de38417c22488fa859f8b2d41`

## Completed Capability Set

The v0.2.1 lifecycle roadmap is complete through:

- Host Lifecycle Runner v1.
- Alternate installed-host update proof.
- Primary non-no-op update.
- Verified primary rollback.
- Primary preserve-data uninstall wiring.
- Production-shaped uninstall proof.
- Strict local primary uninstall activation policy.

Primary uninstall remains preserve-data only. Purge is unsupported. No
destructive uninstall of the active primary installation has been performed.

## Uninstall Activation Model

Status:

```bash
python scripts/primary_uninstall_policy.py status
python scripts/primary_uninstall_policy.py diagnose
```

Enable, local operator only:

```bash
python scripts/primary_uninstall_policy.py enable \
  --acknowledge-primary-uninstall-capability \
  --expires-in-days 30
```

Disable:

```bash
python scripts/primary_uninstall_policy.py disable
```

Optional host-policy permission repair, without enabling uninstall:

```bash
python scripts/primary_uninstall_policy.py repair-permissions \
  --acknowledge-host-policy-repair
```

The assistant may explain these commands and report status. It must not run the
enable command or create the marker from chat.

## Host Policy Finding

The current installed host has
`~/.local/share/personal-agent/host_lifecycle` owned by the current user but
mode `0775`. The strict uninstall policy expects `0700`. This is a real policy
mismatch for an activation marker directory, not a need for broader access: the
host runner and API run as the same user and can read operation records under
`0700`.

The policy repair command is explicit and narrow. It does not create the marker,
does not enable uninstall, and does not change services or runtime state.

## Latency Warnings

Known warning classes:

- `/ready` latency from optional readiness details and cold runtime state.
- `/search/status` latency from repeated SearXNG reachability checks.
- install-preview latency from deterministic package/status route setup.
- transient HTTP timeouts under concurrent proof load.

Release closure keeps proofs sequential. `/ready` now omits optional model-watch
and recent Telegram details from the fast path and includes `timing_ms`.
`/search/status` uses a short bounded cache and reports cache/timing metadata.

## Release Proof

Sequential release closure runner:

```bash
python scripts/v0_2_1_release_closure.py --expected-commit 226c149
```

This runner does not enable the primary uninstall marker and does not run active
primary uninstall.

Core supporting gates:

```bash
bash scripts/promote_local_stable.sh
python scripts/primary_uninstall_policy.py status
python scripts/primary_uninstall_policy_smoke.py
python scripts/primary_uninstall_enablement_smoke.py --allow-primary-uninstall-shaped-proof --expected-commit 226c149
python scripts/primary_update_enablement_smoke.py --allow-primary-update-proof --expected-commit 226c149
python scripts/prove_ready.py
python scripts/docs_truth_smoke.py
python scripts/release_gate_matrix_smoke.py
```

## Recovery Commands

Runtime status:

```bash
curl -fsS http://127.0.0.1:8765/ready
curl -fsS http://127.0.0.1:8765/version
python -m agent doctor
```

Service restart:

```bash
systemctl --user restart personal-agent-api.service
```

Backup/restore:

```bash
python scripts/backup_v1_smoke.py
python scripts/restore_validator_smoke.py
python scripts/restore_execution_smoke.py
```

Update recovery/status:

```bash
python scripts/host_lifecycle_runner.py status --operation-record <operation.json>
systemctl --user restart personal-agent-api.service
```

Uninstall policy/status:

```bash
python scripts/primary_uninstall_policy.py status
python scripts/primary_uninstall_policy.py disable
```

## Release Decision

Choose `v0.2.1-rc1` if host-policy repair remains unapplied on the live host,
latency warnings remain recurrent, or the sequential release-closure proof has
warnings that need observation.

Choose final `v0.2.1` only after functional gates pass, host policy is
understood or repaired, latency warnings are bounded, Debian recovery docs match
the installed host, and the working tree is clean after commit.

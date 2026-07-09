# Primary Preserve-Data Uninstall Enablement

This checkpoint wires primary preserve-data uninstall to the same
`operator.uninstall.v1` and Host Lifecycle Runner v1 path proven by isolated and
alternate-host uninstall proofs.

It does not add purge support, arbitrary target removal, or a proof that
uninstalls the active daily-driver installation.

## Boundary

Primary uninstall is preserve-data only.

Removable resources are limited to exact Personal Agent-owned installation
artifacts:

- promoted runtime release directories under the approved runtime root
- `runtime/current`
- generated API and Telegram user-service units/drop-ins
- generated launcher links, desktop entry, and icon
- generated install metadata

Preserved by default:

- repository checkout and Git history
- state database, preferences, memory, and system baselines
- secret store and Telegram/search credentials
- Backup v1 artifacts, restore snapshots, and update checkpoints outside
  removable runtime roots
- final uninstall backup and durable receipt
- external skill packs, user-authored content, model caches, support bundles,
  unrelated logs, unrelated services, unrelated containers, and arbitrary user
  files

Purge remains unsupported.

## Enablement

The active primary path uses the same operation record schema, target snapshot
hash, final backup logic, receipt format, Host Lifecycle Runner, removal logic,
and preserved-resource verification as the alternate proof path.

Primary execution additionally requires a local enablement marker:

`~/.local/share/personal-agent/host_lifecycle/primary_uninstall_enabled.json`

Without that marker, confirmation returns `uninstall_live_execution_not_enabled`
with `mutated=false`.

## Proof

`scripts/primary_uninstall_enablement_smoke.py` is the explicit
production-shaped proof. It requires:

```bash
python scripts/primary_uninstall_enablement_smoke.py \
  --allow-primary-uninstall-shaped-proof \
  --expected-commit <current-serving-commit>
```

The proof creates an isolated install-shaped Personal Agent tree with runtime
releases, generated service files, launcher metadata, preserved state, backups,
secrets, models, packs, and unrelated files. It confirms uninstall through the
Executor Registry path, runs the shared host lifecycle runner, verifies removed
and preserved resources, checks the durable receipt, proves duplicate execution
is idempotent, forces a truthful partial failure, and verifies the real primary
installation stays healthy and unchanged.

The proof never confirms uninstall against the active primary service, primary
runtime root, primary state root, or port `8765`.

## Recovery

Uninstall is not automatically reversible after runtime files and services are
removed. Recovery is:

1. Reinstall from the preserved trusted repository or release source.
2. Validate the final uninstall Backup v1 artifact.
3. Restore supported state through Restore v1.
4. Reuse the preserved secret store/configuration.
5. Verify `/ready`, `/state`, `/version`, and normal chat.

## Limits

- No purge mode.
- No arbitrary paths, services, containers, repositories, or shell commands.
- No claim of reboot-safe uninstall completion beyond the existing host runner
  state/receipt behavior.
- No destructive active-primary uninstall proof was run.

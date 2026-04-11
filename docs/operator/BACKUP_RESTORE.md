# Backup and Restore

The safest backup is the full canonical runtime state plus the user-service
configuration.

## Minimal Backup Set

Back up these paths before upgrades, risky changes, or recovery work:

- `~/.config/personal-agent`
- `~/.local/share/personal-agent`
- `~/.config/systemd/user`

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

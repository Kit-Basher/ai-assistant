# Local Split Audit

This note records the current split between the repo-backed dev install and the stable bundled install.

## Install Stories

- Repo-backed dev install: `bash scripts/install_local.sh --desktop-launcher`
- Stable bundled install: the bundled `install.sh` from a release bundle, or the packaged release path

## What Each Path Installs

- `scripts/install_local.sh`
  - creates or reuses `.venv` in the checkout
  - installs the package in editable mode
  - installs the dev service
  - optionally installs a dev desktop launcher

- `install.sh`
  - installs the release payload into `~/.local/share/personal-agent/runtime`
  - wires the stable user service
  - installs the stable launcher and desktop entry
  - keeps mutable state under `~/.local/share/personal-agent`

## Service and Launcher Names

- Stable service: `personal-agent-api.service`
- Dev service: `personal-agent-api-dev.service`
- Stable launcher: `personal-agent-webui`
- Dev launcher: `personal-agent-webui-dev`

## Shared State

The stable and dev installs intentionally share the same canonical mutable state:

- `~/.local/share/personal-agent`
- `~/.config/personal-agent`

That covers registry, secrets, db, logs, permissions, and audit files.

## Runtime Paths That Still Depend on Code Location

- `agent/config.py` still resolves `skills_path` from the installed code root when `AGENT_SKILLS_PATH` is not set.
- `agent/version.py` still uses the installed code root to read version metadata.
- The launcher script uses the service name and web UI URL it was installed with.

## Current Overlap Points

- Both install stories still run `python -m agent doctor --fix`.
- Both use the same HTTP UI shape and the same canonical state tree.
- Before this split, both install stories could target the same service name and port.

## Migration Risks

- If a machine already has legacy repo-local db/log files, the first install or doctor run may migrate them into the canonical state tree.
- If a dev launcher or service is active while the stable service is also active, the two copies can compete unless they use different service names and ports.
- If a user keeps the repo checkout but deletes the stable runtime tree, the launcher should still point to the stable service only, not the checkout.

## Safe Rule

- Daily-driver use should go through the stable bundled install.
- Checkout installs are for development only.
- Canonical user state stays in `~/.local/share/personal-agent` unless an explicit migration says otherwise.

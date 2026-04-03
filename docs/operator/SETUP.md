# Setup

Canonical product/runtime source: [`PRODUCT_RUNTIME_SPEC.md`](/home/c/personal-agent/PRODUCT_RUNTIME_SPEC.md).

Canonical first-run command:

- `python -m agent setup`
- JSON: `python -m agent setup --json`
- Dry-run: `python -m agent setup --dry-run`

Canonical install/runtime path:

- repo checkout: `~/personal-agent`
- mutable state: `~/.local/share/personal-agent`
- operator config/policy: `~/.config/personal-agent`
- user service symlink: `~/.config/systemd/user/personal-agent-api.service`
- supported service-management story: `systemctl --user ...`

Legacy `install.sh`, `uninstall.sh`, and `doctor.sh` are retired and fail
closed on purpose. Do not use the old root/system-service path.

Canonical packaging/build path:

- repo install/update: `pip install -e .`
- release artifact build: `python scripts/build_dist.py --outdir dist --clean`
- Debian/system packaging is not supported for this release

## Install

1. Place the repo at `~/personal-agent`.
2. Create the virtualenv and install the package:
   - `cd ~/personal-agent`
   - `python3 -m venv .venv`
   - `. .venv/bin/activate`
   - `pip install -e .`
3. Install the user service:
   - `mkdir -p ~/.config/systemd/user`
   - `ln -sf ~/personal-agent/systemd/personal-agent-api.service ~/.config/systemd/user/personal-agent-api.service`
   - `systemctl --user daemon-reload`
4. Enable reboot-safe user services:
   - `loginctl enable-linger "$USER"`
5. Start the runtime:
   - `systemctl --user enable --now personal-agent-api.service`
6. Complete first run:
   - `python -m agent setup`
   - `python -m agent doctor`
7. If you are validating the install or preparing a release:
   - `python scripts/release_smoke.py`

## First Run

1. Run `python -m agent setup`.
2. Follow exactly one `Next action`.
3. Re-run `python -m agent status` until runtime is stable.
4. Use native UI as primary setup/recovery surface; Telegram mirrors runtime setup state when enabled.
5. Telegram is optional and off by default. Use `python -m agent telegram_status` to inspect it and `python -m agent telegram_enable` to turn it on.

## Setup Complete

Setup is complete when onboarding state is `READY` and:

- `python -m agent status` reports healthy runtime mode.
- Telegram `help/setup/status/health/doctor/memory` follows the same canonical contract text as CLI outputs.
- Telegram `memory` / `what are we doing?` / `resume` routes to continuity summary (not setup fallback).

## Onboarding States

- `NOT_STARTED`
- `TOKEN_MISSING`
- `LLM_MISSING`
- `SERVICES_DOWN`
- `READY`
- `DEGRADED`

## Recovery States

- `TELEGRAM_DOWN`
- `API_DOWN`
- `TOKEN_INVALID`
- `LLM_UNAVAILABLE`
- `LOCK_CONFLICT`
- `DEGRADED_READ_ONLY`
- `UNKNOWN_FAILURE`

## Short Recovery Paths

- token missing:
  - (only when Telegram is enabled)
  - `python -m agent.secrets set telegram:bot_token`
  - `python -m agent telegram_enable`
- telegram down:
  - (only when Telegram is enabled)
  - `python -m agent telegram_status`
  - `python -m agent telegram_enable`
- api down:
  - `systemctl --user restart personal-agent-api.service`
- llm unavailable:
  - `python -m agent setup`
  - `python -m agent doctor`

## Upgrade

1. `cd ~/personal-agent`
2. `git pull --ff-only`
3. `. .venv/bin/activate`
4. `pip install -e .`
5. `python -m agent doctor --fix`
6. `systemctl --user daemon-reload`
7. `systemctl --user restart personal-agent-api.service`
8. `python -m agent status`
9. If this upgrade is a release candidate or a risky recovery, run:
   - `python scripts/release_smoke.py`

`python -m agent doctor --fix` is the canonical safe upgrade helper. It creates
missing local directories and copies legacy repo-local runtime storage into the
canonical state directory when needed.

## Recovery / Reset

1. Run `python -m agent setup`.
2. Follow the single `Next action`.
3. Run `python -m agent doctor --collect-diagnostics` if you need one redacted
   support bundle before making changes.
4. Run `python -m agent doctor --fix` if directories, drop-ins, or legacy
   runtime storage need repair.
5. Restart the runtime:
   - `systemctl --user restart personal-agent-api.service`
6. Verify:
   - `python -m agent status`
7. If this is a release-candidate validation path, run:
   - `python scripts/release_smoke.py`

Manual backup/export is supported by stopping the user services and copying:

- `~/.config/personal-agent`
- `~/.local/share/personal-agent`
- `~/.config/systemd/user`

Manual restore/import is supported by restoring those paths, then running:

- `python -m agent doctor --fix`
- `systemctl --user restart personal-agent-api.service`
- `python -m agent status`

Failed-upgrade or corrupt-state recovery:

- collect diagnostics first: `python -m agent doctor --collect-diagnostics`
- move the broken file aside instead of editing it in place
- restore from backup if available
- run `python -m agent doctor --fix`

## Reinstall / Uninstall

Reinstall over an existing install is supported through the same canonical
upgrade path above. Existing user-local state is preserved unless you remove it
explicitly.

Uninstall:

1. `systemctl --user disable --now personal-agent-api.service`
2. `rm -f ~/.config/systemd/user/personal-agent-api.service`
3. `systemctl --user daemon-reload`
4. Optional cleanup:
   - `rm -rf ~/.config/personal-agent`
   - `rm -rf ~/.local/share/personal-agent`
   - leave them in place if you want to preserve local state/secrets for a
     later reinstall

## Telegram Adapter Control

- status:
  - `python -m agent telegram_status`
- enable:
  - `python -m agent telegram_enable`
- disable:
  - `python -m agent telegram_disable`

`telegram_status` reports whether Telegram is intentionally disabled, running, misconfigured, or blocked by a stale/live poll lock. If a stale lock exists, `telegram_enable` clears it safely before restarting the service.

## If Setup Fails

1. Run `python -m agent doctor`.
2. Follow the single `Next action`.
3. Re-run `python -m agent setup`.

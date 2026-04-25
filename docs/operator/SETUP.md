# Setup

Canonical product/runtime source: [`PRODUCT_RUNTIME_SPEC.md`](/home/c/personal-agent/PRODUCT_RUNTIME_SPEC.md).

Canonical first-run command:

- `python -m agent setup`
- JSON: `python -m agent setup --json`
- Dry-run: `python -m agent setup --dry-run`

Canonical install/runtime path:

- stable runtime install root: `~/.local/share/personal-agent/runtime`
- stable mutable state: `~/.local/share/personal-agent`
- stable operator config/policy: `~/.config/personal-agent`
- stable user service: `personal-agent-api.service`
- supported stable service-management story: `systemctl --user ...`
- stable desktop launcher:
  - `~/.local/share/applications/personal-agent.desktop`
  - `~/.local/bin/personal-agent-webui`

Developer checkout path:

- repo checkout: `~/personal-agent`
- dev user service: `personal-agent-api-dev.service`
- dev desktop launcher, if installed: `Personal Agent (Dev)`

Distributed-install runtime root:

- installed runtime payload: `~/.local/share/personal-agent/runtime`
- versioned releases: `~/.local/share/personal-agent/runtime/releases/<version>`
- current release symlink: `~/.local/share/personal-agent/runtime/current`
- stable launcher command: `~/.local/share/personal-agent/bin/personal-agent-webui`
- stable uninstall command: `~/.local/share/personal-agent/bin/personal-agent-uninstall`

Recommended install path for a new user:

- run the bundled `install.sh` from the release bundle or packaged release

That one command:

- installs the stable runtime under `~/.local/share/personal-agent/runtime`
- wires the stable `personal-agent-api.service`
- installs the stable desktop launcher
- keeps the checkout independent from the running app

If you are developing in the checkout, use:

- `bash scripts/install_local.sh --desktop-launcher`

That checkout install:

- checks Python, systemd user support, and desktop-launcher prerequisites
- creates or reuses `.venv`
- installs the package in editable mode
- runs `python -m agent doctor --fix`
- installs or refreshes the dev user service
- optionally installs the dev desktop launcher

If you do not want the dev desktop launcher, omit `--desktop-launcher`.
If you only want to validate web UI build dependencies, add `--check-webui-build`.

If you downloaded a release bundle, use the bundled installer instead:

- extract the bundle
- run the bundled `install.sh`
- the bundled installer uses the stable runtime root under
  `~/.local/share/personal-agent/runtime`

If you are the maintainer on the local checkout and want to create the real
stable runtime now, use:

- `bash scripts/promote_local_stable.sh`

That helper:

- builds a release bundle from the checkout
- installs that bundle into `~/.local/share/personal-agent/runtime/releases/<version>`
- creates `~/.local/share/personal-agent/runtime/current`
- installs the stable launcher and `personal-agent-api.service`
- keeps the checkout separate from the runtime

If you installed the Debian package instead:

- package-managed runtime root: `/usr/lib/personal-agent/runtime`
- launcher command: `personal-agent-webui`
- uninstaller helper: `personal-agent-uninstall`
- user-local state: `~/.local/share/personal-agent`

First launch completes the user-service registration if needed, then opens the
same browser UI.

Debian package lifecycle:

- upgrade/reinstall: rerun `sudo apt install ./dist/personal-agent_<version>_amd64.deb`
- remove package-owned files: `sudo apt remove personal-agent`
- remove package and user state: run `personal-agent-uninstall --remove-state`
  first if you want a full local reset

Legacy root-level `install.sh`, `uninstall.sh`, and `doctor.sh` are retired and
fail closed on purpose. Do not use the old root/system-service path.

Canonical packaging/build path:

- repo install/update: `pip install -e .`
- release artifact build: `python scripts/build_dist.py --outdir dist --clean`
- Debian package build: `bash scripts/build_deb.sh --clean`
- Debian/Ubuntu install from a built package:
  - `sudo apt install ./dist/personal-agent_<version>_amd64.deb`
- canonical release gate: `python scripts/release_gate.py`
- fast pre-check before the heavier gate: `python scripts/release_smoke.py`
- release, rollback, backup, and support-boundary guidance:
  - `docs/operator/RELEASE.md`
  - `docs/operator/OPERATIONS.md`
  - `docs/operator/BACKUP_RESTORE.md`
  - `docs/operator/KNOWN_LIMITS.md`

## Install

1. Place the repo at `~/personal-agent`.
2. If you want the stable daily-driver, run:
   - run the bundled `install.sh` from the release bundle or packaged release
3. If you want the editable checkout install for development, run:
   - `bash scripts/install_local.sh --desktop-launcher`
4. Open the UI from the desktop menu or browse to `http://127.0.0.1:8765/`.
5. Confirm the active copy:
   - `python -m agent split_status`
6. If you are validating the install or preparing a release:
   - `python scripts/release_gate.py`

Manual fallback if you need to inspect the pieces individually:

- create the virtualenv and install the checkout package:
  - `cd ~/personal-agent`
  - `python3 -m venv .venv`
  - `. .venv/bin/activate`
  - `pip install -e .`
- install the dev user service:
  - `mkdir -p ~/.config/systemd/user`
  - `ln -sf ~/personal-agent/systemd/personal-agent-api-dev.service ~/.config/systemd/user/personal-agent-api-dev.service`
  - `systemctl --user daemon-reload`
- enable reboot-safe user services:
  - `loginctl enable-linger "$USER"`
- start the dev runtime:
  - `systemctl --user enable --now personal-agent-api-dev.service`
- complete first run:
  - `python -m agent setup`
  - `python -m agent doctor`

Bundle update / uninstall basics:

- reinstall the same bundle version: rerun the bundled `install.sh`
- upgrade: extract the newer bundle and rerun the bundled `install.sh`
- live split smoke: `python scripts/split_smoke.py`
- uninstall while preserving state:
  - run the bundled `uninstall.sh`
- uninstall and remove local state:
  - run the bundled `uninstall.sh --remove-state`

## Optional Desktop Launcher

If you want a normal desktop app entry for the checkout/dev install, install
the user-local launcher:

- `bash scripts/install_desktop_launcher.sh`

That installs:

- a desktop menu entry at `~/.local/share/applications/personal-agent-dev.desktop`
- a launcher command at `~/.local/bin/personal-agent-webui-dev`
- a user-local icon at `~/.local/share/icons/hicolor/scalable/apps/personal-agent.svg`

What it does when clicked:

- registers, starts, or wakes `personal-agent-api-dev.service` if needed
- waits briefly for `GET /ready`
- opens the local web UI in your default browser

If it fails, use:

- `http://127.0.0.1:18765/`
- `systemctl --user status personal-agent-api-dev.service`

## First Run

1. Run `python -m agent setup`.
2. Follow exactly one `Next action`.
3. Re-run `python -m agent status` until runtime is stable.
4. Use native UI as primary setup/recovery surface; Telegram mirrors runtime setup state when enabled.
5. Telegram is optional and off by default. Use `python -m agent telegram_status` to inspect it and `python -m agent telegram_enable` to turn it on.
6. If the web UI offers first-run onboarding, you can answer with short freeform intent words like `programming`, `linux help`, or `writing stories`, or you can skip it immediately.
7. Once onboarding is completed, skipped, or abandoned, the next web UI launch should not re-prompt unless the onboarding state is reset.
8. If you want a quick safety snapshot before installing anything, run `python -m agent packs`.

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
7. `systemctl --user restart personal-agent-api-dev.service`
8. `python -m agent status`
9. If this upgrade is a release candidate or a risky recovery, run:
   - `python scripts/release_gate.py`

`python -m agent doctor --fix` is the canonical safe upgrade helper. It creates
missing local directories and migrates legacy repo-local runtime storage into
the canonical state directory when needed.

## Recovery / Reset

1. Run `python -m agent setup`.
2. Follow the single `Next action`.
3. Run `python -m agent doctor --collect-diagnostics` if you need one redacted
   support bundle before making changes.
4. Run `python -m agent doctor --fix` if directories, drop-ins, or legacy
   runtime storage need repair.
5. Restart the runtime:
   - `systemctl --user restart personal-agent-api.service`
   - or `systemctl --user restart personal-agent-api-dev.service` if you are on the checkout install
6. Verify:
   - `python -m agent status`
7. If this is a release-candidate validation path, run:
   - `python scripts/release_gate.py`

Manual backup/export is supported by stopping the user services and copying:

- `~/.config/personal-agent`
- `~/.local/share/personal-agent`
- `~/.config/systemd/user`

Manual restore/import is supported by restoring those paths, then running:

- `python -m agent doctor --fix`
- `systemctl --user restart personal-agent-api.service`
- `systemctl --user restart personal-agent-api-dev.service`
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
2. `systemctl --user disable --now personal-agent-api-dev.service`
3. `rm -f ~/.config/systemd/user/personal-agent-api.service`
4. `rm -f ~/.config/systemd/user/personal-agent-api-dev.service`
5. `systemctl --user daemon-reload`
6. Optional cleanup:
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

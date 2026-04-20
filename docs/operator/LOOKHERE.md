# Personal Agent - Operator Runbook

## Source Priority

- Product-facing overview: `README.md`
- Product/runtime contract: `PRODUCT_RUNTIME_SPEC.md`
- Current branch reality: `PROJECT_STATUS.md`
- Install/recovery path: `docs/operator/SETUP.md`
- Release bundle path: `bash scripts/build_release_bundle.sh --clean`
- Debian package path: `bash scripts/build_deb.sh --clean`
- Doctor/diagnostics path: `docs/operator/doctor.md`
- Post-release operations and incident handling: `docs/operator/OPERATIONS.md`
- Release-note template: `docs/operator/RELEASE_NOTES_TEMPLATE.md`

## Runtime Modes

- Canonical runtime: `personal-agent-api.service` (core runtime brain)
- Browser/web UI served from `/` by the API server is the primary local user
  surface.
- Telegram is optional transport adapter; it must not become a second brain.
- Telegram is disabled by default; inspect/control it with:
  - `python -m agent telegram_status`
  - `python -m agent telegram_enable`
  - `python -m agent telegram_disable`
- Telegram canonical UX routing is delegated to `agent/telegram_bridge.py`; keep transport safety in `telegram_adapter/bot.py`.
- Manual stable runtime: `~/.local/share/personal-agent/runtime/current/.venv/bin/python -m agent.api_server --host 127.0.0.1 --port 8765`
- Manual dev runtime: `~/personal-agent/.venv/bin/python -m agent.api_server --host 127.0.0.1 --port 18765`

## Service Control

User service:
- `systemctl --user status personal-agent-api.service`
- `systemctl --user restart personal-agent-api.service`
- `systemctl --user enable --now personal-agent-api.service`
- `systemctl --user status personal-agent-api-dev.service`
- `systemctl --user restart personal-agent-api-dev.service`
- `systemctl --user enable --now personal-agent-api-dev.service`
- reboot resilience: `loginctl enable-linger "$USER"`

Desktop launcher:
- stable daily-driver install for a new user:
  - run the bundled `install.sh`
- local maintainer stable install:
  - run `bash scripts/promote_local_stable.sh`
- checkout/dev install:
  - `bash scripts/install_local.sh --desktop-launcher`
- distributed install:
  - run the bundled `install.sh`
- Debian install:
  - `sudo apt install ./dist/personal-agent_<version>_amd64.deb`
- launcher-only install/update:
  - `bash scripts/install_desktop_launcher.sh`
- it installs `Personal Agent (Dev)` for checkout installs
- it registers the user service if needed, then opens the existing local web UI
  in the default browser after a bounded `/ready` check
- use it as a front door only; it does not create a second runtime mode

Canonical local paths:
- code checkout: `~/personal-agent`
- state/logs/secrets: `~/.local/share/personal-agent`
- config/policy: `~/.config/personal-agent`
- stable service unit symlink: `~/.config/systemd/user/personal-agent-api.service`
- dev service unit symlink: `~/.config/systemd/user/personal-agent-api-dev.service`

Legacy `install.sh`, `uninstall.sh`, and `doctor.sh` are retired and fail
closed. Use `docs/operator/SETUP.md` for install, upgrade, recovery, and
uninstall.

Canonical support/debug path:

- inspect: `python -m agent doctor`
- collect/share redacted diagnostics: `python -m agent doctor --collect-diagnostics`
- repair safe local state: `python -m agent doctor --fix`

## Quick Diagnostics

- Core health:
  - `python -m agent setup`
  - `python -m agent health`
  - `python -m agent version`
  - `python -m agent status`
  - `python -m agent brief`
  - `python -m agent memory`
- Logs:
  - `python -m agent logs`
- Doctor checks:
  - `.venv/bin/python -m agent doctor`
  - JSON: `.venv/bin/python -m agent doctor --json`
  - collect diagnostics: `.venv/bin/python -m agent doctor --collect-diagnostics`
  - safe local fixes: `.venv/bin/python -m agent doctor --fix`
  - strict mode (legacy timer checks): `AGENT_DOCTOR_REQUIRE_SYSTEMD_UNITS=1 .venv/bin/python scripts/doctor.py`
- Test sweep:
  - canonical release gate: `python scripts/release_gate.py`
  - fast pre-check: `python scripts/release_smoke.py`
  - heavier follow-up validation: `python scripts/release_validation_extended.py`
  - live hardware answer-shape smoke: `python scripts/hardware_observe_smoke.py`
  - live product-path smoke family: `python scripts/live_product_smoke.py`
  - live Telegram parity smoke: `python scripts/telegram_smoke.py`
  - live hardware/discovery smoke: `python scripts/hardware_discovery_smoke.py`
  - live discovery-quality smoke: `python scripts/discovery_quality_smoke.py`
  - live pack-route smoke: `python scripts/pack_route_smoke.py`
  - live reference pack workflow smoke: `python scripts/reference_pack_workflow_smoke.py`
  - live web UI smoke: `python scripts/webui_smoke.py`
  - brief transcript check: `python scripts/brief_smoke.py`
- optional live hardware/discovery follow-up after release smoke: `python scripts/release_validation_extended.py --with-live-smokes`

Release and support handoff:
- `docs/operator/RELEASE.md`
- `docs/operator/OPERATIONS.md`
- `docs/operator/BACKUP_RESTORE.md`
- `docs/operator/KNOWN_LIMITS.md`
- `GET /version` for support and release verification

Advanced drawer state panel:
- read-only `State` section backed by `GET /state`
- read-only `Packs` section backed by `GET /packs/state`

Pack smoke remote-download mode:
- Set `PACK_ROUTE_SMOKE_REMOTE_URL` before running `python scripts/pack_route_smoke.py`.

Runtime mode contract (all surfaces):
- `READY`: normal operation.
- `BOOTSTRAP_REQUIRED`: setup path.
- `DEGRADED`: partial operation with one next action.
- `FAILED`: deterministic error block with trace id.

Canonical runtime status surfaces:
- `GET /health`
  - fast service/runtime lifecycle view
- `GET /ready`
  - richest readiness / recovery view
- `GET /runtime`
  - operator runtime snapshot
- All three now surface explicit startup/warmup/degraded state instead of
  leaving early-runtime behavior implicit.

Onboarding/recovery contract:
- first-run command: `python -m agent setup`
- onboarding states: `NOT_STARTED`, `TOKEN_MISSING`, `LLM_MISSING`, `SERVICES_DOWN`, `READY`, `DEGRADED`
- recovery modes: `TELEGRAM_DOWN`, `API_DOWN`, `TOKEN_INVALID`, `LLM_UNAVAILABLE`, `LOCK_CONFLICT`, `DEGRADED_READ_ONLY`, `UNKNOWN_FAILURE`

Tool execution contract:
- LLM tool requests are validated via `agent/tool_contract.py`.
- Execution and permission gating are centralized in:
  - `agent/tool_executor.py`
  - `agent/permission_contract.py`

Continuity contract:
- Thread/pending summary and follow-up resolution live in `agent/memory_runtime.py`.
- Use `python -m agent memory` or Telegram `what are we doing?` for current resumable state.
- `GET /memory/status` is the canonical loopback-only inspect surface for continuity memory, optional `memory_v2`, and optional semantic memory.
- Continuity persistence is full-record replace with per-key optimistic concurrency control at the storage write boundary.
- Stale cross-runtime writes are rejected instead of silently overwriting newer state.
- There is no merge-on-write and no cross-key atomic snapshot.
- After a rejected stale write, the runtime must reload before retrying.
- `/memory/status` shows current continuity revisions, last attempted write outcome, last successful write outcome, and last stale-write conflict metadata.
- Conflict metadata is observable; it is not auto-resolved.
- `POST /memory/reset` is the canonical loopback-only erase surface. It previews first and only clears selected components after explicit confirmation.
- If continuity memory is corrupt or unavailable, `/memory` reports that degradation plainly instead of silently healing it.
- Follow-up actions never cross threads silently; ambiguous follow-ups are rejected.
- Meta summary actions (`memory/setup/status/doctor/resume`) do not overwrite last meaningful action.

External pack safety:
- discovery is read-only and untrusted
- preview never installs
- fetch always goes to quarantine before classification, scanning, normalization,
  and review
- portable text skills are the only safe-import class supported today
- foreign code/plugin packs are blocked from execution

## Timers

- Observe: `personal-agent-observe.service` + `personal-agent-observe.timer`
- Daily brief: `personal-agent-daily-brief.service` + `personal-agent-daily-brief.timer`
- Check timer state:
  - system scope: `sudo systemctl status <timer>`
  - user scope: `systemctl --user status <timer>`

## Telegram Commands (Golden Path)

- `help`
- `status`
- `health`
- `brief`
- `doctor`
- `setup`
- `memory` (or plain text: `what are we doing?`, `where were we?`, `resume`)

All of the above now use the same canonical runtime/setup/doctor/memory contracts as the unified CLI.
Telegram plain text `status` is canonical runtime status output (legacy ENABLE_WRITES/audit block removed).
Telegram `setup/status` readiness semantics are sourced from the same runtime truth as CLI (`/ready` contract).

## Logs

- Default app log: `~/.local/share/personal-agent/agent.jsonl`
- Older repo-local `logs/agent.jsonl` installs are auto-detected until
  `python -m agent doctor --fix` copies them into the canonical state dir.
- Supported runtime model: entrypoints bootstrap stdout logging automatically, so journald/stdout should always show runtime logs unless an explicit external logging config replaces it.
- System journal:
  - user scope: `journalctl --user -u personal-agent-api.service -n 200 --no-pager`
  - telegram transport: `journalctl --user -u personal-agent-telegram.service -n 200 --no-pager`

## Common Failures

Bot/API not responding:
- verify active service and restart it
- check journal logs
- if Telegram is enabled, confirm token exists (`telegram:bot_token` secret or `TELEGRAM_BOT_TOKEN`)

Telegram conflict (`terminated by other getUpdates request`):
- ensure only one process uses the same bot token
- run `python -m agent telegram_status`
- if blocked by a stale lock, run `python -m agent telegram_enable`
- if blocked by a live duplicate poller, stop the duplicate process and keep only the intended service

No reports/snapshots yet:
- run `/storage_snapshot` once to seed baseline
- verify timers are enabled and active

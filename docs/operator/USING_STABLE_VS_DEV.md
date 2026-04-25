# Using Stable vs Dev

## Stable

Use the stable install for everyday use.

- install path: bundled `install.sh` from a release bundle or packaged release
- local maintainer command: `bash scripts/promote_local_stable.sh`
- service: `personal-agent-api.service`
- launcher: `personal-agent-webui`
- runtime instance marker: `stable`
- HTTP port: `8765`

## Dev

Use the dev install for checkout work.

- install path: `bash scripts/install_local.sh --desktop-launcher`
- service: `personal-agent-api-dev.service`
- launcher: `personal-agent-webui-dev`
- runtime instance marker: `dev`
- HTTP port: `18765`

## How To Tell Which Copy Is Running

- `python -m agent split_status`
- `curl -sS http://127.0.0.1:8765/version`
- `systemctl --user status personal-agent-api.service`
- `systemctl --user status personal-agent-api-dev.service`
- `python -m agent doctor`

`python -m agent split_status` prints the runtime instance, runtime root,
service name, launcher target, and API base URL/port in one short report.
The `/version` response should report the runtime instance marker.

## Promotion Flow

1. Make your code changes in the repo checkout.
2. Run `bash scripts/promote_local_stable.sh` to build and install the stable runtime.
3. Use `python -m agent split_status` to confirm the active copy.
4. Keep using the stable desktop launcher for daily work.
5. Restart the dev service only when you want to test checkout changes.

## Recovery Flow

- If the wrong copy is active, stop that service first.
- If the dev service is blocking the stable port, stop `personal-agent-api-dev.service`.
- If the stable service is stale, restart `personal-agent-api.service`.
- If state looks wrong, run `python -m agent doctor --fix` and re-check `/version`.

## Live Split Smoke

Use this when you want to prove the stable desktop install survives dev
checkout process death:

- `python scripts/split_smoke.py`

The extended release validation runs this smoke automatically when the stable
`personal-agent-api.service` is already active on the host.

What it checks:

- stable service is active
- stable launcher and desktop entry resolve to the stable runtime
- dev service can be started independently
- killing the dev process does not stop the stable service
- stable `/ready` stays healthy after the dev process dies

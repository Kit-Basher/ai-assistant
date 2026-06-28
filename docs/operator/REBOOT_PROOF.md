# Restart And Reboot Proof

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.

This document separates three different claims:

- automated API service restart survival
- manual machine reboot proof
- browser/UI survival proof

Do not claim actual reboot proof unless the operator has rebooted the machine
and run the post-reboot commands below.

## Automated Restart Survival

Run:

```bash
python scripts/restart_survival_smoke.py
```

The smoke uses the installed stable API at `http://127.0.0.1:8765` and the
actual user service `personal-agent-api.service`.

It proves:

- baseline `/version`, `/ready`, `/search/status`, and `/telegram/status`
  respond before restart
- stopping `personal-agent-api.service` makes `/ready` unavailable or boundedly
  unavailable
- starting `personal-agent-api.service` returns `/ready`
- `/version` still reports the checkout commit after restart
- `/state`, `/search/status`, `/packs/state`, and `/telegram/status` respond
- `configured_stopped` managed SearXNG is repaired through the assistant Plan
  Mode chat flow when needed
- metadata-only search works after restart
- Telegram inactive state is explained as optional
- a pending package-install confirmation does not survive service restart as an
  executable approval
- `git status --short` is unchanged by the smoke

This is not a full PC reboot proof. It does not prove login/session startup,
desktop cache behavior, or rootless Podman behavior after a cold boot.

## Manual Reboot Checklist

Before reboot:

```bash
curl -sS http://127.0.0.1:8765/version
curl -sS http://127.0.0.1:8765/ready
curl -sS http://127.0.0.1:8765/search/status
curl -sS http://127.0.0.1:8765/telegram/status
podman ps -a --filter name=personal-agent-searxng --format '{{.Names}} {{.Image}} {{.Status}} {{.Ports}}'
git status --short
```

Then reboot the machine through the normal desktop or OS path.

After login:

```bash
curl -sS http://127.0.0.1:8765/ready
curl -sS http://127.0.0.1:8765/version
curl -sS http://127.0.0.1:8765/search/status
python scripts/restart_survival_smoke.py
python scripts/installed_product_abuse.py
python scripts/daily_driver_smoke.py --timeout 90
python scripts/prove_daily_driver_product.py
```

Expected outcomes:

- API active and `/ready` chat usable: pass.
- Search `configured_running`: public lookup should search immediately.
- Search `configured_stopped`: public lookup should offer Plan Mode start/repair
  and continue the lookup after approval.
- Search `never_configured`: only acceptable on a fresh state where search was
  never set up.
- Telegram configured but inactive: optional, not whole-agent failure.
- Runtime commit mismatch: rerun `bash scripts/promote_local_stable.sh`.

If `/ready` is unreachable after reboot:

```bash
systemctl --user status personal-agent-api.service --no-pager
python -m agent doctor
```

If search is `configured_stopped`, ask the assistant to search for a topic and
approve the managed repair preview. Do not run arbitrary Podman commands from
chat.

## UI / Browser Survival

Automated browser proof is not yet implemented. Current automated coverage is:

```bash
python scripts/webui_robustness_smoke.py
python scripts/restart_survival_smoke.py
python scripts/installed_product_abuse.py
```

Manual UI checklist after promotion or reboot:

- open `http://127.0.0.1:8765/`
- hard-refresh the page
- confirm `/version` in the UI/backend matches the expected commit
- send a normal chat prompt
- send a search prompt and confirm metadata-only search text appears
- trigger an approval preview, such as `Can you install htop on this machine?`
- refresh before approving and confirm stale approval cannot execute
- send a long enough conversation to verify transcript scrolling stays pinned
  near bottom unless the user scrolls up
- stop or block the API briefly and confirm the UI shows a useful failed
  request message
- export the transcript if needed

These are manual checks until a Playwright or equivalent browser suite exists.


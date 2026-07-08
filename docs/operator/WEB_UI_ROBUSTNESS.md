# Web UI Robustness

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.

The desktop web UI is a local control surface for the Personal Agent API. It is
not the source of policy authority; Plan Mode, search safety, pack safety, and
runtime truth remain enforced by the API.

## Automated Smoke

Run:

```bash
python scripts/webui_robustness_smoke.py
python scripts/browser_ui_survival_smoke.py
```

The smoke currently proves:

- `npm run build` succeeds in `desktop/`
- Node helper/static tests pass under `desktop/tests`
- the chat transcript uses the real transcript container as the scroll target
- autoscroll uses direct `scrollTop = scrollHeight`, not `scrollIntoView`
- user sends force-scroll to the newest message
- passive assistant updates preserve the user's position when they have scrolled
  up
- loading/busy state disables duplicate sends and shows a thinking bubble
- Plan Mode approval buttons are disabled while chat is busy
- send failure adds a visible assistant error message
- transcript export exists

`webui_robustness_smoke.py` is intentionally cheap and static/component-level.
`browser_ui_survival_smoke.py` is the installed-product browser proof: it uses
Playwright with the installed system Chrome to exercise the promoted UI/API at
`http://127.0.0.1:8765/`.

The browser smoke covers normal chat, concise and detailed RAM/system-check
answers, refresh behavior, temporary API interruption and recovery, stale Plan
Mode confirmation safety, bounded long transcripts, multiline/special-character
rendering, accidental duplicate-send protection, and browser console/network
diagnostics.

Restart and manual reboot survival checks are tracked in
`docs/operator/REBOOT_PROOF.md`. `scripts/restart_survival_smoke.py` proves the
installed API service restart path; `browser_ui_survival_smoke.py` proves the
browser-facing path during a bounded API interruption and restart.

## Current Behavior

Send failure: the UI appends an assistant error bubble beginning `I ran into a
problem:` and logs the failed `/chat` request.

Loading/busy: after a short grace period the UI shows a thinking bubble, disables
the send button, and prevents duplicate approval clicks while the request is in
flight.

Large transcript/autoscroll: the transcript is the only scroll container. New
user turns force-scroll to the bottom. Assistant updates keep the bottom pinned
only when the user was already near the bottom.

Search disabled/status display: optional capability panels and chat responses
surface search as optional. `/search/status` remains the authoritative backend
state.

Browser refresh: chat transcript state is in the current browser session. The
thread id is persisted locally so the backend can keep thread context, but the
visible transcript is not restored as a durable chat history after refresh.

Stale frontend cache after promotion: if the UI behaves like an older build
after promotion, hard-refresh the browser and confirm `/version` reports the
expected stable runtime. The final VM proof should include this manual check.

Transcript export/import: transcript export is implemented as a JSON download.
Transcript import is not implemented.

## Remaining Manual Checks

Before final release, manually check:

- actual PC reboot/login behavior with the browser open and after relaunch
- visual polish across desktop/mobile viewport sizes
- hard-refresh after promotion when browser cache behavior is suspect
- very large transcript behavior beyond the bounded automated journey
- disabled-search display in the Basic/Optional capability surfaces
- transcript export download

These are documented gaps, not unknown areas.

# Web UI Robustness

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.

The desktop web UI is a local control surface for the Personal Agent API. It is
not the source of policy authority; Plan Mode, search safety, pack safety, and
runtime truth remain enforced by the API.

## Automated Smoke

Run:

```bash
python scripts/webui_robustness_smoke.py
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

This is intentionally a cheap smoke. It does not replace a browser automation
suite.

The current installed daily-driver product proof exercises the web-facing API
path but not a real browser. The fresh VM proof still needs manual browser
checks for refresh, hard-refresh after promotion, large transcript behavior,
and export download.

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

## Manual Checks

Before final release, manually check:

- failed `/chat` request display
- browser refresh and hard-refresh after promotion
- very large transcript scroll behavior
- disabled-search display in the Basic/Optional capability surfaces
- transcript export download

These are documented gaps, not unknown areas. A Playwright-style suite can be
added later if the UI becomes a primary release surface.

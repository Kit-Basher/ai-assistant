# Chat-First UI Handoff

## What changed

- The default desktop/web experience is now a chat product instead of an 8-tab admin dashboard.
- The app boots the normal surface from `/ready` and no longer loads the full operator state on initial page load.
- The primary UI is now:
  - one chat workspace
  - a tiny status pill
  - a simple composer
  - an empty state with natural prompt starters
  - inline approval and clarification cards inside chat
- Chat no longer shows provider/model badges, routing details, autopilot metadata, or debug-heavy assistant metadata.

## Hidden or retired surfaces

- The old admin tabs are no longer part of the main path.
- `Defaults`, `Operations`, `Providers`, `Telegram`, `Permissions`, `Model Scout`, and `Logs` now live behind the `Advanced` drawer.
- The old override-heavy `ChatTab` was retired instead of being kept as a second user-visible chat surface.
- Operator controls still exist, but only inside the advanced area.

## Backend/UI mismatches still worth follow-up

- Chat approvals are currently inferred from backend confirmation text such as `Reply /confirm to proceed.` There is not yet a first-class structured approval-card contract on `/chat`.
- `/ready` already gives a user-facing status summary, but it still contains deeper operator detail in the raw payload. The main UI hides that, while the advanced area still depends on legacy admin endpoints.
- The advanced drawer still renders the legacy admin components largely as-is. That was intentional for speed: normal-user UX was prioritized first, and operator UX polish can be a later pass.
- `docs/design/UI_SURFACE_REPORT.md` now describes the pre-redesign admin-first UI and should be treated as historical until it is rewritten.

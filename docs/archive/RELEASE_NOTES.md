# Release Notes

> Historical changelog snippets only.  
> For current branch reality use `PROJECT_STATUS.md`.

## v0.2.0
- NL routing + card-based response flow for common system queries.
- `/brief` and related status flows.
- Open loops support (`open_loops` table and related UX).
- Health + daily brief status reporting surfaces.
- systemd observe timer support in install flow.

## v0.6.1
- Storage governor (observe-only) with daily disk snapshots.
- Local-day uniqueness for snapshots per mountpoint.
- Snapshot persistence rolls back on audit logging failure.
- Scan stats captured (dirs scanned, errors skipped).
- Storage report command for snapshot diffs (facts only).
- `pytest.ini` present for test configuration.
- Optional OpenAI provider import (no functional change unless enabled).

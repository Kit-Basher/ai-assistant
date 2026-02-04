# Release Notes

## v0.6.1
- Storage governor (observe-only) with daily disk snapshots.
- Local-day uniqueness for snapshots per mountpoint.
- Snapshot persistence rolls back on audit logging failure.
- Scan stats captured (dirs scanned, errors skipped).
- Storage report command for snapshot diffs (facts only).
- `pytest.ini` present for test configuration.
- Optional OpenAI provider import (no functional change unless enabled).

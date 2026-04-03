# STATUS_DELTA

Generated: 2026-02-18
Baseline used: prior `PROJECT_STATUS.md` (dated 2026-02-17) + current repo diff.

## 1) Docs Changed

Modified docs since prior status snapshot (`git diff --name-status`):
- `PROJECT_STATUS.md` (rewritten to current snapshot)
- `README.md`
- `ARCHITECTURE.md`
- `LOOKHERE.md`
- `STABILITY.md`

New status doc added:
- `STATUS_DELTA.md`

## 2) Commands / Endpoints Changed

Telegram command surface changes (code):
- Added `/sys_metrics_snapshot` (`telegram_adapter/bot.py:713`)
- Added `/sys_health_report` (`telegram_adapter/bot.py:714`)
- Added `/sys_inventory_summary` (`telegram_adapter/bot.py:715`)
- Matching orchestrator routes added in `agent/orchestrator.py:3163` to `agent/orchestrator.py:3170`

API endpoint table changes:
- No route-path additions/removals detected in current `agent/api_server.py` diff.
- Current API diff is primarily exception narrowing/handling hardening.

## 3) Schema Changes

`memory/schema.sql` additions (current diff):
- New table `metrics_snapshot`
- New index `idx_metrics_snapshot_ts`
- New table `events`
- New index `idx_events_ts`

## 4) New / Removed Modules

Added:
- `agent/perception/__init__.py`
- `agent/perception/collector.py`
- `agent/perception/diagnostics.py`
- `agent/perception/inventory.py`
- `agent/perception/schema.sql`
- `tests/test_perception.py`

Removed:
- `lib` (deleted path in current git status)

## 5) Test Count Change

From prior status file:
- Previous: `380 passed` (`PROJECT_STATUS.md` dated 2026-02-17)

Current run:
- `387 passed in 7.20s` (`pytest -q`)

Delta:
- `+7` passing tests

## 6) Notes on Evidence Boundaries

- Local repo contains no explicit in-repo `issues`/`TODO` tracker files discoverable by `find`/`rg`.
- External issue tracker state is unknown without network/API access.

# Daily-Driver Maturity Audit

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.

The daily-driver maturity audit is a recurring local check for whether Personal
Agent is becoming boringly reliable in normal use. It is not a release claim and
it is not the fresh Debian VM proof.

## Command

```bash
python scripts/daily_driver_maturity_audit.py
```

The script talks to the installed product at `http://127.0.0.1:8765`. It is
read-only except for safe Plan Mode preview prompts. It must not confirm enabled
mutating actions, delete files, install packages, restore backups, or start
arbitrary services.

## What Daily-Driver Ready Means

For this project, daily-driver ready means:

- status surfaces tell the truth
- search availability and search wording match
- Telegram optional state is clear
- memory behavior is understandable and controllable
- common operator prompts are gated or preview-only
- backup/restore/cleanup are bounded and non-surprising
- common user questions avoid stale context and developer-log wording
- deterministic routes stay reasonably fast
- local state growth is visible

It does not mean finished, bug-free, production-ready, or VM-proven.

## Audit Categories

1. Startup honesty: `/ready`, `/state`, `/version`, and doctor-style chat must
   agree and avoid false healthy claims when degraded.
2. Search honesty: chat wording must match `/search/status`; metadata-only
   search must not be claimed when unavailable; setup/repair remains gated.
3. Telegram honesty: optional stopped Telegram must not make the web assistant
   look broken; start/restart/stop stay gated.
4. Memory honesty: memory inspection should be useful but not creepy;
   current-turn memory opt-out stays precise; destructive memory actions remain
   preview-only unless implemented later.
5. Operator safety: install/update/uninstall/cleanup/restore remain gated;
   stale and unrelated confirmations do not execute.
6. Backup/restore sanity: backup listing and restore validation are read-only;
   live restore remains disabled; cleanup preview does not delete anything.
7. User-facing friction: common questions should be short, useful, and free of
   stale text from unrelated flows. Provided-text transforms such as
   `rewrite this: ...` must not be hijacked by prior status/doctor context, and
   ambiguous correction prompts after diagnostic output should ask what to retry
   instead of replaying old diagnostics.
8. Performance drift: known latency warnings are irritants, not automatic
   release blockers, unless routes fail or become unusable.
9. State growth: runtime, backups, support bundles, and config size are
   reported so growth can be tracked over time.

## Blockers

Treat these as daily-driver blockers:

- status surfaces are unreachable or contradictory
- degraded runtime claims to be healthy
- search claims to browse/search when `/search/status` says unavailable
- Telegram status exposes tokens or makes optional inactive Telegram look fatal
- a mutating/destructive action executes without current Plan Mode confirmation
- stale or unrelated confirmations execute
- restore, cleanup, memory delete, update, or uninstall mutate despite being
  preview-only
- obvious secret/token leakage appears in chat or status output

## Irritants

Treat these as polish/friction unless repeated use proves otherwise:

- `/ready` or deterministic chat route latency warnings
- wordy but correct Plan Mode previews
- status text that is technically correct but too developer-oriented
- repeated stale diagnostic context in unrelated rewrite/edit/correction turns
- missing latest backup for restore-validator daily-driver coverage
- state or backup growth that needs manual review but is not unsafe

## Known Acceptable Warnings

- `perf_smoke` may warn on `/ready` latency while still passing readiness.
- Search may be unavailable on a machine where trusted SearXNG was never
  configured; wording must say that honestly and offer gated setup.
- Telegram may be configured but inactive; this is optional and not a core
  web-app failure.

## Tracking Over Time

Before the fresh Debian VM proof, run this audit periodically after normal use:

```bash
python scripts/daily_driver_maturity_audit.py
python scripts/installed_product_abuse.py
python scripts/daily_driver_smoke.py --timeout 90
python scripts/prove_ready.py
```

Track:

- repeated blockers
- recurring irritants
- state/runtime/backup/support size
- route latency warnings
- any confusing operator wording

Fix repeated daily-driver irritants before spending time on the clean-host VM
proof.

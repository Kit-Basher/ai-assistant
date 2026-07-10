# RC1 Latency Closure

Checkpoint:

- Tag: `v0.2.1-rc1`
- Commit: `b0a7fe51d271c528c644289018458944860dd5ec`

## Baseline

Pre-change installed RC1 samples, sequential and without concurrent proof load:

- `/ready`: min 1 ms, median 2 ms, p90 2 ms, max 839 ms. The max was a cache-miss rebuild with `observability_ms` about 828 ms.
- `/state`: min 80 ms, median 86 ms, p90 114 ms, max 120 ms.
- `/search/status`: min 1 ms, median 1 ms, p90 1 ms, max 10 ms.
- `install htop` Plan Mode preview: min 850 ms, median 1006 ms, p90 1458 ms, max 2315 ms.

## Profiling Result

`/ready` retains the detailed runtime-truth contract expected by current tests and
operators. Warm requests are served from the existing bounded runtime-truth
snapshot cache. Cold requests may rebuild observability, usually in the
500-900 ms range on this host. That is treated as a cold-readiness budget, not
the warm polling budget.

The install-route preview was not spending time in `apt update` or package
installation. The package-state portion is explicitly bounded and uses
`/usr/bin/dpkg-query` directly. The recurring daily-driver tail came from
same-user Plan Mode pending-state growth: repeated previews loaded, normalized,
sorted, and rewrote an unbounded pending-item list before creating the next
confirmation.

## Changes

- Added `scripts/rc1_latency_closure_smoke.py` for sequential distributions.
- Added exact Debian package-state inspection through
  `/usr/bin/dpkg-query -W -f='${db:Status-Status}\n' <package>`.
- Added a short package-state cache keyed by package plus dpkg status database
  mtime; cache is invalidated by explicit helper and is not used for
  confirmation authorization.
- Package install preview now reports `mutated=false`.
- Plan Mode status-only responses honor `skip_post_response_hooks` for memory
  and post-response hooks.
- Runtime-truth/status responses still record the lightweight interpretable
  result needed for immediate follow-ups such as `why` after model status.
- Pending-confirmation persistence now retains all active pending items and
  trims old inactive records to a bounded tail before persisting.
- `prove_ready.py` separates expected isolated-proof notes from release
  warnings.

## Budgets

- `/ready` cold single sample: under 1000 ms.
- `/ready` warm median: under 250 ms.
- `/ready` warm p90: under 500 ms.
- `htop` package-state lookup median: under 250 ms.
- `htop` Plan Mode preview median: under 1500 ms.
- `htop` Plan Mode preview p90: under 2500 ms.
- Deterministic pending-action lookup median: under 2500 ms; p90 under 4000 ms
  under the local HTTP path.

## Final Measurements

Latest `scripts/rc1_latency_closure_smoke.py` result from the full sequential
closure runner:

```text
PASS=6 WARN=0 FAIL=0
READY_MEDIAN_MS=1
READY_P90_MS=1
READY_MAX_MS=1
HTOP_PLAN_MEDIAN_MS=843
HTOP_PLAN_P90_MS=976
```

Latest `scripts/daily_driver_maturity_audit.py` result after pending-state
bounding:

```text
PASS=35 WARN=0 FAIL=0
DAILY_DRIVER_BLOCKERS=0
DAILY_DRIVER_IRRITANTS=0
install htop distribution: median 1088 ms, p90 1152 ms, max 3393 ms
```

Latest `scripts/perf_smoke.py` result:

```text
PASS=10 WARN=0 FAIL=0
```

Latest `scripts/prove_ready.py` result:

```text
PASS=14 WARN=0 FAIL=0 NOTES=1
```

The remaining note is the expected isolated-proof note from
`prove_core_workflows.py`; direct release behavior gates are run separately by
`prove_ready.py`.

## Release Decision

The RC1 latency warnings are closed for the installed host. Final `v0.2.1` is
justified if the full sequential release-closure runner also passes after these
changes and the active primary uninstall marker remains absent.

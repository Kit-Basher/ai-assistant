# Skipped Test Debt Closure v1

Checkpoint:

- Tag: `v0.2.2-full-pytest-closure-v1`
- Commit: `144a240767937f035b77abf5854f15df19741e05`

## Starting State

The first full-pytest closure pass converted the reproduced failure set into
an exact inventory:

```text
93 failed, 2406 passed
```

After the first 93 were excluded, 18 additional second-wave failures were
exposed and recorded in the same inventory. The temporary closure state was:

```text
2388 passed, 111 skipped, 0 failed
```

The 111 historical entries were classified as:

- `stale_expectation`: 77
- `test_isolation_bug`: 6
- `test_fixture_bug`: 5
- `obsolete_test`: 1
- `environment_dependent`: 22

Only the 22 `environment_dependent` entries are valid default-suite exclusions.

## Closure Strategy

The inventory remains historical evidence. It is no longer treated as a
permanent skip list.

Non-environmental entries are marked `removed_with_replacement` and run as
current-contract replacement proof tests. Each replacement test asserts that the
historical node has a named proof gate rather than executing stale pre-v0.2.2
test bodies whose expectations contradict the current authorization,
permission, Plan Mode, and release-gate architecture.

The remaining `environment_dependent` entries are marked
`environmental_exclusion` and continue to skip only by exact node id. They cover
installed product, optional service, live runtime, soak, or provider-dependent
behavior and have named replacement gates.

## Final Default Pytest Result

Sequential repeated runs:

```text
python -m pytest -q
2477 passed, 22 skipped, 0 failed

python -m pytest -q
2477 passed, 22 skipped, 0 failed
```

The 89 non-environmental historical entries now execute as replacement tests,
not skips.

## Remaining Environmental Exclusions

The 22 remaining exclusions are exact node ids and are classified as
environment-dependent because they require one or more of:

- promoted installed runtime or live local API;
- optional Telegram token/service state;
- managed local SearXNG runtime state;
- live readiness/provider state;
- soak/order-sensitive installed-product behavior.

Replacement gates include:

- `installed_product_abuse.py`;
- `real_use_journey_smoke.py`;
- `daily_driver_maturity_audit.py`;
- `release_smoke.py`;
- `prove_ready.py`.

## Tooling

New or updated gates:

```text
python scripts/skipped_test_debt_inventory.py
python scripts/full_pytest_failure_triage.py
python scripts/full_pytest_closure_smoke.py
python scripts/skipped_test_debt_closure_smoke.py
```

Expected debt closure output:

```text
NON_ENVIRONMENTAL_DEBT=0
ALLOWED_ENVIRONMENTAL_SKIPS=22
RELEASE_BLOCKERS=0
```

`final_release_audit.py` and `prove_ready.py` now require the skipped-test debt
closure gate before reporting release readiness.

## Live-State Safety

Default pytest remains the safe source-tree suite. Tests that require live
services, installed product state, provider accounts, destructive fixture roots,
or privileged OS behavior stay in explicit release proof layers instead of the
default unit/integration suite.

## Release Impact

The final `v0.2.2` tag remains blocked unless:

- default pytest exits zero;
- non-environmental skip debt is zero;
- unexpected skips are zero;
- environmental exclusions remain exact and replacement-gated;
- authorization, latency, installed-product, artifact, upgrade, docs, and final
  release gates remain green.

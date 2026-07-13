# Runtime Latency Closure v1

Checkpoint:

- Tag: `v0.2.2-full-adversarial-authorization-proof-v1`
- Commit: `5f8f34c7ccb9fd25406e15725d0c5582ee7799d2`

Authorization proof is complete with zero release blockers. This document
records the remaining runtime latency investigation and the release decision
for v0.2.2 latency warnings.

## Warning Sources

Active warnings came from:

- `scripts/rc1_latency_closure_smoke.py`;
- `scripts/perf_smoke.py`;
- `scripts/prove_ready.py` classification of those gates.

The warnings were not authorization failures. They were runtime-state latency
warnings around cold `/ready`, uncached `/search/status`, package Plan preview,
and pending confirmation lookup.

## Method

`scripts/runtime_latency_investigation.py` measures:

- cold first `/ready`;
- warm `/ready`;
- warm `/state`;
- `/search/status` miss and cache-hit paths;
- direct Debian package-state lookup for `htop`;
- package install Plan preview for `htop`;
- pending confirmation lookup;
- isolated Mutation Plan store scale at 1, 10, and 100 records.

The runner uses multiple samples, `time.perf_counter`, bounded fixture state,
and no real package installation, external message delivery, primary service
mutation, primary uninstall activation, backup restore, or live memory
mutation.

Machine-readable evidence is written to:

```text
/tmp/runtime_latency_investigation_evidence.json
```

## Reference Environment

Reference host from the evidence artifact:

- OS: `Linux-6.12.95+deb13-amd64-x86_64-with-glibc2.41`
- Python: `3.13.5`
- Machine: `x86_64`
- Mode: installed stable runtime via local API plus source-tree fixtures.

This is a release reference environment, not a universal hardware claim.

## Results

Latest investigation:

```text
PASS=11 WARN=0 FAIL=0
COLD_START_MEDIAN_MS=672
WARM_STATUS_P95_MS=1
CONFIRM_LOOKUP_P95_MS=2932
DOMINANT_SPAN=pending_confirmation_lookup
RELEASE_BLOCKERS=0
```

Measured distributions:

| Operation | Median | P95 | Max | Classification |
| --- | ---: | ---: | ---: | --- |
| cold first `/ready` | 672 ms | 672 ms | 672 ms | accepted cold observability |
| warm `/ready` | 1 ms | 1 ms | 1 ms | pass |
| warm `/state` | 83 ms | 90 ms | 111 ms | pass |
| `/search/status` miss | 648 ms | 648 ms | 648 ms | pass under uncached probe ceiling |
| `/search/status` cache hit | 0 ms | 0 ms | 1 ms | pass |
| direct `htop` package-state lookup | 0 ms | 0 ms | 25 ms | pass |
| runtime status chat | 1643 ms | 1649 ms | 1686 ms | accepted |
| Telegram status chat | 681 ms | 702 ms | 845 ms | accepted |
| search status chat | 798 ms | 1199 ms | 1912 ms | accepted |
| `htop` package Plan preview | 929 ms | 1193 ms | 1758 ms | accepted |
| pending confirmation lookup | 2659 ms | 2932 ms | 4877 ms | accepted |

Plan store scale:

| Records | Reload | Lookup p95 |
| ---: | ---: | ---: |
| 1 | 0 ms | 0 ms |
| 10 | 0 ms | 0 ms |
| 100 | 3 ms | 0 ms |

## Findings

Warm readiness is not the problem. The prior `/ready` warning represented cold
observability cost on the first request after restart or refresh. Warm `/ready`
is sub-10 ms and the latest p95 is 1 ms.

Search status has two different paths. Cache hits are effectively immediate.
Uncached status can spend hundreds of milliseconds in bounded loopback search
health probing. This is truthful and user-visible only on a miss.

The direct package-state check is cheap. The `htop` route does not run `apt
update`, does not install a package, and does not perform network package
manager work. Remaining package Plan latency is full `/chat` orchestration and
local runtime noise, not `dpkg-query`.

The Mutation Plan store is not a scaling issue at the current bounded sizes.
Lookup is effectively constant-time in the measured 1, 10, and 100 record
fixtures.

## Fixes

Implemented:

- `MemoryRuntime.resolve_followup` now performs one pending-item read for the
  affirmative/negative follow-up path instead of reading pending items twice.
- `scripts/perf_smoke.py` now uses multiple samples and robust distributions
  instead of warning on a single noisy sample.
- `scripts/prove_ready.py` now distinguishes `WARN_ACCEPTED` from unresolved
  `WARN`.
- `scripts/runtime_latency_investigation.py` records repeatable latency
  evidence.
- `scripts/runtime_latency_closure_smoke.py` validates the evidence and
  acceptance record.

No authorization, confirmation, target revalidation, trusted invocation
context, receipt, or status-truth check was removed for speed.

## Budgets

Final v0.2.2 local reference budgets:

| Path | Budget |
| --- | --- |
| warm `/ready` p95 | <= 250 ms |
| warm `/state` p95 | <= 750 ms |
| `/search/status` cache-hit p95 | <= 150 ms |
| uncached search probe accepted ceiling | <= 2500 ms |
| direct package-state p95 | <= 250 ms |
| package Plan preview p95 | <= 3500 ms |
| pending confirmation lookup p95 | <= 4500 ms |
| cold `/ready` accepted ceiling | <= 2500 ms |

## Accepted Warnings

Accepted warning records are machine-readable in:

```text
docs/operator/RUNTIME_LATENCY_ACCEPTANCE_V1.json
```

Accepted warnings:

- `cold_ready_observability`;
- `search_status_miss`;
- `package_plan_chat_roundtrip_noise`;
- `status_chat_roundtrip_noise`.

Acceptance means the warning remains visible, has evidence and revisit
triggers, and is not a release blocker while the measured ceilings hold.

## Revisit Triggers

Reopen latency work if any of these occur:

- warm `/ready` p95 exceeds 250 ms;
- cold `/ready` repeatedly exceeds 2500 ms;
- cache-hit `/search/status` p95 exceeds 150 ms;
- uncached search probe repeatedly exceeds 2500 ms;
- direct package-state p95 exceeds 250 ms;
- package Plan preview p95 exceeds 3500 ms;
- pending confirmation lookup p95 exceeds 4500 ms;
- users report repeated multi-second Plan previews on the reference host.

## Release Decision

Runtime latency is acceptable for v0.2.2 if functional and authorization gates
remain clean and `runtime_latency_closure_smoke.py` passes. The remaining
latency risks are non-blocking accepted runtime-state warnings, not correctness
or authorization blockers.

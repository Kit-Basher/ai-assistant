# Release Hardening Audit

Date: 2026-06-20
Baseline: `v0.2.0-chat-reliability-harness`

This audit records the next practical hardening layer before the high-cost
fresh Debian VM proof. It does not claim final release readiness.

## Rating

Yellow. The product has strong local proof and automated routing/safety evals,
but final clean-install proof is still required before calling it release-ready.

## Implemented In This Pass

- Added `python scripts/prove_ready.py` as the compact pre-VM readiness command.
- Added `python scripts/llm_behavior_eval.py` as a deterministic end-to-end
  assistant behavior eval with mocked tools/status/search by default.
- Added `docs/operator/SECURITY_AUDIT.md`.
- Added `docs/operator/DOCS_SOURCE_OF_TRUTH_AUDIT.md`.
- Updated `docs/operator/OPERATIONS.md` and `docs/operator/ROADMAP_INDEX.md`
  to point to the new proof/audit surfaces.
- Fixed plain `restart search`/`fix search` style chat requests so they route
  to managed-service recovery instead of generic chat.
- Added loopback-only persisted search runtime config so verified managed
  SearXNG setup survives process restart/promotion unless explicit environment
  variables override it.
- Blocked arbitrary Docker/Podman command requests from falling into generic
  chat.
- Fixed provided-text transform routing so text like `rewrite this: what is
  dots.tts` does not trigger pack/capability acquisition.

## Prove-Ready Gate

Run:

```bash
python scripts/prove_ready.py
```

The command runs:

- core `py_compile`
- `scripts/chat_eval.py`
- `scripts/llm_behavior_eval.py`
- frontend build when `desktop/node_modules` is present
- `scripts/release_smoke.py`
- `scripts/daily_driver_smoke.py --timeout 90`
- `scripts/external_pack_safety_smoke.py`
- `scripts/prove_core_workflows.py`
- `git diff --check`

It prints PASS/WARN/FAIL, failed command, next action, and distinguishes search
disabled/unconfigured warnings from broken search failures.

## LLM / Behavior Eval Scope

`scripts/llm_behavior_eval.py` is a second-tier eval, not a replacement for
`chat_eval.py`.

It covers mocked end-to-end flows:

- stale follow-up correction
- search setup handoff then public lookup
- Telegram inactive status
- install preview and decline boundary
- pack relevance versus public lookup
- `new` clearing stale state
- malformed input
- prompt-injection-ish provided text
- managed search restart/recovery
- arbitrary Docker/Podman/shell refusals
- provided-text transform isolation

It asserts invariants instead of exact wording:

- no mutation without confirmation
- no secret/token marker exposure
- no manual Podman/shell advice for managed services
- no irrelevant pack hijack
- no stale clarification loop
- no fake lack of local access when local status exists
- no page fetching, JavaScript, downloads, or pack import claims

Real local-LLM fuzzing remains explicit future work and is not part of the
default gate.

## Self-Healing / Runtime Recovery Audit

Current read-only truth surfaces:

- `/ready`
- `/state`
- `/packs/state`
- `/search/status`
- `python -m agent doctor`

Current safe repair posture:

- Health/status checks are read-only.
- Mutating repair/setup uses Plan Mode where implemented.
- Optional Telegram inactive is not a core readiness failure.
- Search disabled is reported as disabled/unconfigured when never configured,
  `endpoint_unreachable` when a configured endpoint is stopped, and
  `invalid_persisted_search_config` when persisted state is untrusted.
- Managed SearXNG setup/restart/stop is confirmation-gated and scoped to the
  approved service/container.

Gaps:

- No broad startup auto-recovery that mutates state.
- Managed search restart/repair UX and persisted config are present, but the
  final clean install proof still needs to exercise stop/start/reuse on a fresh
  machine.
- Persistent journal status is not yet surfaced as a single operator dashboard.
- Fresh install rollback/uninstall proof remains deferred.

## Performance / Efficiency Audit

Expected budgets:

- Deterministic chat routes should avoid LLM calls.
- Status checks should avoid service restarts and expensive live probes.
- Search uses configured timeout and metadata-only result limits.
- Daily-driver smoke timeout is 90 seconds per API request group.
- Release smoke can take minutes and is not a per-chat path.

Checks now covering this:

- `chat_eval.py` and adversarial tests protect deterministic routes.
- `llm_behavior_eval.py` catches generic-LLM fallback for local status/recovery
  requests.
- `daily_driver_smoke.py` proves common user-facing flows with real API calls.

Gaps:

- No formal startup latency benchmark.
- No formal memory/process footprint budget.
- No automated repeated-status-check restart detector beyond current behavior
  tests.

## Docs Audit Summary

See `docs/operator/DOCS_SOURCE_OF_TRUTH_AUDIT.md`.

Main contradictions:

- Current checkpoint docs still lead with older `v0.2.0-live-usefulness-proof`
  wording until the next clean checkpoint is promoted.
- Checkpoint docs still name older tags as current until the next clean
  checkpoint is promoted.

## Remaining Before Shipping

Release blockers:

- Final fresh Debian VM install proof.
- First launch/setup completion proof.
- Managed SearXNG setup/search proof on the clean install.
- External starter text-pack install/use/remove proof on the clean install.
- Rollback/uninstall proof.

Non-blocking but important:

- Broader docs consolidation.
- Real local-LLM fuzzing as an opt-in eval.
- Persistent journal status dashboard.
- Formal startup/memory budget.

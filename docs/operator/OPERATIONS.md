# Operations

This is the lightweight post-release operating guide for Personal Agent.
If another note disagrees, trust the live state surfaces plus `python -m agent doctor`.

## Quick Health Cadence

### After startup or deploy

Check these in order:

- `curl -sS http://127.0.0.1:8765/ready`
- `curl -sS http://127.0.0.1:8765/state`
- `curl -sS http://127.0.0.1:8765/packs/state`

Confirm:

- the runtime phase matches expectation
- `state_label`, `reason`, `next_step`, and `recovery` are present when the system is not healthy
- installed packs match what you expect to be present on that machine

### During normal operation

- spot-check `GET /ready` when behavior feels off
- check `GET /state` when the assistant sounds contradictory or stale
- check `GET /packs/state` when pack behavior seems wrong
- run `python scripts/release_smoke.py` if you suspect a regression in the core path

### After any issue

- re-check all three state surfaces
- confirm the recovery object matches the current condition
- only treat the system as recovered when the blocker has cleared and the state surface says so

## Incident Classification

Use one of three classes:

| Class | Meaning | Urgency | Typical response | Update release gate? |
| --- | --- | --- | --- | --- |
| A | Blocking | Immediate | Stop traffic or usage, diagnose, recover, or roll back | Yes, if it is a product regression |
| B | Degraded | Soon | Continue only if the degraded behavior is acceptable, then fix and verify | Usually, if it is new behavior |
| C | Cosmetic | Low | Fix wording, docs, or a non-critical inconsistency | Usually not, unless it hides real truth |

Rules of thumb:

- blocking means the product is not usable for its intended task
- degraded means partial functionality still works
- cosmetic means the truth is still correct, but the presentation is rough

## Incident Response Flow

### 1. Identify

Collect:

- `/ready`
- `/state`
- `/packs/state`
- `python -m agent doctor`
- `python -m agent version`

Record:

- `state_label`
- `reason`
- `next_step`
- `recovery.kind`
- any obvious pack blocker or runtime blocker

### 2. Classify

Decide whether the issue is:

- expected startup
- blocked
- degraded
- missing input or state
- a real bug

### 3. Act

Follow the surfaced `next_step` first.

Typical actions:

- wait for startup to finish
- restart the user service
- run `python -m agent doctor --fix`
- back up state before risky recovery
- roll back if a release regression is confirmed

### 4. Verify

After the fix:

- re-run `/ready`, `/state`, and `/packs/state`
- confirm there are no stale recovery objects
- confirm the state labels now match the actual machine state

## Support Bundle

For any bug report or support request, collect:

- `python -m agent version`
- `curl -sS http://127.0.0.1:8765/version`
- `curl -sS http://127.0.0.1:8765/ready`
- `curl -sS http://127.0.0.1:8765/state`
- `curl -sS http://127.0.0.1:8765/packs/state`
- `python -m agent doctor --collect-diagnostics`
- the recent action sequence:
  - install / remove
  - confirm / replay
  - restart / warmup
  - discovery changes

The doctor bundle is the canonical redacted support artifact. It is enough for
normal diagnosis without asking for logs first.

## Bug Triage Loop

When a bug is confirmed:

1. Classify it:
   - H1: malformed input / misuse
   - H3: transition legality / idempotency
   - H4: recovery wording or recovery object drift
   - H5: concurrency or contention
   - H7/H8: drift / soak / cleanup
2. Reproduce it from the support bundle.
3. Fix it minimally.
4. Add or update the regression test in the matching bucket.
5. Re-run `python scripts/release_gate.py`.

## Release Notes Discipline

Every release should record:

- what changed
- what was fixed
- behavior changes
- operator-impacting changes
- required actions, if any
- known issues, if any

Template: `docs/operator/RELEASE_NOTES_TEMPLATE.md`

## Live Smoke Promotion Criteria

Keep these as post-release operator checks until they are boring enough to
promote:

- `python scripts/webui_smoke.py`
- `python scripts/reference_pack_workflow_smoke.py`

Promotion criteria:

- passes reliably in at least 10 consecutive runs
- no environment-specific flakiness
- deterministic enough that false failures are rare
- stable on the actual release machine, not just in tests

Until then, they stay separate from the mandatory gate.

## Runbook Maintenance Rule

Any time behavior changes, update the runbook in the same change.

If a fix changes:

- a recovery path
- a status label
- a support bundle field
- a release gate step

then the relevant operator docs must change too.

## Lightweight Monitoring Mindset

Do not depend on a new dashboard before you can diagnose normal issues.
The first-line truth is already available through:

- `/ready`
- `/state`
- `/packs/state`
- `python -m agent doctor`

Logs are for deeper debugging. They should not be required for ordinary diagnosis.

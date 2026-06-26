# Operations

This is the lightweight post-release operating guide for Personal Agent.
If another note disagrees, trust the live state surfaces plus `python -m agent doctor`.

Daily-driver service:

- `personal-agent-api.service`

Checkout/dev service:

- `personal-agent-api-dev.service`

The stable service is the default local desktop app. The dev service is only
for repo work and must not compete with the stable launcher.

## Quick Health Cadence

### After startup or deploy

Check these in order:

- `curl -sS http://127.0.0.1:8765/ready`
- `curl -sS http://127.0.0.1:8765/state`
- `curl -sS http://127.0.0.1:8765/packs/state`
- `curl -sS http://127.0.0.1:8765/search/status`

Confirm:

- the runtime phase matches expectation
- `state_label`, `reason`, `next_step`, and `recovery` are present when the system is not healthy
- installed packs match what you expect to be present on that machine
- search is either available through a trusted loopback SearXNG backend or clearly blocked with one next action
- `/version` reports the expected runtime instance

### During normal operation

- spot-check `GET /ready` when behavior feels off
- check `GET /state` when the assistant sounds contradictory or stale
- check `GET /packs/state` when pack behavior seems wrong
- check `GET /search/status` before treating internet/search as available
- run `python scripts/chat_eval.py` when chat routing feels stale, contradictory,
  or dependent on manual daily-driver discovery
- run `python scripts/llm_behavior_eval.py` when end-to-end assistant wording or
  multi-turn behavior feels wrong but deterministic route classification passes
- run `python scripts/perf_smoke.py` when the assistant feels slow; it is
  read-only and reports generous latency warnings instead of flaky failures
- run `python scripts/release_smoke.py` if you suspect a regression in the core path
- run `python scripts/prove_ready.py` for the compact pre-VM readiness gate; it
  distinguishes release-blocking failures from optional-runtime warnings such
  as isolated search being disabled
- run `python scripts/prove_pre_vm_complete.py` before the expensive fresh VM
  proof; it adds backup/restore, Web UI robustness, and CI/live gate split
  checks
- run `python -m agent split_status` when you need a quick stable-vs-dev identity check

When a live chat route is wrong, capture it as an eval case before fixing it:
add a small JSON file under `tests/fixtures/bad_chat_cases/` with the user
message, expected semantic intent/route/kind, and any `must_not_contain` text.
Then run `python scripts/chat_eval.py`. Keep fixtures redacted and minimal.

Development runtimes can retain old blocked smoke-test external packs from earlier
manual or release-smoke runs. Treat them as local dev state when `/packs/state`
labels them `Installed · Blocked` with an explicit blocker such as executable
code. They are not release proof failures by themselves. Clean them only through
the confirmed pack remove/tombstone flow, or use a fresh state directory for
final install proof.

### After any issue

- re-check `/ready`, `/state`, `/packs/state`, and `/search/status`
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
- `/search/status`
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
- restart the stable or dev user service you are actually using
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

To check which copy is active:

- `curl -sS http://127.0.0.1:8765/version`
- `curl -sS http://127.0.0.1:18765/version`
- `systemctl --user status personal-agent-api.service`
- `systemctl --user status personal-agent-api-dev.service`
- `python -m agent doctor`

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

## Pre-VM Readiness Gate

Before the expensive fresh Debian VM proof, run:

- `python scripts/prove_ready.py`
- `python scripts/prove_pre_vm_complete.py`

`prove_ready.py` runs core compile checks, deterministic chat evals, release
smoke, daily-driver smoke, external-pack safety smoke, core workflow proof, and
whitespace checks. `prove_pre_vm_complete.py` adds the bounded backup/restore
proof, Web UI robustness smoke, CI/live gate matrix smoke, and subsystem audit
table. Treat `FAIL` as release-blocking. Treat isolated search `BLOCKED` as an
expected warning only when the command says search is disabled or unconfigured
rather than broken.

Gate split:

- CI-safe gates are documented in `docs/operator/RELEASE_GATE_MATRIX.md` and do
  not require local services.
- local-runtime gates require the installed Personal Agent API/runtime.
- optional integration gates require configured SearXNG, Telegram, local model,
  or similar services and must report `BLOCKED` when those are absent.

## Live Smoke Promotion Criteria

Keep these as post-release operator checks until they are boring enough to
promote:

- `python scripts/prove_core_workflows.py`
- `python scripts/webui_smoke.py`
- `python scripts/reference_pack_workflow_smoke.py`
- `bash scripts/promote_local_stable.sh`
- `python scripts/split_smoke.py`

Promotion criteria:

- passes reliably in at least 10 consecutive runs
- no environment-specific flakiness
- deterministic enough that false failures are rare
- stable on the actual release machine, not just in tests

Until then, they stay separate from the mandatory gate.

## Safe Web Search / SearXNG

Check search with:

- `GET /search/status`

If search is missing, preview setup with:

- `POST /search/setup/plan`
- `POST /search/setup/apply` with the returned confirmation token
- `POST /search/setup/prerequisite/plan` for the narrow Podman prerequisite
  plan when Podman is missing
- `POST /search/setup/prerequisite/apply` with the returned confirmation token

The managed setup path accepts only loopback SearXNG URLs or the approved local
`personal-agent-searxng` container plan. It must not install Podman, Docker,
SearXNG, or system packages silently. On Linux, rootless Podman is preferred;
when Podman is missing, the default setup path previews a Podman prerequisite
install for the `podman` package only and requires confirmation before running
the prerequisite path. If privilege is required, apply returns an elevated
terminal handoff for `sudo apt-get install -y podman`; the background API
service must not run hidden sudo or store sudo passwords. Docker appears only as
an explicit fallback plan with a warning, fallback reason, and Docker fallback
confirmation flag. The first managed SearXNG container seeds and validates the
approved owned `settings.yml` before mounting `/etc/searxng`; empty config
mounts and arbitrary settings content are rejected. The seeded config enables
JSON output for metadata-only safe search and creates or preserves a
non-default `server.secret_key`; the inherited `ultrasecretkey` value is
rejected and the key is redacted from journals, diagnostics, and support
output. If the approved config directory is not writable by the service user,
setup stops before pull/run and returns a bounded ownership handoff instead of
running hidden sudo. If the approved container already exists, setup reuses or
restarts it only after inspect confirms the approved image, loopback bind, and
config mount; mismatches are blocked and not removed automatically. It waits up
to 30 seconds for first boot, accepts HTTP 200 as healthy, and retries with
`GET` when `HEAD` does not prove readiness. If health fails, it captures
redacted `ps -a` and `logs --tail 120`
diagnostics for `personal-agent-searxng` before rolling back that owned
container. It updates the running Personal Agent search configuration only
after the SearXNG JSON endpoint verifies, then persists a small loopback-only
runtime search config next to the runtime database. Explicit service
environment variables still override that persisted state when set.

Search lifecycle:

- `never_configured`: first public lookup previews trusted local SearXNG setup;
  no first-time install/start runs without confirmation.
- `configured_running`: public lookup uses metadata-only search immediately.
- `configured_stopped`: trusted search config survived restart/promotion/reboot
  but the local SearXNG endpoint is not reachable. A public lookup should offer
  inline managed start/repair confirmation and then continue after confirmation;
  it must not ask the user to perform a separate manual start ritual.
- `invalid_or_untrusted_config`: refuse use and preview safe reconfiguration.

Personal Agent does not currently install a SearXNG reboot autostart unit.
After a PC reboot, managed search may enter `configured_stopped` until the user
confirms the inline start/repair plan or configures their container runtime to
auto-start the approved `personal-agent-searxng` container.

## Core Workflow Proof

Run this before any real acceptance claim:

- `python scripts/prove_core_workflows.py`

Read the report literally. `PASS` means the script observed the runtime path and
state change. `BLOCKED` means the current machine is missing required runtime
configuration, such as a ready provider/model or trusted SearXNG endpoint.
`NOT_PROVEN` means a claim still needs a separate direct command.

The behavior/release gate remains authoritative when run directly:

- `python -m pytest -q tests/test_chat_behavior_audit.py tests/test_live_user_barrage.py tests/test_assistant_behavior_release_gate.py`

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

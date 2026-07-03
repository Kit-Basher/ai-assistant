# P0 Reliability Recap

This is an operator checkpoint for the five P0 reliability batches from
`RELIABILITY_COVERAGE_GAP_AUDIT.md`. It is not a final release claim and does
not replace the fresh Debian VM proof.

## Completed Clean Tags

1. `v0.2.1-search-lifecycle-fault-injection-clean`
   - Search lifecycle fault injection.
2. `v0.2.1-telegram-lifecycle-fault-injection-clean`
   - Telegram stale-lock and duplicate-poller reliability.
3. `v0.2.1-secret-store-reliability-clean`
   - Secret-store corruption and redaction reliability.
4. `v0.2.1-executor-failure-journal-boundedness-clean`
   - Executor partial-artifact failure handling and journal boundedness.
5. `v0.2.1-plan-mode-confirmation-matrix-clean`
   - Plan Mode stale-confirmation matrix across major action families.

## Guarantees Now Covered

- Search failure modes are predictable for never-configured, stopped,
  unreachable, HTML/non-JSON, malformed JSON, timeout, unsafe setup URL, and
  local-planning prompt cases. Search remains metadata-only.
- Telegram remains optional and reports missing token, stopped service, stale
  dead-PID lock, live lock, duplicate poller, and token-redaction states without
  making the web/API assistant look broken.
- Secret-store missing/corrupt/decrypt-failed states are structured in status
  surfaces and chat. Redacted reads remain available; raw secret exposure is
  refused in chat.
- Executor failures return structured results. Partial support/backup artifacts
  are recorded with scoped rollback hints, and executor journals are bounded and
  redacted.
- Plan Mode confirmations are bound to the current pending plan and thread.
  Wrong-thread, cancelled, expired, overwritten, no-plan, and preview-only
  confirmations do not execute unsafe actions.

## Bugs Fixed

- Search no longer falls through to false manual setup or stale web-search
  advice for configured states.
- Telegram stale locks and duplicate poller evidence are visible and redacted.
- Corrupt secret-store state no longer looks like a normal missing Telegram
  token.
- Raw-token chat prompts now get deterministic refusal/status wording.
- Support bundle failures before final manifest no longer look successful.
- Oversized executor journal records are compacted instead of recursively
  bloating support/backup summaries.
- `/confirm` now rejects expired pending confirmations.
- `confirm`, `go ahead`, and `proceed` with no current plan now return a clear
  no-current-action response.

## Trusted Checkpoint Gates

Use these as the current operator truth for this checkpoint:

```bash
bash scripts/promote_local_stable.sh
python scripts/installed_product_abuse.py
python scripts/daily_driver_maturity_audit.py
python scripts/executor_registry_smoke.py
python scripts/support_bundle_v2_smoke.py
python scripts/backup_v1_smoke.py
python scripts/release_smoke.py
python scripts/prove_ready.py
python scripts/docs_truth_smoke.py
python scripts/release_gate_matrix_smoke.py
git diff --check
git status
```

`prove_ready.py` remains the canonical pre-VM readiness gate. Expected warnings
can still include runtime latency drift and isolated proof limits; release
blockers are the important value.

## Daily-Driver Irritants

At the P0 recap checkpoint, the daily-driver maturity audit had one known
irritant:

- `rewrite this: search for dots.tts`

The tiny response-quality polish batch after the recap fixed this path. Current
expected response:

```text
Search for dots.tts.
```

This was response-quality polish, not a reliability blocker. It did not
indicate unsafe mutation, search leakage, or stale confirmation execution.

## Real-Use Journey Gap

The P0 reliability batches prove safety gates and focused fault handling; they
do not mean every natural user journey has been exercised through the installed
web or Telegram surfaces. The first post-P0 real-use checks found two journey
gaps:

- Telegram messages cannot be answered when the optional
  `personal-agent-telegram.service` is stopped, even if the token is configured.
- Casual model/provider questions mentioning Ollama must answer status, not
  switch models.

`python scripts/real_use_journey_smoke.py` is the read-only installed-product
audit for these paths. It verifies web chat greeting, casual Ollama status
wording, immediate `why` follow-up context, `/telegram/status`, and Telegram
poller/service truth. Deterministic Telegram transport tests prove
incoming text -> local `/chat` payload -> outbound reply. Full live Telegram
send/receive remains optional/manual because it requires a real Telegram chat,
token, and network path.

## Remaining P1/P2 Gaps

- Installed-product search repair failure proof for configured-stopped repair
  failures and port-conflict wording.
- Installed Telegram service start/restart/stop failure proof for systemctl
  failure and live-host duplicate-poller evidence.
- Promoted-runtime corrupt secret-store fixture proof for support bundle and
  backup summaries.
- Executor journal retention/rotation policy for long-running daily use.
- Optional installed-product Plan Mode restart matrix across every action
  family; deterministic Batch 5 covers the core state machine.
- Fresh Debian VM proof remains intentionally not run.

## Recommended Next Step

The tiny response-quality polish batch is complete. Stop here and use the
system for real daily-driver observation before starting another broad lane.

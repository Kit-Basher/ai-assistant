# Project State

This is the operator truth snapshot for the current local checkpoint. It is not
marketing copy and it is not a final release claim.

## Current Checkpoint

- Tag: `v0.2.1-search-recovery-product-pass`
- Commit: `c7e2e2d`
- Fresh Debian VM proof: not run
- Release status: ready for VM proof, not finished

Current confirmed proof:

- `python scripts/installed_product_abuse.py`: `PASS=42 WARN=0 FAIL=0`
- `python scripts/prove_daily_driver_product.py`: `PASS`
- `python scripts/daily_driver_smoke.py --timeout 90`: `PASS=9 BLOCKED=0 FAIL=0`
- `python scripts/prove_pre_vm_complete.py`: `PRE_VM_COMPLETE=yes`, `BLOCKERS=0`, `UNKNOWN_AREAS=0`, `WARNINGS=7`
- `python scripts/prove_ready.py`: `READY_FOR_VM_PROOF=yes`, `RELEASE_BLOCKERS=0`, `WARNINGS=2`
- `git status`: clean at the checkpoint

New restart/browser survival lane:

- `python scripts/restart_survival_smoke.py`: automated stable API service
  restart proof. It is not a full PC reboot proof.
- `docs/operator/REBOOT_PROOF.md`: manual reboot and browser survival
  checklist.

Earlier readiness gates passed before the first real web UI search request
failed. That failure proved the repo needed an installed-product gate, not only
internal and mock-heavy tests. `installed_product_abuse.py` and
`prove_daily_driver_product.py` are now the stronger product-facing gates.

## Meaning Of Gates

- `scripts/prove_ready.py`: canonical pre-VM readiness gate. It separates
  release-blocking failures from runtime-state warnings and expected isolated
  proof limits.
- `scripts/prove_pre_vm_complete.py`: broader local subsystem gate. It asks
  whether the local runtime is hardened enough that the fresh VM proof should
  be confirmation, not discovery.
- `scripts/installed_product_abuse.py`: strict installed-runtime abuse harness.
  It talks to the promoted API surface, verifies runtime freshness, checks
  endpoint wiring, and drives confused user flows for search, Telegram, memory,
  and Plan Mode.
- `scripts/prove_daily_driver_product.py`: wrapper around the installed-product
  abuse harness for the current daily-driver product proof.
- `scripts/daily_driver_smoke.py`: user-facing smoke for ready/state/search,
  pack use, package preview, ordinary chat, and doctor. It can repair
  configured-stopped managed search through the assistant Plan Mode flow.
- `scripts/release_smoke.py`: deterministic release smoke for core behavior.
- `scripts/perf_smoke.py`: read-only latency and no-LLM deterministic route
  check. Small latency warnings are not release blockers by themselves.
- `scripts/restart_survival_smoke.py`: installed stable service stop/start
  proof. It verifies runtime freshness, status surfaces, managed search repair,
  metadata-only search after restart, Telegram optional wording, and stale
  confirmation rejection across service restart.

## Proven Now

- The promoted stable runtime reports build metadata and can be compared with
  the checkout commit.
- The installed API surface exists for the routes used by the web UI and
  operator proof scripts.
- Managed local SearXNG setup and recovery are confirmation-gated and
  product-facing.
- `configured_stopped` SearXNG state repairs through assistant Plan Mode and
  then continues the original lookup.
- Search remains metadata-only: no page fetch, browser automation, downloads,
  JavaScript, or pack install from search.
- Podman detection works from the stable service, including approved absolute
  paths such as `/usr/bin/podman`.
- Telegram configured-but-inactive is reported as optional, not whole-agent
  failure.
- Telegram start/restart/stop prompts route to bounded Plan Mode previews.
- Common prompt-injection and ambiguous action prompts do not bypass Plan Mode.
- Stale confirmations and missing confirmations are rejected cleanly.
- External pack safety smoke still passes and packs remain text-only unless
  separately approved through explicit lifecycle gates.
- Daily-driver smoke passes with installed managed search running.
- The stable API service can survive a controlled service stop/start in the
  automated restart smoke. Actual PC reboot remains a separate manual proof.

## Still Partial

These are not unknowns, but they are not finished:

- Installer/update/uninstall: install, promotion, bundle, and package paths have
  coverage; fresh-host partial-failure recovery and uninstall proof remain.
- Storage/log growth: growth surfaces are documented; a single read-only
  `storage_status` and cleanup preview flow remain future work.
- Web UI robustness: static/component smoke exists; browser automation and
  manual refresh/large-transcript checks remain. `REBOOT_PROOF.md` carries the
  manual UI checklist.
- Telegram runtime behavior: optional-service semantics and status UX are
  covered; full start/stop/restart execution proof remains partial.
- Memory completion: audits and safety checks exist; deterministic
  memory-status, explain, and forget-X UX remain partial.
- Release/CI automation: CI-safe and live-runtime gates are split; broader CI
  adoption remains future work.
- Model/provider management: deterministic guidance and switching paths are
  covered; opt-in real local LLM soak remains future work.
- Fresh Debian VM proof: intentionally not run yet.

## What Not To Claim

- Do not claim the agent is finished.
- Do not claim final release readiness.
- Do not claim the fresh Debian VM proof is complete.
- Do not claim full memory completion.
- Do not claim full installer/update/uninstall lifecycle completion.
- Do not claim bug-free behavior.
- Do not claim all external skills execute or are complete.
- Do not claim full web browsing; current search is metadata-only.

Safe wording:

- “ready for VM proof”
- “installed daily-driver product gate passes”
- “not finished”
- “release candidate pending VM proof and lifecycle completion”

## Next Release Lanes

1. Manual reboot proof.
2. Automated browser/UI survival proof.
3. Operator lifecycle: status, storage, backup, restore, update, uninstall,
   repair.
4. Memory explain/forget/status completion.
5. Plan Mode v2 canonical action layer.
6. Skill pack lifecycle hardening.
7. Model/provider management and real local LLM soak.
8. Clean VM proof.
9. Release candidate.

## Operator Quick Commands

```bash
bash scripts/promote_local_stable.sh
python scripts/installed_product_abuse.py
python scripts/prove_daily_driver_product.py
python scripts/daily_driver_smoke.py --timeout 90
python scripts/restart_survival_smoke.py
python scripts/prove_pre_vm_complete.py
python scripts/prove_ready.py
git status
```

## Search Lifecycle Truth

- `never_configured`: no trusted endpoint is configured. First public lookup
  offers local SearXNG setup and requires confirmation.
- `configured_running`: trusted endpoint is configured and JSON metadata search
  works. Public lookups search immediately.
- `configured_stopped`: trusted endpoint is configured but unreachable. Public
  lookups offer inline managed start/repair confirmation. After approval,
  Personal Agent repairs only the trusted managed endpoint, rechecks status,
  and continues the original lookup if search becomes available.
- `invalid_or_untrusted_config`: persisted or configured endpoint failed trust
  checks. The runtime refuses to use it and offers safe reconfiguration.

Search is not a browser. It returns untrusted SearXNG metadata only.

## Telegram Truth

- Telegram is optional.
- Configured but inactive Telegram is not a whole-agent failure.
- Start/restart/stop are bounded Plan Mode actions.
- Telegram tokens must remain redacted from status, chat, logs, docs, and
  support output.

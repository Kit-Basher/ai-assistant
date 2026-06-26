# PRE-VM Completion Audit

Date: 2026-06-20

This audit is stricter than `READY_FOR_VM_PROOF`. It asks whether the local
runtime is hardened enough that the fresh Debian VM proof should be final
confirmation, not a discovery phase.

Current result: **PRE_VM_COMPLETE: no**.

The project has zero known release-readiness blockers in `prove_ready.py`, but
several local subsystems are still partial or unknown. Do not start the fresh VM
proof until the unknown areas below are resolved or explicitly accepted.

## Command

Run:

```bash
python scripts/prove_pre_vm_complete.py
```

The command runs or summarizes:

- `python scripts/prove_ready.py`
- `python scripts/backup_restore_proof.py`
- `python scripts/chat_eval.py`
- `python scripts/llm_behavior_eval.py`
- `python scripts/perf_smoke.py`
- `python scripts/release_smoke.py`
- `python scripts/daily_driver_smoke.py --timeout 90`
- `python scripts/external_pack_safety_smoke.py`
- `python scripts/prove_core_workflows.py`
- `git diff --check`

It prints:

- `PRE_VM_COMPLETE: yes/no`
- `BLOCKERS`
- `WARNINGS`
- `UNKNOWN_AREAS`
- a subsystem status table
- next actions

## Subsystem Audit

| Subsystem | Status | Blocker | Unknown | Notes |
| --- | --- | --- | --- | --- |
| Backup/restore | hardened | no | no | Bounded proof covers valid archive validation, dry-run restore, corrupt archive refusal, strict version mismatch refusal, secret redaction in output, and restore into temp state without live mutation. |
| Installer/update/uninstall | partial | no | no | Install, promotion, bundle, and Debian package paths have tests, but fresh host partial-failure recovery and final uninstall proof are deferred. |
| Storage/log growth | partial | no | no | Growth surfaces are documented. There is no single read-only `storage_status` command or enforced cleanup policy. |
| Observability/debuggability | hardened | no | no | Trace ids, route/timing fields, eval reports, doctor, and support bundle redaction exist. Journal dashboard remains future work. |
| Web UI robustness | partial | no | yes | Build and autoscroll are covered, but refresh, retry, large transcript, stale cache, and transcript import/export behavior are not broadly proven. |
| Telegram runtime behavior | partial | no | no | Optional-service semantics and token redaction are covered. Start/stop/restart Plan Mode golden path remains unproven. |
| Model/provider management | hardened | no | no | Provider/model guidance, model switch journaling, stale follow-up escape, and timeout/fallback behavior have automated coverage. |
| Memory completion | partial | no | no | Memory audit, no-memory override, secret redaction, and scoped clears exist. Forget-X and memory explainability UX remain partial. |
| Security/capabilities | hardened | no | no | Plan Mode, external pack gates, metadata-only search, localhost managed services, and arbitrary shell/podman blocking are covered. |
| Release/CI automation | partial | no | yes | Local gates exist. CI feasibility and fresh VM proof remain unknown/deferred. |

## Blockers

None currently known after `scripts/backup_restore_proof.py`.

## Warnings

- Search may be disabled in isolated proof environments. That is not a release
  blocker when `/search/status` reports `search_disabled` honestly and managed
  SearXNG setup remains confirmation-gated.
- `perf_smoke.py` can warn on small runtime-state variance, especially `/ready`.
  Deterministic chat routes must remain no-LLM and should keep post-response
  memory hooks at `0ms`.
- Daily-driver search remains `BLOCKED` until trusted SearXNG is configured in
  the target runtime.

## Unknown Areas

- Web UI robustness beyond build/autoscroll.
- CI feasibility for the full local-runtime gate matrix.

## Next Actions

1. Add read-only `storage_status`.
2. Add Web UI robustness smoke for retry, refresh, stale cache, large transcript,
   and disabled-search display.
3. Split CI-safe deterministic gates from live local-runtime gates.
4. Then rerun `python scripts/prove_pre_vm_complete.py`.

Do not use this audit to claim final release readiness. Fresh Debian VM proof
is still required after PRE_VM_COMPLETE becomes `yes`.

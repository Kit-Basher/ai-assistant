# Test Suite Rationalization

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.
Current full-pytest failure classification lives in
`docs/operator/PYTEST_FAILURE_TRIAGE.md`.
Subsystem reliability guarantees and missing fault-injection coverage live in
`docs/operator/RELIABILITY_COVERAGE_GAP_AUDIT.md`.

This document maps the Personal Agent tests and proof scripts so the project can
avoid adding overlapping gates by default. It is an operator classification, not
a request to delete coverage immediately.

## Inventory Summary

- Top-level pytest modules: 266.
- Smoke/proof/eval scripts by naming pattern: 41.
- Primary live installed-product gates: 16.
- Primary CI-safe deterministic gates: 9.
- Current full `python -m pytest -q` audit result from this pass:
  `90 failed, 2298 passed` in about 19 minutes. The failures are spread across
  older workflows, LLM/provider expectations, safe-mode transcripts, pack
  acquisition, and environment-sensitive search/setup paths. Treat full pytest
  as an inventory sweep until those stale expectations are triaged; it is not
  the current canonical release blocker.
- Follow-up short-traceback triage result:
  `108 failed, 2280 passed` in about 15 minutes. The count changed between
  runs, confirming order/environment sensitivity in the full inventory.

The suite has good safety coverage, but it has accumulated overlapping proof
lanes. The main risk is not missing tests; it is running too many similar gates
and treating every small wording failure as a release blocker.

The reliability coverage audit is the guardrail for future test additions: add
coverage to close a named subsystem guarantee or fault-injection gap, not just
because another phrasing failed once.

## Test Module Groups

| Group | Files | Purpose | CI-safe | Release blocker | Decision |
| --- | ---: | --- | --- | --- | --- |
| Orchestrator/chat behavior | `test_orchestrator.py`, `test_chat_behavior_audit.py`, `test_live_user_barrage.py`, `test_assistant_behavior_release_gate.py`, `test_adversarial_chat_routing.py` | Protect deterministic routing, Plan Mode safety, search/pack/status behavior, and messy chat inputs. | Yes | Yes for focused gate | Keep; do not add every new phrase here unless it protects an invariant. |
| Release gates | `test_release_gate.py`, `test_release_smoke.py`, `test_release_bundle.py`, `test_release_validation_extended.py` | Build/release contract checks. | Yes | Yes for release paths | Keep; prefer one canonical release smoke command. |
| Safe web search and managed service | `test_safe_web_search.py`, `test_search_setup_ux.py`, `test_managed_local_services.py`, `test_persistent_managed_action_journal.py` | SearXNG lifecycle, localhost-only safety, Podman/Docker selection, journals. | Yes with mocks | Yes for search safety | Keep as standalone safety gate. |
| Plan Mode and executors | `test_plan_policy.py`, `test_executor_registry.py`, lifecycle preview tests | Confirmation, preview-only refusal, journals, additive executors. | Yes | Yes | Keep as standalone safety gate. |
| External pack lifecycle | `test_api_packs_endpoints.py`, `test_api_pack_sources_endpoints.py`, `test_external_pack_ingestion.py`, pack lifecycle/source/permission tests | Text-pack safety, quarantine/review/enable/grant/remove boundaries. | Yes | Yes | Keep; external pack smoke remains a separate release blocker. |
| Memory and working memory | memory/working-memory/thread/prefs tests | Durable memory, transient state, no-memory behavior, scope separation. | Yes | Partial | Keep; route user-facing memory lifecycle through installed smoke, keep internals in pytest. |
| Runtime truth/readiness/doctor | ready/state/recovery/doctor/runtime contract tests | Truth surfaces and optional-service wording. | Yes | Yes for `/ready`/doctor basics | Keep; avoid duplicating every installed smoke case here. |
| LLM/model/provider | 29 `test_llm_*`, model/provider/model-scout tests | Provider registry, setup flow, model scout, notifications, local model behavior. | Mostly yes | Some are feature gates, not all release gates | Split later into model provider gate vs historical modelops tests. |
| Telegram | 9 Telegram tests | Token loading, optional service state, bridge, router parity. | Mostly yes | Token redaction/status yes; live bridge optional | Keep unit tests; live Telegram smokes stay optional. |
| Backup/restore/support/cleanup | `test_backup_restore_proof.py`, executor tests, smoke tests | Additive backup, restore validator, support bundle redaction, cleanup preview and isolated cleanup execution. | Mostly yes | Yes for redaction/no-mutation | Keep as operator safety gate; avoid adding another backup proof lane. |
| Web UI/desktop | webui/desktop/launcher tests | Build, launcher, conversation smoke, browser survival, latency bundle. | Mostly yes | Build yes; browser proof is installed-product only | Keep build/static checks in readiness; run browser proof after promotion or before UI/restart checkpoints. |
| Soak/stress/regression | concurrency, soak, barrage, extended tests | Find flakes, load issues, long-tail regressions. | Mixed | No by default | Occasional/manual; do not put in the fast loop. |

## Named Proof Scripts

| Script | Purpose | Runtime cost | Requires installed product | Mutates | CI-safe | Release blocker | Daily-driver detector | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `scripts/prove_ready.py` | Canonical pre-VM readiness wrapper. | Slow, minutes | Partly | Runs read-only and live smokes; backup/support may create bounded artifacts through child gates only when included. | No | Yes | Yes | Keep as full local release proof. |
| `scripts/release_smoke.py` | Deterministic core pytest smoke. | Slow, about 1-2 minutes | No | No | Yes | Yes | No | Keep as CI-safe release blocker. |
| `scripts/chat_eval.py` | Large deterministic chat-routing invariant eval. | Fast | No | No | Yes | Yes | Yes | Keep; do not replace installed gates. |
| `scripts/llm_behavior_eval.py` | Mocked e2e conversation invariant eval. | Fast | No | No | Yes | Yes | Yes | Keep; expand by replacing duplicate phrase tests, not by adding new lanes. |
| `scripts/docs_truth_smoke.py` | Forbidden release-claim/doc truth scan. | Fast | No | No | Yes | Yes | No | Keep. |
| `scripts/release_gate_matrix_smoke.py` | Verifies gate docs and CI/live split. | Fast | No | No | Yes | Yes for docs changes | No | Keep. |
| `scripts/external_pack_safety_smoke.py` | Core external text-pack safety proof. | Fast | No | Uses temp state only | Yes | Yes | No | Keep standalone. |
| `scripts/prove_core_workflows.py` | Historical core workflow proof. | Medium | No | Temp/local proof state | Yes | Yes, but search BLOCKED expected when no backend | Partial | Keep, but do not treat expected isolated search BLOCKED as failure. |
| `scripts/daily_driver_smoke.py` | Installed happy-path daily-driver smoke. | Medium | Yes | May repair managed search through confirmation path | No | Yes on release machine | Yes | Keep; default installed product check. |
| `scripts/installed_product_abuse.py` | Strict installed API abuse harness. | Medium | Yes | Can confirm bounded search repair; otherwise avoids unsafe mutation | No | Yes on release machine | Yes | Keep; primary product-facing truth gate. |
| `scripts/prove_daily_driver_product.py` | Wrapper around installed product abuse. | Medium | Yes | Same as child script | No | Yes on release machine | Yes | Keep as human-friendly product gate. |
| `scripts/daily_driver_maturity_audit.py` | Recurring daily-driver audit with blockers vs irritants. | Medium | Yes | Preview-only except safe product interactions; does not confirm enabled mutations | No | No by itself | Yes | Keep; merge future friction checks here rather than adding new proof scripts. |
| `scripts/perf_smoke.py` | Latency/no-LLM deterministic route check. | Medium | Yes | No | No | Warning-only unless extreme | Yes | Keep warning-only; do not block release on minor latency. |
| `scripts/restart_survival_smoke.py` | Stable API stop/start survival. | Slow/disruptive | Yes | Stops/starts user API service; may repair search | No | Yes before VM proof | Yes | Keep as occasional local proof, not fast loop. |
| `scripts/first_run_smoke.py` | Isolated fresh HOME/state first-run proof. | Medium | No installed stable needed | Temp dirs/process only | Mostly | Yes | No | Keep. |
| `scripts/browser_ui_survival_smoke.py` | Installed browser/UI survival proof using Playwright and system Chrome. | Slow/disruptive | Yes | Stops/starts user API service; no unsafe mutation | No | Yes before UI/restart checkpoint | Yes | Keep as standalone live-runtime proof; do not put in fast loop. |
| `scripts/webui_robustness_smoke.py`, `scripts/webui_smoke.py` | Web UI build/static/API-root robustness proof. | Medium | Mixed | No | Mixed | Partial | Yes | Keep lightweight; pairs with browser survival smoke. |
| `scripts/operator_lifecycle_smoke.py` | Installed operator prompt preview lane. | Medium | Yes | Preview-only | No | Yes for operator safety | Yes | Keep, but consider folding its read-only checks into maturity audit later. |
| `scripts/memory_lifecycle_smoke.py` | Installed memory lifecycle preview lane. | Medium | Yes | Preview-only | No | Yes for memory safety | Yes | Keep until memory executor grows; then revisit. |
| `scripts/plan_mode_v2_smoke.py` | Installed Plan Mode v2 UX/safety lane. | Medium | Yes | Preview-only refusals; service restart check may be delegated elsewhere | No | Yes | Yes | Keep as standalone safety gate. |
| `scripts/executor_registry_smoke.py` | Installed executor registry result/refusal lane. | Medium | Yes | Support-bundle executor may create temp artifact | No | Yes | No | Keep as standalone safety gate. |
| `scripts/support_bundle_v2_smoke.py` | Additive support bundle packaging proof. | Medium | Yes | Creates temp support bundle | No | Yes for diagnostics packaging | No | Keep under operator safety group. |
| `scripts/backup_v1_smoke.py` | Additive Backup v1 proof. | Medium | Yes | Creates bounded backup artifact | No | Yes for backup lane | No | Keep; do not run in every fast loop. |
| `scripts/cleanup_preview_smoke.py` | Cleanup preview and cancel-no-delete proof. | Medium | Yes | No | No | Yes for cleanup safety | Yes | Keep under operator safety group. |
| `scripts/cleanup_execution_smoke.py` | Cleanup execution proof against isolated generated fixture plus installed preview/cancel check. | Medium | Partly; installed API needed for preview section | Isolated fixture only | Yes for preview section | Yes for cleanup execution safety | Yes | Keep under operator safety group; do not delete live daily-driver artifacts in proof. |
| `scripts/restore_validator_smoke.py` | Read-only restore validator proof. | Medium | Yes | Temp malformed fixture only | No | Yes for restore safety | No | Keep under operator safety group. |
| `scripts/restore_execution_smoke.py` | Restore Executor v1 proof for validated Backup v1 fixture restore, staging, safety snapshot, and rollback. | Medium | No installed stable needed | Isolated fixture state only | Yes | Yes for restore execution safety | No | Keep under operator safety group; never restore daily-driver state in this proof. |
| `scripts/prove_pre_vm_complete.py` | Broad pre-VM subsystem gate. | Slow | Mixed | Runs summaries/child gates | No | Yes before VM proof | No | Keep as periodic; not daily loop. |
| `scripts/vm_proof_smoke.py` | Post-install VM verifier. | Medium | Clean VM installed product | No destructive mutation | No | Manual VM proof only | No | Keep manual/historical until VM lane starts. |
| Legacy/live smoke scripts | `assistant_*_smoke.py`, `live_*`, `telegram_*`, `hardware_*`, `provider_matrix_smoke.py`, `split_smoke.py`, `brief_smoke.py` | Mixed | Mixed | Mixed | Mixed | No by default | Sometimes | Historical/manual unless a current gate names them. |

## Duplicate Or Overlapping Areas

- `daily_driver_smoke.py`, `installed_product_abuse.py`, and
  `daily_driver_maturity_audit.py` all hit installed `/chat`. Keep all three
  only because they answer different questions: happy path, abuse path, and
  long-term irritants.
- `operator_lifecycle_smoke.py`, `support_bundle_v2_smoke.py`,
  `backup_v1_smoke.py`, `cleanup_preview_smoke.py`, and
  `restore_validator_smoke.py` overlap on operator prompts. Keep the focused
  scripts as safety gates, but do not add another generic operator smoke.
- `plan_mode_v2_smoke.py` and `executor_registry_smoke.py` overlap on
  confirmation safety. Keep both: Plan Mode proves UX/confirmation semantics;
  Executor Registry proves apply/result/journal semantics.
- `release_smoke.py`, `prove_ready.py`, and `prove_pre_vm_complete.py` overlap
  intentionally. Treat `release_smoke.py` as CI-safe core, `prove_ready.py` as
  canonical local readiness, and `prove_pre_vm_complete.py` as broader periodic
  subsystem audit.
- Many phrase-level tests in `test_orchestrator.py`,
  `test_chat_behavior_audit.py`, and `test_adversarial_chat_routing.py` overlap
  with `chat_eval.py`. Future weird-input regressions should become generated
  eval cases first, then only add a unit test if a helper or safety branch needs
  direct coverage.

## Tests That Are Too Broad

- `tests/test_orchestrator.py` is the biggest rabbit-hole risk. It is valuable,
  but broad phrase assertions can encourage patching isolated wording. Prefer
  semantic assertions: route, tool used, mutation status, and banned unsafe text.
- `scripts/prove_ready.py` is intentionally broad. Do not add new child scripts
  to it unless they are release blockers; otherwise `prove_ready` becomes slow
  enough that it stops being used.
- `scripts/prove_pre_vm_complete.py` is broad by design. Use it as an occasional
  checkpoint, not the default local loop.

## Tests That Are Too Narrow

- Single-phrase regressions for messy chat should usually move into
  `chat_eval.py` or `daily_driver_maturity_audit.py` unless they exercise a
  specific deterministic branch.
- Historical live smoke scripts should not be promoted to default gates without
  a clear user-visible invariant.
- Implementation-detail tests that check exact helper internals should stay near
  the module they protect and should not be duplicated by installed-product
  scripts.

## Recommended Command Groups

### 1. Fast Local Dev Check

Use while editing a narrow behavior path:

```bash
python -m pytest -q tests/test_orchestrator.py tests/test_chat_behavior_audit.py
python scripts/chat_eval.py
python scripts/docs_truth_smoke.py
git diff --check
```

For non-chat work, swap in the subsystem tests, for example:

```bash
python -m pytest -q tests/test_executor_registry.py tests/test_safe_web_search.py
```

### 2. Behavior Gate

Use before considering a behavior change done:

```bash
python scripts/chat_eval.py
python scripts/llm_behavior_eval.py
python -m pytest -q tests/test_chat_behavior_audit.py tests/test_live_user_barrage.py tests/test_assistant_behavior_release_gate.py
python scripts/external_pack_safety_smoke.py
```

### 3. Installed Daily-Driver Check

Use after promoting stable runtime or changing user-facing chat behavior:

```bash
bash scripts/promote_local_stable.sh
python scripts/installed_product_abuse.py
python scripts/prove_daily_driver_product.py
python scripts/daily_driver_smoke.py --timeout 90
python scripts/daily_driver_maturity_audit.py
python scripts/browser_ui_survival_smoke.py
```

### 4. Operator Safety Check

Use after changing Plan Mode, executors, backup/restore/support/cleanup, or
memory/operator lifecycle wording:

```bash
python scripts/plan_mode_v2_smoke.py
python scripts/executor_registry_smoke.py
python scripts/operator_lifecycle_smoke.py
python scripts/memory_lifecycle_smoke.py
python scripts/support_bundle_v2_smoke.py
python scripts/backup_v1_smoke.py
python scripts/cleanup_preview_smoke.py
python scripts/cleanup_execution_smoke.py
python scripts/restore_validator_smoke.py
python scripts/restore_execution_smoke.py
```

### 5. Full Local Release Proof

Use before tagging or before the fresh VM proof:

```bash
python scripts/docs_truth_smoke.py
python scripts/release_gate_matrix_smoke.py
python scripts/release_smoke.py
python scripts/prove_ready.py
git diff --check
git status
```

Run the full pytest inventory separately when auditing stale tests:

```bash
python -m pytest -q
```

Do not block a local release checkpoint on full pytest until the 90 failing
legacy/broad tests are classified or repaired. The current release blocker is
`prove_ready.py`, plus the relevant focused pytest files for the touched
subsystem.

### 6. Historical/Manual Proofs

Run only when the relevant subsystem is being worked or before a major release:

```bash
python scripts/prove_pre_vm_complete.py
python scripts/restart_survival_smoke.py
python scripts/browser_ui_survival_smoke.py
python scripts/vm_proof_smoke.py --expected-commit <commit>
python scripts/telegram_bridge_smoke.py
python scripts/provider_matrix_smoke.py
python scripts/hardware_discovery_smoke.py
```

## Release Blocker Policy

Release blockers:

- CI-safe release smoke failures.
- Plan Mode mutation/confirmation bypass.
- Search claiming availability when `/search/status` disagrees.
- External pack safety gate failures.
- Token/secret leakage in status, chat, support bundle, or backup output.
- Installed product abuse failures on the release machine.
- Backup/support/cleanup/restore mutating outside their scoped additive or
  preview-only policies.

Daily-driver irritants:

- Wordy but safe Plan Mode previews.
- Mild deterministic route latency warnings.
- Stale or awkward but non-dangerous wording.
- Search configured-stopped repair prompts that are correct but tedious.
- State growth warnings with safe cleanup preview available.

Historical/manual:

- Hardware/provider matrix tests.
- Telegram bridge live delivery tests.
- VM proof smoke before the clean VM lane is active.
- Long soak/stress tests.

## Consolidation Plan

1. Keep `prove_ready.py` as the canonical local release gate, but be conservative
   about adding child commands.
2. Keep `installed_product_abuse.py` as the primary product-facing daily-driver
   gate.
3. Use `RELIABILITY_COVERAGE_GAP_AUDIT.md` to decide whether a new bug belongs
   in a unit test, deterministic eval, installed-product smoke,
   fault-injection smoke, or documentation-only bucket.
4. Merge new wording/friction checks into `daily_driver_maturity_audit.py`,
   not into new proof scripts.
5. Add generated messy-input cases to `chat_eval.py` first; add unit tests only
   for specific helper logic or safety branches.
6. Treat focused lifecycle smokes as operator safety gates, not normal daily
   loops.
7. Mark old live/hardware/provider scripts as manual until they are promoted by
   a current release lane.
8. Revisit `test_orchestrator.py` after VM proof: split high-value semantic
   router cases from historical phrase regressions if maintenance remains slow.

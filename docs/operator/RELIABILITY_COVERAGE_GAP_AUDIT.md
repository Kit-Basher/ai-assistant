# Reliability Coverage Gap Audit

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.
Test/proof-suite ownership lives in
`docs/operator/TEST_SUITE_RATIONALIZATION.md`.

This audit maps subsystem guarantees to current proof coverage and missing
fault-injection work. It is not a request to add another broad proof lane. The
goal is to make the installed local assistant boringly reliable by fixing the
highest-risk unproven guarantees first.

## How To Read This

Status labels:

- `covered`: current tests/proofs exercise the guarantee directly.
- `partial`: covered in the happy path or by mocks, but missing important
  fault-injection or installed-product coverage.
- `weak`: documented or indirectly covered, but likely to miss real user
  failures.
- `missing`: no meaningful proof found.

Priority:

- `P0`: daily-driver blocker or safety guarantee.
- `P1`: fail-graceful or self-repair gap likely to create repeated friction.
- `P2`: polish, diagnostics, or lower-frequency operator confidence.

## Highest-Risk Missing Guarantees

The main risk is not lack of tests overall. It is that several subsystems have
happy-path installed-product proof but limited fault injection.

P0/P1 gaps to address first:

1. Search fault injection for bad endpoints, provider JSON failure, port
   conflict, and configured-stopped repair failure.
2. Telegram stale-lock/duplicate-poller fault injection and bounded repair
   wording.
3. Stale confirmation persistence across restart/thread/session boundaries for
   more action families, not only the current smoke examples.
4. Backup/support/executor journal boundedness over repeated runs.
5. Secret-store missing/corrupt behavior across doctor, support bundle, backup,
   and chat status surfaces.

## Subsystem Map

### 1. Startup / Runtime Health

Required guarantees:

- `/ready`, `/state`, `/version`, doctor, and split/status surfaces agree.
- API stopped/restarted is detected and recovers through the service manager.
- runtime/current mismatch is visible, not silently ignored.
- degraded/bootstrap states include a clear next action.
- optional services do not make core health fail.
- stale lock files do not produce false healthy status.

Existing proof:

- `prove_ready.py`, `release_smoke.py`, `restart_survival_smoke.py`,
  `daily_driver_maturity_audit.py`, doctor tests, ready/state/runtime truth
  tests.

Weak or missing:

- `partial`: stale lock-file injection.
- `partial`: runtime/current mismatch is checked by installed abuse, but not
  heavily fault-injected.
- `weak`: split_status coverage is not as prominent as `/ready` and `/state`.

Recommended fault injection:

- fake stale lock file
- fake runtime/current target mismatch
- API unavailable then restored
- degraded model/provider but ready web shell

Self-repair expectation:

- service restart is operator-controlled or smoke-controlled; stale locks may be
  auto-cleared only when ownership and age are safe.

Priority: `P1`.

Recommended test type: fault-injection smoke plus focused unit tests.

### 2. Search

Required guarantees:

- `never_configured` offers trusted managed SearXNG setup, not automatic setup.
- `configured_running` searches metadata-only immediately.
- `configured_stopped` offers inline start/repair or repairs only the trusted
  managed endpoint.
- bad or untrusted config is refused.
- provider failures do not become fake search success.
- port conflict and setup failure produce exact next actions.
- no page fetch, browser automation, downloads, JavaScript, or pack import.
- chat never claims it searched when it did not.

Existing proof:

- `test_safe_web_search.py`, `test_managed_local_services.py`,
  `daily_driver_smoke.py`, `installed_product_abuse.py`,
  `restart_survival_smoke.py`, `prove_daily_driver_product.py`,
  `daily_driver_maturity_audit.py`.

Weak or missing:

- `partial`: configured-stopped repair happy path is covered; repair failure is
  less covered.
- `partial`: bad endpoint/provider JSON failure covered in unit tests more than
  installed-product fault injection.
- `weak`: port conflict is known from prior bugs but should be a deliberate
  fault-injection case.

Recommended fault injection:

- invalid `SEARXNG_BASE_URL`
- HTML-only endpoint with no JSON
- endpoint timeout/refused connection
- owned container stopped
- port 8080/8888 occupied
- provider returns malformed JSON

Self-repair expectation:

- only known trusted managed SearXNG may be started/repaired; first-time setup
  remains Plan Mode gated.

Priority: `P0`.

Recommended test type: installed-product fault-injection smoke and unit tests.

### 3. Telegram

Required guarantees:

- missing token is optional and clear.
- configured token with inactive service is optional, not whole-agent failure.
- duplicate poller/stale lock is detected.
- token is never exposed.
- start/stop/restart remain Plan Mode gated.
- stale lock self-repair runs only when safe and bounded.

Existing proof:

- Telegram unit tests, `installed_product_abuse.py`,
  `daily_driver_maturity_audit.py`, `operator_lifecycle_smoke.py`,
  `prove_ready.py` doctor behavior.

Weak or missing:

- `partial`: duplicate poller and stale lock behavior is not strongly covered
  by installed-product fault injection.
- `partial`: start/stop/restart execution remains preview/Plan Mode focused.

Recommended fault injection:

- fake stale Telegram lock owned by this runtime
- fake stale Telegram lock not safely owned
- duplicate poller evidence in status/log fixture
- missing token, malformed token, token configured but service inactive

Self-repair expectation:

- stale lock may be cleaned only with strict owner/age validation; service
  start/restart remains a mutating Plan Mode action.

Priority: `P1`.

Recommended test type: focused unit tests plus installed-product abuse cases.

### 4. Memory

Required guarantees:

- memory status and “what do you remember” are understandable.
- current-turn no-memory opt-out is deterministic and does not disable tools.
- remembered facts, open loops, anchors, and preferences persist where intended.
- thread/global memory scopes do not leak unexpectedly.
- export/delete/redact/dedupe remain preview-only unless bounded executors are
  implemented.
- stale context does not hijack new explicit intent.
- memory never authorizes mutation by itself.

Existing proof:

- `memory_lifecycle_smoke.py`, `daily_driver_maturity_audit.py`,
  memory unit tests, open-loop tests, working-memory tests,
  `chat_eval.py`, `llm_behavior_eval.py`.

Weak or missing:

- `partial`: durable memory restart/persistence has tests, but full-suite state
  leakage and old replay fixtures remain noisy.
- `missing`: no-memory opt-out for longer multi-turn flows after a topic switch.
- `weak`: “why did you use that memory?” explainability is mostly UX preview.

Recommended fault injection:

- conflicting durable memory vs current user instruction
- old pending context plus explicit rewrite/search/status request
- corrupt or missing memory DB
- thread preference in one thread then unrelated thread

Self-repair expectation:

- transient pending state can be cleared; durable destructive changes require
  Plan Mode confirmation.

Priority: `P1`.

Recommended test type: deterministic eval plus targeted unit tests.

### 5. Plan Mode / Confirmations

Required guarantees:

- mutators create inspectable canonical plans.
- confirmation binds to the exact current plan and current thread/session.
- stale plans expire.
- cancel clears pending action.
- preview-only/dangerous actions do not mutate.
- ambiguous confirmation with no plan does nothing.

Existing proof:

- `plan_mode_v2_smoke.py`, `executor_registry_smoke.py`,
  `installed_product_abuse.py`, `test_plan_policy.py`, executor tests.

Weak or missing:

- `partial`: broad action-family matrix for stale confirmation after restart.
- `partial`: plan expiry fault injection uses simulated paths more than
  installed-product time manipulation.

Recommended fault injection:

- confirm after restart for backup/support/search/Telegram/memory previews
- confirm in unrelated thread
- confirm after cancel
- tampered plan id/token/action target

Self-repair expectation:

- reject and ask for fresh preview.

Priority: `P0`.

Recommended test type: installed-product smoke plus unit tamper tests.

### 6. Executor Registry

Required guarantees:

- enabled executors return the canonical result schema.
- preview-only and unavailable executors return `mutated=false`.
- resources touched and rollback hints are scoped.
- journals redact secrets and do not grow recursively.
- exceptions record whether mutation happened.

Existing proof:

- `executor_registry_smoke.py`, `support_bundle_v2_smoke.py`,
  `backup_v1_smoke.py`, `test_executor_registry.py`.

Weak or missing:

- `partial`: journal size/rotation/boundedness over repeated runs.
- `partial`: executor exception after partial artifact creation.
- `weak`: recursive inclusion of journal summaries over time is documented but
  should be stress-checked.

Recommended fault injection:

- executor throws before artifact finalization
- executor throws after partial artifact directory creation
- oversized journal summary
- secret-like strings in journal rows

Self-repair expectation:

- report partial artifact and scoped rollback hint; never delete outside owned
  artifact path automatically.

Priority: `P1`.

Recommended test type: unit and operator safety smoke.

### 7. Backup / Restore / Cleanup

Required guarantees:

- backups are bounded, redacted, and additive.
- latest valid backup detection works.
- malformed backup is detected.
- outside paths are rejected.
- live restore remains disabled.
- cleanup preview protects latest backup/current runtime/secrets/services.
- preview-only cleanup never deletes.
- state growth is visible.

Existing proof:

- `backup_v1_smoke.py`, `restore_validator_smoke.py`,
  `cleanup_preview_smoke.py`, `operator_lifecycle_smoke.py`,
  backup/restore proof tests.

Weak or missing:

- `partial`: repeated backup growth and cleanup-candidate drift.
- `partial`: corrupt backup archive/directory variants beyond missing manifest.
- `missing`: live restore executor by design; not a current gap if documented.

Recommended fault injection:

- backup with missing manifest
- backup with wrong schema
- backup with raw secret marker
- backup outside approved path
- many backups/support bundles/runtime releases

Self-repair expectation:

- validator explains; cleanup previews candidates but does not delete yet.

Priority: `P1`.

Recommended test type: existing smokes plus additional malformed fixtures.

### 8. External Packs / Skills

Required guarantees:

- source trust model holds.
- discovery does not approve/enable/use.
- approval does not enable/grant/use.
- enable validates input types.
- pack text cannot execute code.
- pack guidance is relevant and does not hijack local planning/search prompts.

Existing proof:

- `external_pack_safety_smoke.py`, `test_api_packs_endpoints.py`,
  pack lifecycle/source tests, `installed_product_abuse.py`, recent focused
  pack-enable validation fix.

Weak or missing:

- `partial`: installed-product pack relevance under messy inputs is covered by
  abuse/eval, but not exhaustive by pack category.
- `weak`: old pack acquisition tests encode stale route expectations.

Recommended fault injection:

- hostile pack text asking to bypass Plan Mode
- malformed lifecycle payloads
- pack enabled but permissions absent
- removed/tombstoned pack still referenced by chat

Self-repair expectation:

- refuse unsafe lifecycle step and explain next safe action.

Priority: `P0` for safety, `P2` for relevance polish.

Recommended test type: deterministic eval and focused API unit tests.

### 9. Secrets

Required guarantees:

- raw reads require explicit `--show` or equivalent.
- redacted reads work.
- support bundles/backups/chat/status do not expose tokens/API keys/passwords.
- missing/corrupt secret store is reported safely.

Existing proof:

- `test_agent_secrets_cli.py`, support bundle/backup smokes, installed abuse
  secret scans, docs truth and redaction helpers.

Weak or missing:

- `partial`: corrupt secret-store handling across all user-facing status
  surfaces.
- `partial`: confirmation tokens/plan ids redaction policy is documented less
  clearly than API keys/Tokens.

Recommended fault injection:

- missing secret store
- unreadable/corrupt secret store
- token-like values in logs/journal/memory

Self-repair expectation:

- never print raw value; ask for reconfiguration or support bundle with
  redaction.

Priority: `P0`.

Recommended test type: unit tests plus support/backup artifact scans.

### 10. UI / API Behavior

Required guarantees:

- web root responds.
- endpoints return coherent JSON.
- unsupported methods return useful method errors.
- bad input returns actionable errors.
- normal chat does not expose developer-log junk.
- frontend does not assume stale endpoints.

Existing proof:

- `installed_product_abuse.py`, `first_run_smoke.py`,
  `webui_robustness_smoke.py`, `release_gate_matrix_smoke.py`, API tests.

Weak or missing:

- `partial`: browser refresh/hard-refresh/large transcript remains mostly
  manual/static.
- `partial`: endpoint manifest is checked by installed abuse for key routes,
  but not every documented operator route.

Recommended fault injection:

- wrong HTTP method
- malformed JSON
- missing required fields
- frontend hard refresh after promotion
- request timeout/failure display

Self-repair expectation:

- return structured error and next action; do not ask user to debug internals.

Priority: `P1`.

Recommended test type: installed-product API smoke plus UI checklist or future
browser automation.

### 11. Response Quality / Routing

Required guarantees:

- normal chat works without forcing tools.
- provided-text rewrite/translate/summarize does not trigger search/status.
- correction prompts do not replay stale diagnostics.
- status prompts use local status.
- public lookup prompts use metadata-only search when available.
- local repo/planning prompts stay local.
- errors are concise and include safe next action.

Existing proof:

- `chat_eval.py`, `llm_behavior_eval.py`, `daily_driver_smoke.py`,
  `daily_driver_maturity_audit.py`, `installed_product_abuse.py`,
  `test_chat_behavior_audit.py`, `test_orchestrator.py`.

Weak or missing:

- `partial`: broad generated eval exists, but live model response quality can
  still vary.
- `partial`: local planning/repo prompts had recent bugs; keep them in maturity
  audit.
- `weak`: language consistency guard catches CJK leakage, but other quality
  regressions are harder to score.

Recommended fault injection:

- stale doctor context then rewrite
- ambiguous “try again” after status output
- “what is next step” after repo/planning context
- explicit “do not search” plus public entity
- mixed prompt injection inside provided text

Self-repair expectation:

- ask one clarifying question or route deterministically; do not keep repeating
  stale context.

Priority: `P1`.

Recommended test type: deterministic eval plus daily-driver maturity audit.

## Weak Or Redundant Existing Tests

Weak because they do not hit installed product:

- many older phrase tests in `tests/test_orchestrator.py`
- older planner/safe-mode transcript tests
- old pack acquisition chat-route tests

Weak because they are environment-sensitive:

- Telegram live/poller tests
- Ollama/provider health tests
- managed-service tests that assume Podman/Docker presence or absence

Redundant but useful as separate safety gates:

- `plan_mode_v2_smoke.py` and `executor_registry_smoke.py`
- operator lifecycle smoke plus backup/support/cleanup/restore focused smokes
- daily-driver smoke plus installed-product abuse plus maturity audit

Recommendation: do not delete these now. Use this audit when deciding where a
new regression belongs. Add a generated eval case or a focused unit test before
creating another proof script.

## Fault-Injection Tests To Add First

1. Search configured-stopped repair failure matrix:
   bad endpoint, owned container stopped, port conflict, HTML-only endpoint,
   malformed JSON.
2. Telegram stale lock and duplicate poller matrix:
   safe owned stale lock, unsafe lock, duplicate poller evidence, inactive
   optional service.
3. Secret-store corruption and redaction matrix:
   missing/corrupt store across doctor/status/support/backup/chat.
4. Executor artifact failure matrix:
   support/backup executor fails before final manifest and after partial
   artifact creation; result includes `mutated=false` or scoped partial rollback
   hint as appropriate.
5. Confirmation restart/thread matrix:
   backup/support/search/Telegram/memory previews cannot be confirmed from
   stale or unrelated contexts.

## Missing Self-Repair Behavior

- Search can recover configured-stopped managed SearXNG, but repair failure
  diagnostics need stronger fault-injection proof.
- Telegram stale lock cleanup should remain conservative; safe auto-clear is
  not proven enough to enable broadly.
- Secret-store corruption should offer a bounded next action and support bundle
  path without exposing secret material.
- Backup/support partial artifacts should be clearly reported with rollback
  hints scoped only to the new artifact.
- Full cleanup/delete/restore self-repair remains preview-only by design.

## Top 5 Next Implementation Items

Keep the next batch small:

1. Add a search fault-injection smoke or focused test matrix for
   configured-stopped/bad-endpoint/HTML-only/malformed-JSON/port-conflict
   behavior.
2. Add Telegram stale-lock and duplicate-poller fault-injection tests with
   precise optional-service wording and no token exposure.
3. Add secret-store missing/corrupt redaction tests across status, support
   bundle, backup, and chat.
4. Add executor partial-artifact failure tests for support bundle and backup
   result/journal/rollback behavior.
5. Add a Plan Mode stale-confirmation matrix across backup, support, search,
   Telegram, and memory action families.

Do not start VM proof, add broad executors, or create another umbrella proof
script until these P0/P1 gaps are closed or explicitly deferred.

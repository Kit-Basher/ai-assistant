# Pytest Failure Triage

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.
Test-suite ownership lives in `docs/operator/TEST_SUITE_RATIONALIZATION.md`.

This triage classifies the full pytest inventory failure set. It does not make
full pytest a release gate.

## Run Captured

Command:

```bash
python -m pytest -q --tb=short | tee /tmp/personal-agent-pytest-full.txt
```

Result from this pass:

- Failed: 108.
- Passed: 2280.
- Runtime: about 15 minutes.
- Prior rationalization run: 90 failed, 2298 passed.

The changed count means the full inventory is order/environment sensitive. The
curated gates still matter more for current release truth.

## Summary Classification

| Group | Count | Meaning | Release contradiction |
| --- | ---: | --- | --- |
| Stale expected wording | 15 | The test expects old text such as `Now using ...` while current behavior says `Temporarily switched ...` or uses newer grounded wording. | No. |
| Stale route behavior | 27 | The test expects an older route or planner path, but current semantic routing, safe search setup, or Plan Mode v2 intercepts first. | Mostly no; a few should be reviewed. |
| Environment/dependency assumption | 13 | The test assumes token state, Podman absence, Ollama health shape, Telegram poller availability, or path defaults that do not hold in this runtime. | No for release gates; yes for isolated-test quality. |
| Duplicate of current release gates | 18 | Old transcript/barrage/eval tests duplicate `chat_eval.py`, `llm_behavior_eval.py`, `release_smoke.py`, or installed-product gates with older expectations. | No. |
| Obsolete removed/legacy behavior | 14 | Tests assert old assistant planner, safe-mode transcript, single-service topology, or older install-path behavior. | No unless the behavior is intentionally revived. |
| Flaky/timing/soak/order-sensitive | 10 | Stress, concurrency, warmup, scheduled notification, or full-suite order-sensitive tests. | No by default. |
| Real possible regression | 11 | Failures that may point at an actual bug or missing current contract. | Investigate before claiming full pytest clean. |
| Unknown | 0 | Every failure has at least an initial classification. | N/A. |

Total: 108.

## Failure Groups

### 1. Stale Expected Wording

Representative failures:

- `tests/test_agent_workflows.py::...controlled_mode_blocked_action_then_approval_and_success`
- `tests/test_agent_workflows.py::...discovery_to_approval_install_verify_and_switch_flow`
- `tests/test_api_server.py::...llm_control_mode_can_enable_controlled_mode_from_safe_baseline_and_return`
- `tests/test_api_server.py::...temporary_chat_model_switch_journals_and_does_not_persist_default`
- `tests/test_api_server.py::...temporary_chat_model_restore_to_default_clears_override_and_persists_verified_journal`
- `tests/test_default_model_policy_integration.py::*`
- `tests/test_search_setup_ux.py::*`

Likely cause:

- Model/provider switching and setup wording changed to current temporary/default
  switch semantics and Plan Mode wording.
- Search setup/status wording now goes through managed SearXNG lifecycle text.

Recommended action:

- Update tests only if the current product wording is the documented behavior.
- Prefer route/result assertions over exact prose.
- Keep out of release path until updated.

### 2. Stale Route Behavior

Representative failures:

- `tests/test_api_packs_endpoints.py::...external_pack_is_usable_through_chat_and_removed_cleanly`
- `tests/test_api_server.py::...chat_ordinary_message_still_works_through_orchestrator`
- `tests/test_api_server.py::...chat_preserves_response_envelope_for_preflight_short_circuit`
- `tests/test_assistant_planner.py::*`
- `tests/test_pack_acquisition.py::*`
- `tests/test_provider_setup_flow.py::*`
- `tests/test_setup_chat_flow.py::...real_os_package_install_still_routes_to_package_manager`
- `tests/test_value_optimal_selection.py::...api_chat_leaves_default_model_selection_to_canonical_inference`

Likely cause:

- Current semantic routing and deterministic setup/search/package/Plan Mode
  routes run before older generic LLM/planner expectations.
- Some unit fixtures do not configure search/service state, so public lookup
  questions route to managed SearXNG setup.

Recommended action:

- Split tests into current route-contract tests and historical planner tests.
- For pack lifecycle, rely on `external_pack_safety_smoke.py` and focused pack
  lifecycle tests before broad chat-route expectations.
- Fix fixtures where they accidentally trigger search setup instead of the
  intended pack/model path.

### 3. Environment Or Dependency Assumptions

Representative failures:

- `tests/test_api_server.py::...start_embedded_telegram_sets_running_state_and_prints_events`
- `tests/test_api_server.py::...telegram_secret_and_test_endpoints`
- `tests/test_managed_local_services.py::...setup_prompt_with_docker_available_offers_podman_prerequisite`
- `tests/test_observe_config.py::*`
- `tests/test_ollama_probe.py::...health_marks_ollama_up_when_native_ok_openai_404`
- `tests/test_ready_endpoint.py::*`
- `tests/test_scheduled_model_scout.py::*`
- `tests/test_telegram_audit_logging.py::*`
- `tests/test_telegram_model_provider_router.py::*`

Likely cause:

- Full pytest ran on a machine with live Podman, Docker, Telegram token/service
  state, and possible Telegram poller conflict output.
- Some tests assume missing Telegram token, missing Podman, or mocked service
  states that differ from the local environment.

Recommended action:

- Isolate these tests from live user config and service state.
- Mark truly live Telegram/Ollama/provider tests as optional/manual.
- Keep `installed_product_abuse.py`, `daily_driver_smoke.py`, and
  `prove_ready.py` as current installed-runtime truth.

### 4. Duplicate Of Current Release Gates

Representative failures:

- `tests/test_assistant_interaction_barrage.py::*`
- `tests/test_behavioral_eval_battery.py::*`
- `tests/test_chat_behavior_audit.py::*` when run inside full pytest
- `tests/test_safe_mode_transcript.py::*`
- `tests/test_authoritative_domain_gate.py::*`

Likely cause:

- These tests duplicate current release/eval coverage but still encode older
  transcript text, old planner assumptions, or order-sensitive state.
- Notably, `tests/test_chat_behavior_audit.py` passes in the curated
  `prove_ready.py` gate but fails in full pytest, indicating full-suite
  contamination/order sensitivity rather than a release-gate failure.

Recommended action:

- Move high-value invariants into `chat_eval.py`/`llm_behavior_eval.py`.
- Retire or mark old transcript tests historical unless they protect a unique
  safety invariant.
- Investigate full-suite state leakage separately.

### 5. Obsolete Or Legacy Behavior

Representative failures:

- `tests/test_assistant_planner.py::*`
- `tests/test_install_first_run_hardening.py::*`
- `tests/test_runtime_status.py::...source_and_docs_do_not_reference_obsolete_single_service_topology`
- `tests/test_safe_mode_transcript.py::*`
- `tests/test_control_plane.py::...repo_canonical_tasks_file_is_parseable`

Likely cause:

- The project moved from older planner/safe-mode/install-path contracts to
  semantic routing, Plan Mode v2, installed-product proof lanes, and current
  runtime split docs.

Recommended action:

- Decide whether each old behavior is still product-supported.
- Delete or move obsolete tests to historical/manual once their invariant is
  covered elsewhere.
- Do not fix product behavior just to satisfy old topology wording.

### 6. Flaky, Timing, Or Soak

Representative failures:

- `tests/test_concurrency_stress.py::*`
- `tests/test_extended_soak.py::*`
- `tests/test_regression_soak.py::*`
- `tests/test_runtime_concurrency.py::*`
- `tests/test_runtime_events.py::*`
- `tests/test_working_memory_concurrency.py::*`
- `tests/test_working_memory_cross_session_replay.py::*`

Likely cause:

- These are stress/order/warmup/compaction tests and are not suitable for every
  default run until isolated.

Recommended action:

- Move to an explicit soak command group.
- Run before major release candidates, not after every daily-driver patch.
- Fix only if failure reproduces in isolation or contradicts current memory
  lifecycle/user-facing proof.

### 7. Real Possible Regressions

Representative failures:

- `tests/test_adversarial_requests.py::...enable_rejects_non_boolean_enabled_value`
- `tests/test_agent_secrets_cli.py::...get_redacted_masks_secret_value`
- `tests/test_api_server.py::...health_normalization_marks_disabled_provider_and_models_with_timestamps`
- `tests/test_api_server.py::...provider_disabled_health_stamps_all_provider_models_even_when_enabled_flag_true`
- `tests/test_open_loops.py::...open_loop_priority_and_views`
- `tests/test_runtime_status.py::...permission_denied_degrades`
- `tests/test_runtime_status.py::...report_structure_and_redaction`
- `tests/test_working_memory_behavioral_replay.py::*`
- `tests/test_working_memory_persistence_hardening.py::*`

Likely cause:

- These failures may be real stale tests, but they touch validation ordering,
  CLI secret redaction, status redaction/health semantics, and memory replay.
  Those are important enough to investigate rather than blanket-retire.

Recommended action:

- Triage these first in a focused pass.
- If current behavior is correct, update narrow expectations.
- If current behavior is wrong, fix product behavior and add a focused
  regression.
- Do not add them all to `prove_ready.py` until they are clean in isolation.

## Does This Contradict Current Release Gates?

No current curated release gate failed in this pass.

Current gate status after triage:

- `scripts/docs_truth_smoke.py`: pass.
- `scripts/release_gate_matrix_smoke.py`: pass.
- `scripts/release_smoke.py`: pass.
- `scripts/prove_ready.py`: `READY_FOR_VM_PROOF=yes`,
  `RELEASE_BLOCKERS=0`, warnings only.

There is one important caveat: a few tests from `test_chat_behavior_audit.py`
fail only in the full pytest run but pass when run through `prove_ready.py`.
That suggests full-suite state leakage or order sensitivity. It should be
investigated, but it does not invalidate the current curated release gate.

## Recommended Next Passes

1. Focused real-regression pass:
   `test_adversarial_requests.py`, `test_agent_secrets_cli.py`,
   `test_runtime_status.py`, and working-memory replay/persistence tests.
2. Fixture isolation pass:
   tests accidentally hitting live Podman/Telegram/search setup should be made
   hermetic or moved to optional live gates.
3. Planner/transcript retirement pass:
   classify `test_assistant_planner.py` and `test_safe_mode_transcript.py` as
   current, historical, or delete candidates.
4. Pack acquisition route pass:
   update broad pack chat tests to the current Plan Mode lifecycle contract or
   rely on existing pack lifecycle safety gates.
5. Soak group pass:
   move stress/soak/concurrency tests into an explicit occasional command group.


# Release Readiness Audit Baseline

Date: 2026-06-08
Checkpoint: `5195f6a` Persist journals for support bundle creation
Latest clean checkpoint before this pass: `5195f6a` Persist journals for support bundle creation

This checkpoint captures the current operator/project baseline so future chats and helpers can resume from the same product and safety state.

## Current Truth

- The user talks to the assistant layer.
- The assistant layer interprets intent, asks the agent layer for grounded runtime/tool facts or bounded action results when needed, and explains those results back to the user.
- The agent layer validates capabilities, reads runtime truth/native skill output, performs only approved bounded actions, and returns structured results.
- Direct native report commands remain raw and deterministic. Presentation rewrites, narration, and style transforms are not hidden core runtime behavior; they belong in explicitly bounded text-only skills or presentation adapters if added later.
- External pack acquisition remains preview-first and confirmation-gated. Source trust is not content trust, and imported content remains untrusted until it passes quarantine, review, enablement, configuration, and permission gates.
- Managed-action reliability now covers the high/medium normal-user write paths listed below; remaining follow-ups are tracked in `docs/operator/MANAGED_ACTION_RELIABILITY_AUDIT.md`.

## Operator History Baseline

## Latest Known Commits

- `aac06ba` Update managed action recovery journal docs
- `5195f6a` Persist journals for support bundle creation
- `7397f02` Update persistent journal helper for preference reset
- `042c98e` Persist journals for preference reset cleanup
- `b4952f3` Add persistent managed action journal skeleton
- `d807cb0` Update preference reset API reliability tests
- `b662821` Add reliability journaling for preference reset cleanup
- `295b578` Harden semantic memory indexing reliability
- `9989002` Split external pack review approval confirmation
- `3dc123f` Add imported pack review state UX
- `4dd8e80` Document external pack format
- `34ed8c1` Improve live barrage answer quality

## What Is Now True

- Native safe web search exists using SearXNG metadata search.
- Search results are untrusted metadata only.
- Pack acquisition can use safe search for untrusted source leads.
- Source approval is separate from content trust.
- Approved source can be fetched only into quarantine/import-for-review.
- Review state is shown before review approval.
- Review approval is previewed and explicitly confirmed before recording approval only.
- Enablement is previewed and explicitly confirmed before recording enablement only.
- Permission/configuration is previewed and explicitly confirmed before recording metadata/config only.
- Managed-adapter invocation is previewed and explicitly confirmed before running a core-owned adapter operation.
- Confirm-gated SearXNG managed local service setup can run only the approved Docker/Podman image, name, loopback bind, approved primary/fallback ports, and managed volume.
- Managed local service setup now journals owned changes and rolls back the SearXNG container it created if health checks fail.
- Web search cleanup is a separate confirmed action that only targets `personal-agent-searxng`.
- Managed action reliability now has a product-wide standard and audit. Only SearXNG setup/cleanup currently meets the full journal/owned-rollback pattern; other mutating flows are tracked as follow-up work.
- Model acquisition/import flows now attach managed-action journals, verify post-action inventory, and clean up owned generated Modelfiles/temp artifacts. They do not delete unproven Ollama cache/model data or user-provided Modelfiles.
- Provider/API key configuration now attaches managed-action journals, persists redacted managed-action journal status transitions, verifies secret/config writes, redacts secret metadata and provider-test response bodies from journals, and rolls back failed verified key saves to the previous key/source or removes the failed new key. Provider config update and the provider config/secret portions of OpenRouter setup also persist redacted journal rows; later model/default mutations remain separate follow-up work.
- Default model changes now attach managed-action journals, preflight chat capability/provider/model usability, verify persisted defaults or temporary override state after mutation, and roll back only previous defaults/temporary target state if verification fails.
- Telegram token setup now attaches managed-action journals, verifies secret writes by readback, redacts token metadata, and rolls back failed token saves to the previous token or removes the failed new token. Telegram enable/disable now journals the known Personal Agent drop-in, approved `systemctl --user` daemon-reload/restart/stop actions, runtime status verification, and restores/removes only the owned drop-in on service verification failure.
- Pack lifecycle metadata mutations now attach managed-action journals for source approval, import records, review approval, enablement, and selected-file permission grants. They verify metadata by readback and restore only owned prior pack/source/grant metadata when verification fails.
- Registry/autoconfig/self-heal/hygiene/cleanup/capabilities reconcile/bootstrap/rollback flows now attach managed-action journals through the transactional registry path, verify registry state by readback/hash, and restore the pre-action registry snapshot when verification fails and ownership is proven.
- Preference-backed memory/bootstrap markers and onboarding/preferences writes now attach managed-action journals, verify by readback, restore/remove only the owned target key on verification failure, keep global and per-thread preference scopes separate, and redact raw preference/memory content from journals.
- Bulk preference reset/clear paths now attach managed-action journals for explicit global/user/thread target keys and approved user-pref prefixes, record redacted scoped snapshot hashes, persist redacted managed-action journal status transitions, verify target keys are removed while unrelated scopes stay unchanged, restore the previous scoped preference snapshot on failed verification, and keep raw preference values and raw persisted keys out of persistent journal rows.
- Support bundle artifacts now write a redacted managed-action journal inside the owned temp bundle, persist redacted managed-action journal status transitions, verify expected files by readback, remove only the newly created `agent-support-*` directory on failed verification, and record recovery_needed if owned cleanup cannot complete. Notification test/send/prune now journal policy/target metadata, verify local notification history writes and prune count/window results, restore prior notification history on failed local verification where a snapshot exists, and verify action-ledger appends by readback.
- Pack removal/source deletion cleanup now attaches managed-action journals, verifies removed/tombstoned/source-policy state by readback, restores prior owned metadata on verification failure, and keeps hostile imported text redacted from tombstones/support output.
- Semantic memory remains disabled by default and release-gated, but optional semantic ingest/rebuild/repair paths now attach redacted managed-action journals, verify source/chunk/vector/index-state readback, keep duplicate observe writes idempotent through deterministic source hashes, remove only owned failed new ingest rows, preserve prior usable index state on failed repair, and expose a read-only semantic doctor plus confirmed repair path.
- Remaining managed-action reliability gaps are audited. A minimal persistent managed-action journal storage skeleton now exists, and preference reset/clear, support bundle creation, and provider/API key config are converted reference flows. Highest priority next target is read-only restart/status surfacing plus converting the next managed-action family to persistent journal writes; package install/directory creation shell flows, semantic-memory soak before any default-on promotion, quarantine artifact cleanup, and future filesystem writes remain tracked follow-ups. Remote notification delivery and action ledger records remain append-only by design after local readback verification.
- Release readiness is Yellow at `5195f6a`: suitable for a controlled public trial only after clean install verification, not a broad Green release. See `docs/operator/RELEASE_READINESS_AUDIT.md`.
- External pack format is documented.
- Live barrage quality now rejects weak fallback answers like "I’m not sure" and generic "try rephrasing".

## Current Acquisition Chain

missing capability
→ approved catalog search
→ safe_web_search untrusted source leads if no catalog candidate
→ source approval preview
→ explicit source trust record
→ quarantine fetch preview
→ fetch/import for review only
→ review state shown
→ review approval preview
→ explicit review approval confirmation
→ enablement preview
→ explicit enablement confirmation
→ permission/configuration preview
→ scoped grant preview
→ explicit permission/configuration confirmation
→ managed adapter invocation preview
→ explicit managed adapter invocation confirmation
→ core-owned managed adapter operation only if `usable=true`

## Hard Safety Invariants

- No arbitrary external code execution.
- No `handler.py` execution.
- No dependency install.
- No shell.
- No browser/OAuth/network access from packs.
- No pack approval during source approval.
- No enablement during quarantine fetch/import.
- No enablement during review approval.
- No permission grant during enablement.
- No permission grant unless explicitly confirmed.
- No adapter invocation during permission/configuration grant.
- No automatic adapter invocation after permission/configuration grant.
- No external code, shell, subprocess, browser, OAuth, or pack-owned network access during managed-adapter invocation.
- No arbitrary Docker commands, arbitrary ports, host networking, privileged containers, random mounts, silent container deletion, or external-pack-triggered service execution.
- No managed action may silently mutate pre-existing user resources during recovery.
- Expired confirmations must not execute, and consumed confirmations must not replay.
- No managed action should leave silent background services after failed setup.
- Source trust is not content trust.
- Review approval is not enablement.
- Enablement is not permission grant.
- Permission grant is not arbitrary code execution.
- Permission grant is not adapter invocation or pack use.

## Required Proof Set

Run this after external-pack, search, acquisition, or routing changes:

1. `bash scripts/promote_local_stable.sh`
2. `python scripts/external_pack_safety_smoke.py`
3. `python -u scripts/live_user_barrage.py --base-url http://127.0.0.1:8765 --telegram-bridge --timeout 90 --strict-quality`
4. `git status`

`external_pack_safety_smoke` currently covers 39 hostile-intake, lifecycle, managed-service, and recovery gates. It proves hostile intake gates. `live_user_barrage` proves normal assistant behavior and answer quality did not regress.

## Next Likely Work

- Add read-only persistent-journal status surfacing and convert the next managed-action family before making broader crash/restart recovery claims.
- Run a clean release-bundle install/launch/onboarding/uninstall pass before any public trial.
- Continue improving product UX/readability of barrage answers.
- Add future core-owned content operations only after separate preview/confirmation and safety tests.
- Add more managed adapters only through core-owned safety boundaries.
- Do not expand arbitrary plugin execution.

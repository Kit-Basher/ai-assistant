# External Pack Acquisition Safety + Quality Baseline

Date: 2026-05-21

This checkpoint captures the current operator/project baseline so future chats and helpers can resume from the same product and safety state.

## Latest Known Commits

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
- Provider/API key configuration now attaches managed-action journals, verifies secret/config writes, redacts secret metadata, and rolls back failed verified key saves to the previous key/source or removes the failed new key.
- Default model changes now attach managed-action journals, preflight chat capability/provider/model usability, verify persisted defaults or temporary override state after mutation, and roll back only previous defaults/temporary target state if verification fails.
- Telegram token setup now attaches managed-action journals, verifies secret writes by readback, redacts token metadata, and rolls back failed token saves to the previous token or removes the failed new token. Telegram enable/disable now journals the known Personal Agent drop-in, approved `systemctl --user` daemon-reload/restart/stop actions, runtime status verification, and restores/removes only the owned drop-in on service verification failure.
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

- Continue improving product UX/readability of barrage answers.
- Add future core-owned content operations only after separate preview/confirmation and safety tests.
- Add more managed adapters only through core-owned safety boundaries.
- Do not expand arbitrary plugin execution.

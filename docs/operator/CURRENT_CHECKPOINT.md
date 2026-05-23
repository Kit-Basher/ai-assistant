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
→ configure/permission
→ managed adapter use only if `usable=true`

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
- Source trust is not content trust.
- Review approval is not enablement.
- Enablement is not permission grant.
- Permission grant is not arbitrary code execution.

## Required Proof Set

Run this after external-pack, search, acquisition, or routing changes:

1. `bash scripts/promote_local_stable.sh`
2. `python scripts/external_pack_safety_smoke.py`
3. `python -u scripts/live_user_barrage.py --base-url http://127.0.0.1:8765 --telegram-bridge --timeout 90 --strict-quality`
4. `git status`

`external_pack_safety_smoke` currently covers 29 hostile-intake and lifecycle gates. It proves hostile intake gates. `live_user_barrage` proves normal assistant behavior and answer quality did not regress.

## Next Likely Work

- Continue improving product UX/readability of barrage answers.
- Implement careful configuration/permission continuation if not already complete.
- Add more managed adapters only through core-owned safety boundaries.
- Do not expand arbitrary plugin execution.

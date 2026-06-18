# Roadmap Index

Date: 2026-06-15

This is the authoritative index for roadmap, checkpoint, operator, and design
documents. It does not delete, rename, or supersede detailed docs; it tells
future work where to look first.

## Current Authority

| Need | Authoritative doc |
| --- | --- |
| Product promise and assistant/agent boundary | `docs/product/PROJECT_INTENT.md` |
| Current working state and proof commands | `docs/operator/CURRENT_CHECKPOINT.md` |
| Release history and tag ledger | `docs/operator/RELEASE_LEDGER.md` |
| Roadmap/doc routing | `docs/operator/ROADMAP_INDEX.md` |
| Release readiness boundary | `docs/operator/RELEASE_READINESS_AUDIT.md` |
| Core workflow proof expectations | `docs/operator/CORE_WORKFLOW_PROOF.md` |
| External pack lifecycle | `docs/design/PACK_LIFECYCLE.md` and `docs/design/PACK_ACQUISITION.md` |
| External pack format and safety | `docs/design/EXTERNAL_PACK_FORMAT.md` |
| Plan Mode policy | `docs/design/PLAN_MODE_POLICY.md` |
| Managed local services and sandboxed tools | `docs/design/MANAGED_LOCAL_SERVICES_AND_SANDBOXED_TOOLS.md` |
| Managed SearXNG operator details | `docs/operator/SAFE_WEB_SEARCH.md` |
| Managed-action reliability | `docs/design/MANAGED_ACTION_RELIABILITY_STANDARD.md` and `docs/operator/MANAGED_ACTION_RELIABILITY_AUDIT.md` |
| Persistent journal scope and limits | `docs/design/PERSISTENT_MANAGED_ACTION_JOURNAL.md` |
| Local model provider boundaries | `docs/operator/LOCAL_MODEL_PROVIDER_SUPPORT.md` |
| Install/operate/backup/doctor | `docs/operator/SETUP.md`, `docs/operator/OPERATIONS.md`, `docs/operator/BACKUP_RESTORE.md`, `docs/operator/doctor.md` |

## Supporting Docs

These remain useful for implementation detail, audits, or historical context,
but the table above is the first routing layer.

### Design

- `docs/design/ACTION_TOOL_INTENT_AUDIT.md`
- `docs/design/ASSISTANT_AGENT_PLANNING.md`
- `docs/design/ASSISTANT_FIRST_ARCHITECTURE.md`
- `docs/design/BYPASS_BEHAVIOR.md`
- `docs/design/CANONICAL_HANDOFF_V3.md`
- `docs/design/CAPABILITY_SETUP_UX.md`
- `docs/design/CHAT_FIRST_UI_HANDOFF.md`
- `docs/design/CONTROL_PLANE_AUDIT.md`
- `docs/design/DOCKER_HELPER_SKILL_PACK.md`
- `docs/design/LLM_SETUP_HANDOFF.md`
- `docs/design/MANAGED_ACTION_RECOVERY.md`
- `docs/design/MANAGED_ADAPTER_LANE.md`
- `docs/design/MANAGED_LOCAL_SERVICES.md`
- `docs/design/MANAGED_PACK_ADAPTERS.md`
- `docs/design/MEMORY_ARCHITECTURE_NEXT_STEP.md`
- `docs/design/MEMORY_UPGRADE_GATE.md`
- `docs/design/MODEL_SCOUT_V2.md`
- `docs/design/MODEL_SELECTION_POLICY.md`
- `docs/design/NATIVE_SKILLS_AUDIT.md`
- `docs/design/RUNTIME_ARCHITECTURE.md`
- `docs/design/RUNTIME_GAP_ANALYSIS.md`
- `docs/design/RUNTIME_TRUTH_SERVICE.md`
- `docs/design/SAFE_MODE.md`
- `docs/design/SYSTEM_ARCHITECTURE_MAP.md`
- `docs/design/TELEGRAM_API_BACKEND_HANDOFF.md`
- `docs/design/TELEGRAM_EXTRACTION_CHECKLIST.md`
- `docs/design/TELEGRAM_THIN_ADAPTER_PLAN.md`
- `docs/design/UI_SURFACE_REPORT.md`

### Operator

- `docs/operator/EXTERNAL_PACK_PROOF_SET.md`
- `docs/operator/GOLDEN_PATH.md`
- `docs/operator/KNOWN_LIMITS.md`
- `docs/operator/LIVE_BEHAVIOR_SMOKES.md`
- `docs/operator/LOCAL_SPLIT_AUDIT.md`
- `docs/operator/LOOKHERE.md`
- `docs/operator/RELEASE.md`
- `docs/operator/RELEASE_NOTES_TEMPLATE.md`
- `docs/operator/USING_STABLE_VS_DEV.md`
- `docs/operator/telegram_setup.md`

### Product, History, And Archive

- `docs/assistant_viability.md`
- `docs/control_plane.md`
- `docs/history/HANDOFF_GOOD_ASSISTANT_FOUNDATION.md`
- `docs/history/STATUS_DELTA.md`
- `docs/archive/CANONICAL_TRACKING.md`
- `docs/archive/FINAL_CHECK.md`
- `docs/archive/RELEASE_NOTES.md`

Archive and history docs are historical or supporting. Do not delete them just
because they overlap current docs.

## Current Next Work

1. Keep the current track on final release proof, not broad new infrastructure.
   The current checkpoint is `v0.2.0-live-usefulness-proof` at `e5dc9f8`.
   Live managed SearXNG search and starter text-pack use have been proven
   through user-facing `/chat` and API paths.
2. Keep managed SearXNG live verification separate from isolated proof:
   `prove_core_workflows.py` can honestly report search `BLOCKED` when no
   backend is configured, while live `/search/status` proves the configured
   managed SearXNG runtime.
3. Continue using Plan Mode for new user-facing mutators: classify first,
   preview a plan, require explicit confirmation, validate apply, journal the
   write, verify the result, and roll back only owned resources.
4. If adding more managed local services or sandboxed tool/MCP runtimes, start
   from `docs/design/MANAGED_LOCAL_SERVICES_AND_SANDBOXED_TOOLS.md`.
5. Before any public trial, run the final fresh Debian VM install proof and
   rerun the verification command groups in
   `docs/operator/RELEASE_LEDGER.md`.

## Do Not Start Until Later

- Fresh Debian VM install proof: now the next release-track step, but do not
  start it until explicitly requested because it is high-cost and
  time-consuming.
- Broad persistent-journal rollout beyond the proof-critical flows.
- Startup auto-recovery that mutates state.
- Direct llama.cpp binary/library management.
- Broad MCP/sandboxed tool runtime execution.
- Arbitrary shell, package install, filesystem write, Docker/Podman, or
  systemctl exposure as normal assistant actions.
- Semantic memory default-on promotion.
- External pack code execution or pack install/import from web search results.

## Existing Docs Discovered Under `docs/`

This list was generated from `find docs -maxdepth 3 -type f | sort` on
2026-06-14.

```text
docs/archive/CANONICAL_TRACKING.md
docs/archive/FINAL_CHECK.md
docs/archive/RELEASE_NOTES.md
docs/assistant_viability.md
docs/control_plane.md
docs/design/ACTION_TOOL_INTENT_AUDIT.md
docs/design/ASSISTANT_AGENT_PLANNING.md
docs/design/ASSISTANT_FIRST_ARCHITECTURE.md
docs/design/BYPASS_BEHAVIOR.md
docs/design/CANONICAL_HANDOFF_V3.md
docs/design/CAPABILITY_SETUP_UX.md
docs/design/CHAT_FIRST_UI_HANDOFF.md
docs/design/CONTROL_PLANE_AUDIT.md
docs/design/DOCKER_HELPER_SKILL_PACK.md
docs/design/EXTERNAL_PACK_FORMAT.md
docs/design/LLM_SETUP_HANDOFF.md
docs/design/MANAGED_ACTION_RECOVERY.md
docs/design/MANAGED_ACTION_RELIABILITY_STANDARD.md
docs/design/MANAGED_ADAPTER_LANE.md
docs/design/MANAGED_LOCAL_SERVICES.md
docs/design/MANAGED_LOCAL_SERVICES_AND_SANDBOXED_TOOLS.md
docs/design/MANAGED_PACK_ADAPTERS.md
docs/design/MEMORY_ARCHITECTURE_NEXT_STEP.md
docs/design/MEMORY_UPGRADE_GATE.md
docs/design/MODEL_SCOUT_V2.md
docs/design/MODEL_SELECTION_POLICY.md
docs/design/NATIVE_SKILLS_AUDIT.md
docs/design/PACK_ACQUISITION.md
docs/design/PACK_LIFECYCLE.md
docs/design/PERSISTENT_MANAGED_ACTION_JOURNAL.md
docs/design/PLAN_MODE_POLICY.md
docs/design/RUNTIME_ARCHITECTURE.md
docs/design/RUNTIME_GAP_ANALYSIS.md
docs/design/RUNTIME_TRUTH_SERVICE.md
docs/design/SAFE_MODE.md
docs/design/SYSTEM_ARCHITECTURE_MAP.md
docs/design/TELEGRAM_API_BACKEND_HANDOFF.md
docs/design/TELEGRAM_EXTRACTION_CHECKLIST.md
docs/design/TELEGRAM_THIN_ADAPTER_PLAN.md
docs/design/UI_SURFACE_REPORT.md
docs/history/HANDOFF_GOOD_ASSISTANT_FOUNDATION.md
docs/history/STATUS_DELTA.md
docs/operator/BACKUP_RESTORE.md
docs/operator/CORE_WORKFLOW_PROOF.md
docs/operator/CURRENT_CHECKPOINT.md
docs/operator/EXTERNAL_PACK_PROOF_SET.md
docs/operator/GOLDEN_PATH.md
docs/operator/KNOWN_LIMITS.md
docs/operator/LIVE_BEHAVIOR_SMOKES.md
docs/operator/LOCAL_MODEL_PROVIDER_SUPPORT.md
docs/operator/LOCAL_SPLIT_AUDIT.md
docs/operator/LOOKHERE.md
docs/operator/MANAGED_ACTION_RELIABILITY_AUDIT.md
docs/operator/OPERATIONS.md
docs/operator/RELEASE.md
docs/operator/RELEASE_NOTES_TEMPLATE.md
docs/operator/RELEASE_READINESS_AUDIT.md
docs/operator/SAFE_WEB_SEARCH.md
docs/operator/SETUP.md
docs/operator/USING_STABLE_VS_DEV.md
docs/operator/doctor.md
docs/operator/telegram_setup.md
docs/product/PROJECT_INTENT.md
```

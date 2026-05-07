# Memory Upgrade Gate

This gate audits the existing Personal Agent memory system before any major rewrite. The current system already has tiered memory, durability, retrieval, status, and reset paths. The next upgrade should make those paths measurable, inspectable, and policy-bound instead of adding another external memory framework.

## Current Architecture Map

### 1. Durable app memory

Files:

- `memory/schema.sql`
- `memory/db.py`

Storage:

- SQLite at `Config.db_path`.
- Domain tables include `projects`, `tasks`, `notes`, `open_loops`, `reminders`, `preferences`, `user_prefs`, `thread_prefs`, `thread_anchors`, `thread_labels`, graph tables, activity/audit/resource tables, and pack-related tables.

Authority:

- Authoritative for explicit local product state: projects, tasks, notes, open loops, reminders, user preferences, thread anchors, and audit/log state.
- `user_prefs` is also the backing store for continuity runtime state. It is not only preference storage.

Trust model:

- Deterministic writes through `MemoryDB` methods are higher-trust than LLM-derived recalls.
- Values still must not override current user input, system policy, or fresh tool/runtime truth.

### 2. Continuity runtime memory

Files:

- `agent/memory_runtime.py`
- `agent/memory_contract.py`
- Used heavily by `agent/orchestrator.py`

Storage:

- JSON records in `user_prefs` under keys such as:
  - `memory_runtime:{user_id}:thread_state`
  - `memory_runtime:{user_id}:pending_items`
  - `memory_runtime:{user_id}:last_meaningful_user_request`
  - `memory_runtime:{user_id}:last_agent_action`
  - `memory_runtime:{user_id}:working_memory_state`
  - `memory_runtime:{user_id}:persistence_status`

Authority:

- Authoritative for follow-up continuity, current topic, last meaningful request/action, and pending confirmations.
- Uses per-key optimistic concurrency through `set_user_pref_if_revision`; cross-key snapshots are not atomic.

Important behavior:

- `resolve_followup()` accepts explicit confirmation/cancel variants, refuses cross-thread matches, reports ambiguity when multiple pending items match, and reports expired pending items.
- `inspect_user_state()` and `inspect_all_state()` report corrupt JSON, persistence conflicts, and working-memory health.
- `normalize_pending_item()` preserves structured `context`, so a pending action can survive save/load.

### 3. Working memory

Files:

- `agent/working_memory.py`
- `agent/orchestrator.py`
- `agent/memory_runtime.py`

Storage:

- Serialized `WorkingMemoryState` under `memory_runtime:{user_id}:working_memory_state`.

Layers:

- Hot turns: recent user/assistant turns kept as chat messages.
- Warm summaries: deterministic summary blocks from older turns.
- Cold state blocks: compacted older state.
- Pinned turn IDs and compaction diagnostics.

Authority:

- Authoritative for bounded active-chat continuity, not for durable factual truth.
- Summaries are compressed context and must be treated as fallible.

Injection path:

- `Orchestrator._prepare_working_memory_for_chat()` loads working memory, appends current transcript/user text, runs `manage_working_memory()`, builds `memory_context_text` with `build_working_memory_context_text()`, and replaces long history with hot messages plus a system-context block.
- `Orchestrator._handle_generic_chat()` injects remembered context into the system prompt under: `Relevant remembered context (use only if it directly helps; never invent beyond it):`.
- Working-memory compaction can also call `_working_memory_durable_ingestor()`, which stores raw chunks and summaries in semantic memory when semantic memory is configured.

### 4. Deterministic memory_v2

Files:

- `agent/memory_v2/types.py`
- `agent/memory_v2/storage.py`
- `agent/memory_v2/ingest.py`
- `agent/memory_v2/retrieval.py`
- `agent/memory_v2/inject.py`
- `agent/api_server.py`

Storage:

- SQLite tables `memory_items`, `memory_events`, and `bootstrap_state`.

Layers:

- `episodic`: timestamped events and bootstrap snapshots.
- `semantic`: current deterministic facts with `fact_key`, `fact_group`, supersession fields, and provenance.
- `procedural`: helper/procedure memory items.

Authority:

- Authoritative only for narrow deterministic facts it creates or stores with provenance.
- Bootstrap semantic promotion is intentionally allowlisted, currently including OS, GPU availability, enabled providers, installed capsules, and available interfaces.

Injection path:

- `AgentRuntime.build_memory_context_for_payload()` calls `select_memory()` and `with_built_context()`.
- Output is `MEMORY[...]` blocks grouped by memory level and attached as `payload["memory_context_text"]` at the API boundary for ordinary `/chat`.

Failure behavior:

- If disabled or unavailable, selection returns empty context and debug metadata.
- If retrieval raises, `AgentRuntime._record_memory_issue()` records the failure and the chat continues without memory_v2 context.

### 5. Semantic vector memory

Files:

- `agent/semantic_memory/types.py`
- `agent/semantic_memory/storage.py`
- `agent/semantic_memory/service.py`
- `agent/memory_ingest.py`
- `agent/conversation_memory.py`
- `agent/api_server.py`

Storage:

- SQLite tables `semantic_sources`, `semantic_chunks`, `semantic_vectors`, and `semantic_index_state`.

Source kinds:

- `conversation`
- `note`
- `document`

Authority:

- Searchable recall only. It is not automatically trusted factual state.
- It stores text chunks with source metadata, embedding target metadata, scope, score, and index health.

Injection path:

- `SemanticMemoryService.build_context_for_payload()` retrieves scoped candidates and renders `SEMANTIC_MEMORY[...]` blocks.
- `AgentRuntime.build_memory_context_for_payload()` can merge semantic context with memory_v2 context, or use semantic context alone when memory_v2 is disabled.
- `/semantic/documents/ingest` and `/semantic/rebuild` are loopback/operator-only mutating endpoints.

Failure behavior:

- Fail-closed. Disabled service, missing index, stale index, partial index, provider unavailability, embedding failure, and dimension mismatch return no candidates with debug reasons.
- Ingestion failures mark index state as stale/partial/failed and return `ok: False` rather than silently claiming durable recall.

### 6. Graph/thread-focus memory

Files:

- `memory/schema.sql`
- `memory/db.py`
- `tests/test_memory_graph.py`
- `tests/test_memory_graph_hygiene.py`

Storage:

- `graph_nodes`, `graph_edges`, `graph_aliases`, `thread_focus`, relation type/mode/constraint tables.

Authority:

- Existing local graph primitives. They are not currently the central chat recall path.
- Do not add graph memory as the next phase unless a concrete measurable use case beats the simpler gate improvements below.

## Audit Answers

### 1. What memory layers currently exist?

The repo has durable app memory, continuity runtime memory, working memory, deterministic memory_v2, semantic vector memory, and graph/thread-focus primitives. The first three are always central to continuity; memory_v2 and semantic memory are optional/configurable retrieval layers.

### 2. Which layer is authoritative for what?

- Current user input is authoritative for the active turn.
- System/developer policy and SAFE MODE override memory.
- Fresh runtime/tool truth overrides remembered runtime state.
- `MemoryDB` domain tables are authoritative for explicit tasks, notes, reminders, preferences, and audit/resource state.
- `MemoryRuntime` is authoritative for pending follow-ups, current topic, last request/action, and working-memory persistence.
- `WorkingMemoryState` is authoritative only for compact active-chat continuity.
- `memory_v2` semantic facts are authoritative only for allowlisted deterministic facts with provenance/currentness.
- Semantic vector memory is never authoritative by itself; it is retrieved evidence.

### 3. What gets injected into chat context, when, and why?

At the `/chat` API boundary, `AgentRuntime.build_memory_context_for_payload()` may attach `memory_context_text` for ordinary chat. Deterministic runtime/setup routes skip retrieval. The memory context can include `MEMORY[...]` from memory_v2 and `SEMANTIC_MEMORY[...]` from semantic vector recall.

Inside the orchestrator, `_handle_generic_chat()` adds selective continuity context for prompts that ask to continue, recap, rewind, reuse preferences, or discuss the larger task. It then calls `_prepare_working_memory_for_chat()`, which builds hot/warm/cold working-memory context and injects it into the system prompt only as relevant remembered context.

The reason is continuity under a bounded context budget, not blind truth replacement.

### 4. What can be searched but not automatically trusted?

Semantic vector memory, working-memory summaries, old episodic events, user/doc/conversation chunks, and graph-derived recall are searchable but must be treated as candidate evidence. They can contain stale, contradictory, user-authored, or prompt-injection content.

### 5. What memory writes are deterministic versus LLM-derived?

Deterministic:

- `MemoryDB` structured writes for app state.
- `MemoryRuntime` thread state, pending items, last request/action, and persistence metadata.
- `memory_v2` bootstrap episodic captures and allowlisted semantic promotions.
- Working-memory turn append, thresholding, and compaction mechanics.
- Semantic storage, chunking, embedding metadata, index state, and retrieval debug.

Potentially LLM-derived or model-influenced:

- Assistant responses later saved as conversation memory.
- User/document text stored in semantic memory.
- Working-memory summaries if future implementations use model summarization. Current `working_memory.py` uses deterministic summarization primitives, but the contract should treat compressed summaries as lossy and untrusted.

### 6. What happens when memory retrieval fails?

The current behavior is mostly fail-closed:

- Semantic selection returns empty candidates with debug reasons for disabled/missing/stale/partial/unavailable states.
- `AgentRuntime.build_memory_context_for_payload()` catches semantic and memory_v2 selection exceptions, records memory issues, and returns empty or partial context.
- `MemoryRuntime.load_working_memory_state()` can report an issue; `_prepare_working_memory_for_chat()` avoids saving when load is corrupt.
- Chat continues without recalled memory where possible.

Gap: failures are observable in `/memory/status` and envelopes, but there is no single upgrade gate that asserts "memory unavailable did not degrade into false memory confidence."

### 7. Can stale or contradictory memory override current user input?

There is no intentional code path where memory should override current input. The prompt explicitly says to use remembered context only if it directly helps and not to invent beyond it.

Risk remains prompt-level: stale semantic chunks or summaries can be injected near the system prompt. The next gate should add tests proving current user input and fresh runtime truth win over contradictory recalled context.

### 8. Can prompt-injection content enter durable memory?

Yes. Conversation, note, document, and working-memory durable ingests can store user-authored or document-authored text. The code labels rendered semantic memory as `SEMANTIC_MEMORY[...]` and JSON-quotes text, but the stored content can still include adversarial instructions.

This is acceptable only if durable memory is treated as untrusted evidence. The next phase should add explicit prompt-injection regression tests for semantic memory, working-memory summaries, and document ingest.

### 9. Can the user inspect/edit/delete memories clearly?

Partially.

Existing inspect/reset surfaces:

- `GET /memory/status` reports continuity, working-memory summaries, memory_v2, semantic memory, table counts, health, persistence model, and last errors.
- `POST /memory/reset` previews destructive reset and requires `confirm=true`; it is loopback/operator-only.
- `GET /semantic/status` reports semantic memory status.
- `POST /semantic/documents/ingest` and `POST /semantic/rebuild` are loopback/operator-only.
- `MemoryRuntime.inspect_user_state()` and `inspect_all_state()` expose detailed continuity state for status reporting.

Gaps:

- No clear unified user-facing list/edit/delete surface for individual memories across continuity, memory_v2, semantic chunks, and domain tables.
- Reset is coarse by component, not scoped to a specific memory item/source/thread.
- Semantic sources/chunks can be reported internally, but the user-facing edit/delete contract is not yet explicit.

### 10. Does follow-up context survive interruption, restart, and long sessions?

Mostly yes for the deterministic continuity path:

- Pending items live in `user_prefs` and are normalized by `memory_contract.py`.
- Follow-up resolution is thread-scoped and tested for ambiguity, expiry, and natural confirmation variants.
- Working memory is serialized into `MemoryRuntime` and tested across persistence, concurrency, and cross-session replay.
- Long-session behavior is tested through working-memory soak/behavioral replay.

Remaining risk:

- Cross-key snapshots are not atomic, so thread state, pending items, and working memory can be individually consistent but temporarily out of sync.
- Very old pending items rely on expiry/status hygiene.
- The current test suite covers many restart paths, but the gate should make "pending action with context survives restart and does not execute without confirmation" a named invariant.

## Existing Tests

Representative coverage:

- Continuity/pending contract: `tests/test_memory_contract.py`, `tests/test_memory_runtime.py`
- Corruption, reset, status, loopback protection: `tests/test_memory_hardening.py`
- API memory injection: `tests/test_api_server_memory_injection.py`
- Semantic service fail-closed and retrieval: `tests/test_semantic_memory_service.py`, `tests/test_api_server_semantic_memory.py`, `tests/test_memory_ingest_semantic.py`
- Deterministic memory_v2 retrieval: `tests/test_memory_v2_retrieval.py`
- Working-memory compaction/soak/cross-session/concurrency: `tests/test_working_memory.py`, `tests/test_working_memory_soak.py`, `tests/test_working_memory_behavioral_replay.py`, `tests/test_working_memory_cross_session_replay.py`, `tests/test_working_memory_concurrency.py`, `tests/test_working_memory_persistence_hardening.py`
- Orchestrator follow-up and long-chat paths: `tests/test_orchestrator.py`, `tests/test_api_server.py`
- Graph hygiene: `tests/test_memory_graph.py`, `tests/test_memory_graph_hygiene.py`
- Prompt-regression fixtures: `tests/test_prompt_regression_pack.py`

## Recommended Test Matrix

| Area | Invariant | Existing coverage | Add/strengthen |
| --- | --- | --- | --- |
| Authority ordering | Current user input beats contradictory memory | Partial via behavioral replay | Add direct chat test with stale memory saying opposite of current instruction |
| Runtime truth | Fresh tool/runtime state beats stale recalled runtime state | Partial via resource follow-up tests | Add memory-injected stale provider/process fact regression |
| Prompt injection | Memory text cannot issue instructions to assistant | Prompt pack coverage exists, memory-specific unclear | Add semantic/document/working-memory injection regressions |
| Retrieval failure | Missing/stale/partial semantic index produces empty recall and debug | Covered in semantic service tests | Add API envelope assertion for debug without false recall |
| Follow-up restart | Pending item plus context survives restart but requires confirmation | Partial | Add named end-to-end restart test with pending context and explicit confirm/cancel |
| Long session | Compaction remains bounded and preserves task-critical facts | Covered | Add status metric assertions after soak |
| User inspect | User can see memory layers, counts, health, and warnings | Covered for `/memory/status` | Add UI/API contract tests for item-level list once implemented |
| User delete | Reset requires preview/confirm and loopback | Covered | Add source/item-scoped delete tests when added |
| Semantic source lifecycle | Ingest, rebuild, stale-index detection, source replacement | Covered | Add delete-source and reindex tests when item delete exists |
| Concurrency | Stale writes do not clobber newer continuity state | Covered | Add cross-key consistency warning assertions |
| Capability interaction | Memory does not turn unsupported capability into false claim | Capability Rescue tests separate | Add chat test with stale memory claiming skill is installed |
| Privacy | Status/report endpoints avoid leaking full sensitive content by default | Partial | Add redaction/summarization checks for future inspect APIs |

## SOTA Gap Analysis

The repo already has the practical foundations seen in mature assistant memory systems: recency tiers, compaction, durable facts, semantic recall, health/status, preview-before-reset, and fail-closed retrieval. The main gaps are governance and measurement, not raw capability.

High-value gaps:

- Authority labels are implicit. Injected context does not carry a machine-readable trust tier such as `authoritative`, `candidate_evidence`, `compressed_summary`, or `stale_runtime_fact`.
- There is no unified memory provenance ledger explaining why a specific memory was injected into a specific answer.
- Prompt-injection handling depends on prompt wording and quoting, not memory-specific tests and labels.
- User inspect/edit/delete is component-level, not item-level.
- Contradiction handling is uneven: memory_v2 semantic facts have current/superseded fields, while semantic vector memory primarily depends on source scope, recency, pinned status, and retrieval scores.
- Memory quality is not scored over time. There is status health, but not recall precision, false-positive rate, stale-hit rate, or "current input overrode memory" metrics.
- Cross-key continuity state uses per-key concurrency, so a gate should measure and surface snapshot skew instead of pretending atomicity.

Lower-payoff or premature gaps:

- External memory frameworks would add integration risk without solving the local authority/test gaps.
- Graph memory is already present at the storage layer, but the next phase should not promote it unless a measurable workflow needs relationship traversal over simpler scoped retrieval.

## Bugs Or Risks Found

No blocking bug was found during this audit.

Risks to track:

- Durable prompt injection can enter semantic memory through conversation, note, document, or working-memory durable ingestion.
- Stale semantic chunks can be injected as remembered context unless retrieval and prompt discipline keep them as evidence.
- Working-memory summaries are compact and lossy; they should not be treated as exact transcript.
- `/memory/reset` is destructive but correctly preview/confirm gated and loopback/operator-only. Future item-level delete should keep the same preview/confirm pattern.
- Semantic ingest/rebuild are loopback/operator-only, but future UI affordances must not create one-click destructive or broad-ingest paths.
- Cross-key continuity is not atomic; consistency diagnostics should be explicit.
- There are several memory stores with overlapping names. Without an upgrade gate, future changes can accidentally make the wrong layer authoritative.

## Ranked Improvements

1. High payoff, low effort: define memory authority labels and include them in injected context diagnostics.
2. High payoff, low effort: add regression tests for current-input-over-memory and fresh-runtime-truth-over-memory.
3. High payoff, low effort: add memory-specific prompt-injection tests for semantic/document/working-memory content.
4. High payoff, medium effort: add a memory injection audit object to chat envelopes with selected IDs, source kind, authority tier, score, age, and reason.
5. High payoff, medium effort: add user-facing item/source list and delete preview APIs for semantic memory and memory_v2, with loopback/operator guard for destructive changes.
6. Medium payoff, low effort: expose cross-key continuity snapshot-skew warnings in `/memory/status`.
7. Medium payoff, medium effort: add contradiction/currentness handling for semantic vector sources, starting with source replacement and stale-hit warnings.
8. Medium payoff, medium effort: add quality metrics: selected/not-used rate, stale recall rate, retrieval failure rate, and user correction rate.
9. Low payoff now, high effort: promote graph memory into primary chat recall.
10. Low payoff now, high effort: add an external memory framework.

## Recommended Next Phase

Phase name: Memory Upgrade Gate v1.

Scope:

- Formalize memory authority labels.
- Add regression tests proving current input, policy, and fresh runtime truth outrank memory.
- Add memory-specific prompt-injection tests.
- Add injection diagnostics to chat envelopes without changing recall behavior.
- Draft item/source-level inspect/delete contracts before implementing new mutation endpoints.

Non-goals:

- Do not rewrite memory.
- Do not add an external memory framework.
- Do not add graph memory as a default retrieval path.
- Do not make semantic vector recall authoritative.
- Do not silently store arbitrary external content as trusted memory.
- Do not add automatic memory edits based on LLM inference without provenance, user visibility, and rollback/delete paths.

## Files Audited

- `agent/memory_runtime.py`
- `agent/memory_contract.py`
- `agent/working_memory.py`
- `agent/memory_ingest.py`
- `agent/conversation_memory.py`
- `agent/memory_v2/types.py`
- `agent/memory_v2/storage.py`
- `agent/memory_v2/ingest.py`
- `agent/memory_v2/retrieval.py`
- `agent/memory_v2/inject.py`
- `agent/semantic_memory/types.py`
- `agent/semantic_memory/storage.py`
- `agent/semantic_memory/service.py`
- `agent/orchestrator.py`
- `agent/api_server.py`
- `memory/schema.sql`
- `memory/db.py`
- `tests/test_memory_contract.py`
- `tests/test_memory_runtime.py`
- `tests/test_memory_hardening.py`
- `tests/test_api_server_memory_injection.py`
- `tests/test_semantic_memory_service.py`
- `tests/test_api_server_semantic_memory.py`
- `tests/test_memory_ingest_semantic.py`
- `tests/test_memory_v2_retrieval.py`
- `tests/test_working_memory.py`
- `tests/test_working_memory_soak.py`
- `tests/test_working_memory_behavioral_replay.py`
- `tests/test_working_memory_cross_session_replay.py`
- `tests/test_working_memory_concurrency.py`
- `tests/test_working_memory_persistence_hardening.py`
- `tests/test_memory_graph.py`
- `tests/test_memory_graph_hygiene.py`
- `tests/test_orchestrator.py`
- `tests/test_api_server.py`

## Files Changed

- `docs/design/MEMORY_UPGRADE_GATE.md`

# Memory Architecture Next Step

This note captures the current memory system, the target architecture for optional semantic recall, and the phased implementation plan.

## Current State

### Deterministic memory is active and canonical

The source of truth for structured memory remains the deterministic SQLite-backed runtime:

- `memory/db.py`
- `memory/schema.sql`
- `agent/memory_contract.py`
- `agent/memory_runtime.py`
- `agent/orchestrator.py`
- `agent/setup_chat_flow.py`
- `agent/nl_router.py`
- `agent/telegram_bridge.py`
- `telegram_adapter/bot.py`
- `agent/cli.py`
- `skills/core/handler.py`
- `skills/recall/handler.py`

This layer owns thread state, prefs, anchors, labels, open loops, reminders, graph structures, and the deterministic memory summary/snapshot path.

### Optional memory_v2 exists, but it is still deterministic

`agent/memory_v2/*` and the `AGENT_MEMORY_V2_ENABLED` gate provide an optional helper store with structured episodic and semantic-fact rows. Today it does not do embedding-based semantic search. It is useful as a bounded compatibility layer and bootstrap store, but it is not the semantic layer this plan adds.

### Prompt injection is already bounded

The current chat boundary injects memory through:

- `agent/api_server.py::build_memory_context_for_payload`
- `agent/llm/chat_preflight.py::_memory_prefix_messages`
- `agent/orchestrator.py::_selective_chat_memory_context`

That path is where semantic recall should be added, not by inventing a second prompt path.

### Existing note/conversation/document-adjacent paths

- Notes are persisted in `memory/schema.sql::notes` via `skills/core/handler.py::remember_note`
- Conversation and intent traces currently flow through `agent/conversation_memory.py` and `agent/memory_ingest.py`, but those modules are compatibility shims today
- Document/file content is not yet indexed semantically; it is only represented indirectly through other runtime facts, reports, and snapshots

## Target State

### Principle

Deterministic memory remains authoritative. Semantic memory is an optional helper layer that can suggest relevant context, but it must never overwrite facts, thread state, prefs, or other canonical memory.

### Memory layers

1. Deterministic structured memory
   - Thread state, prefs, anchors, labels, open loops, reminders, graph facts, and note/task rows remain canonical
   - The runtime summary/snapshot path stays deterministic and inspectable

2. Semantic recall for conversations
   - Conversation turns can be embedded and indexed as episodic semantic candidates
   - Recall is optional, bounded, and tag/scope aware

3. Semantic recall for notes
   - Notes stay canonical in SQLite
   - Their text can be embedded as optional semantic candidates with provenance back to the note row

4. Semantic recall for documents/files
   - File or document chunks can be indexed explicitly when a caller chooses to ingest them
   - Semantic chunks always carry source provenance and scope

### Retrieval contract

- Always retrieve deterministic memory first
- Query semantic memory only when it is enabled, configured, and healthy enough to do so
- Merge semantic candidates after deterministic context
- Prefer explicit scoring and provenance over opaque heuristics
- Fail closed when embeddings are unavailable, stale, misconfigured, or dimension-mismatched

### Storage contract

Semantic storage should use explicit SQLite tables for:

- sources
- chunks
- embeddings/vectors
- index state / health

Every recalled semantic item should carry:

- source kind
- source reference
- scope
- thread/project association when available
- embedding model/provider provenance
- timestamps
- pin state
- index state

## Retrieval Flow

1. Build deterministic context
   - Thread state and snapshot summary
   - Deterministic `memory_v2` selection if enabled
   - Any other authoritative runtime facts

2. Query semantic context
   - Build a semantic query from the chat text
   - Resolve the configured embed model explicitly
   - Check provider capability and index health
   - Skip semantic search if the index is missing, stale, or incompatible

3. Merge candidates
   - Deduplicate by provenance/text hash
   - Keep deterministic items first
   - Rank semantic candidates by explicit score components
   - Truncate to a bounded prompt budget

4. Inject prompt text
   - Deterministic block first
   - Semantic block second
   - Both blocks must remain inspectable in the debug envelope

## Storage Model

The semantic layer should add tables that track:

- source rows with provenance and scope
- chunk rows with chunk boundaries and content hashes
- vector rows with provider/model/dimension metadata
- index state rows with status, stale markers, and last error details

The store should not require embeddings to exist before source metadata is captured. It should, however, refuse to use partially indexed or incompatible data for retrieval.

## Embedding Model Selection

The configured `embed_model` is the canonical embedding target.

Selection rules:

- Accept only models that the registry marks as embedding-capable
- Resolve provider availability explicitly before embedding
- Track dimensions from the first successful embedding response
- Mark the index stale when the configured model changes
- Skip semantic recall when the stored dimensions do not match the current target

## Failure Behavior

Semantic memory must degrade gracefully:

- Embeddings unavailable: deterministic memory still works, semantic recall is skipped
- Provider does not support embeddings: semantic recall is skipped
- Index missing: semantic recall is skipped
- Index stale: semantic recall is skipped until reindexed
- Dimension mismatch: semantic recall is skipped
- Partial ingestion: only fully indexed chunks may be selected
- Disabled by config: semantic layer is inactive and deterministic behavior is unchanged

## API and Runtime Integration

Semantic memory should be integrated at these points:

- Ingestion:
  - conversation turns
  - notes
  - explicit document/file ingestion calls

- Retrieval:
  - chat prompt preparation
  - optional memory lookup endpoints

- Status surfaces:
  - runtime status
  - chat/memory debug envelopes
  - health summaries that expose semantic index state and embedding-model health

## Phased Plan

### Phase 1

- Add explicit embedding support to provider interfaces
- Add semantic memory tables and state tracking
- Add semantic memory service with fail-closed retrieval
- Wire semantic recall into the existing chat-memory injection path
- Hook conversation and note ingestion into the semantic helper layer
- Add tests for deterministic-only, embeddings-enabled, disabled fallback, dimension mismatch, stale/missing index, ranking, and prompt safety

### Phase 2

- Add document/file ingestion entry points that call one canonical semantic
  ingestion service
- Add explicit reindex/rebuild operations that rebuild vectors from stored
  source/chunk rows, not from live heuristics
- Add richer status and operator surfaces for semantic health and recovery
- Keep compatibility shims as thin wrappers only if they still have live
  callers; otherwise consolidate them behind the canonical semantic service

### Phase 2 Implementation Rules

- `semantic_sources` are provenance records, `semantic_chunks` are the indexed
  content slices, and `semantic_vectors` are derived artifacts
- deterministic memory remains the authoritative source of truth and always
  stays ahead of semantic recall in prompt assembly
- semantic recall may be queried only when the configured embedding target and
  stored index state agree, or when a repair flow is explicitly rebuilding the
  index
- repair flows must fail closed and surface a concrete recovery reason instead
  of silently skipping mismatched or partial data
- operator surfaces should show:
  - current embedding target
  - stored index state
  - recovery state
  - the next recommended operator action when the index is stale, missing, or
    partial

### Phase 3

- Consider pruning compatibility shims that become unused after the semantic service is stable
- Revisit whether `memory_v2` should keep serving as a deterministic compatibility layer or be folded into the canonical deterministic memory surface

# Dynamic Context Degradation Spec

Goal: prevent hard context-window amnesia by degrading conversation history gradually instead of dropping it all at once.

## Outcome

The agent should maintain a 3-layer working memory:

1. **Hot context** — most recent turns, verbatim.
2. **Warm context** — older turns, chunk summaries.
3. **Cold context** — oldest history compressed to durable facts / unresolved threads / user preferences.

The agent must never wait until the prompt is already too large. It should begin degrading at a soft threshold and do so incrementally.

---

## Core Design

### 1. Context Budgeting

Define explicit token budgets.

Example:

```python
MAX_CONTEXT_TOKENS = model_limit
RESERVED_OUTPUT_TOKENS = 2000
RESERVED_SYSTEM_AND_TOOLS = 3000
WORKING_MEMORY_BUDGET = MAX_CONTEXT_TOKENS - RESERVED_OUTPUT_TOKENS - RESERVED_SYSTEM_AND_TOOLS
SOFT_DEGRADE_THRESHOLD = int(WORKING_MEMORY_BUDGET * 0.72)
HARD_DEGRADE_THRESHOLD = int(WORKING_MEMORY_BUDGET * 0.85)
PANIC_THRESHOLD = int(WORKING_MEMORY_BUDGET * 0.93)
```

### 2. Layer Policy

#### Hot context
- Keep newest turns verbatim.
- Target ~35–50% of working memory budget.
- Contains raw user/assistant turns, recent tool outputs, and active unresolved work.

#### Warm context
- Replace older raw chunks with structured summaries.
- Each summary should represent a bounded slice of conversation, for example 6–12 turns.
- Preserve chronology.

#### Cold context
- Extract only durable information:
  - user preferences
  - established facts
  - project state
  - decisions made
  - unresolved threads
  - promises/constraints still relevant
- This can feed semantic memory or a compact long-term state block.

---

## Required Behaviors

### A. Progressive degradation, not hard truncation

Never drop old context all at once except at panic threshold.

### B. Structured summaries only

Do not replace history with vague prose. Use structured summaries.

Recommended summary schema:

```json
{
  "range": {
    "start_turn_id": "...",
    "end_turn_id": "..."
  },
  "time_span": "optional ISO timestamps",
  "topic": "brief label",
  "facts": ["..."],
  "decisions": ["..."],
  "open_threads": ["..."],
  "user_preferences": ["..."],
  "artifacts": ["files, docs, outputs, code touched"],
  "tool_results": ["important grounded findings only"],
  "compression_level": 1
}
```

Compression levels:
- `1` = light summary
- `2` = tighter abstraction
- `3` = distilled facts only

### C. Summary drift control

Do not endlessly summarize summaries into summaries without guardrails.

Rules:
- Raw turns may be summarized into level-1 summaries.
- Level-1 summaries may be merged into a level-2 summary.
- Level-2 summaries may be merged into a level-3 summary.
- Never go beyond level 3.
- Keep metadata showing provenance.

### D. Semantic memory extraction happens before eviction

Before discarding detailed history, extract durable facts into your semantic memory / long-term store.

### E. Active thread pinning

Any thread still in progress should be pinned so it remains verbatim or only lightly compressed, even if older.

Pinned examples:
- current coding task
- unresolved bug investigation
- user decision awaiting follow-up
- active plan being executed

---

## Data Model

## Conversation units

```python
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]

@dataclass
class Turn:
    turn_id: str
    role: Role
    text: str
    token_count: int
    created_at: str
    pinned: bool = False
    topic_hint: str | None = None
    references: list[str] = field(default_factory=list)
```

## Summary blocks

```python
@dataclass
class SummaryBlock:
    block_id: str
    source_turn_ids: list[str]
    start_turn_id: str
    end_turn_id: str
    token_count: int
    compression_level: int
    topic: str
    facts: list[str]
    decisions: list[str]
    open_threads: list[str]
    user_preferences: list[str]
    artifacts: list[str]
    tool_results: list[str]
    created_at: str
```

## Working memory state

```python
@dataclass
class WorkingMemoryState:
    hot_turns: list[Turn]
    warm_summaries: list[SummaryBlock]
    cold_state_blocks: list[SummaryBlock]
    pinned_turn_ids: set[str]
    last_compaction_at: str | None = None
```

---

## Degradation Algorithm

## High-level loop

Run every turn before prompt assembly.

```python
def manage_working_memory(state: WorkingMemoryState, budget: int) -> WorkingMemoryState:
    used = estimate_state_tokens(state)

    if used < SOFT_DEGRADE_THRESHOLD:
        return state

    state = maybe_extract_durable_memory(state)

    while estimate_state_tokens(state) > SOFT_DEGRADE_THRESHOLD:
        if has_old_uncompressed_turns(state):
            state = summarize_oldest_unpinned_raw_chunk(state)
            continue

        if has_mergeable_level1_summaries(state):
            state = merge_summaries(state, from_level=1, to_level=2)
            continue

        if has_mergeable_level2_summaries(state):
            state = merge_summaries(state, from_level=2, to_level=3)
            continue

        break

    if estimate_state_tokens(state) > HARD_DEGRADE_THRESHOLD:
        state = enforce_hard_compaction(state)

    if estimate_state_tokens(state) > PANIC_THRESHOLD:
        state = emergency_trim(state)

    return state
```

---

## Chunk selection policy

When summarizing raw turns:

1. Choose the **oldest contiguous unpinned** span.
2. Prefer chunks of 6–12 turns or a token range like 1200–2500 tokens.
3. Do not split in the middle of a tightly coupled exchange unless required.
4. Preserve at least the most recent N user turns verbatim.

Example:

```python
def select_chunk_for_summarization(turns: list[Turn]) -> list[Turn]:
    candidates = [t for t in turns if not t.pinned]
    # choose oldest contiguous region above minimum size
    ...
```

---

## Prompt assembly order

When building the model prompt:

1. System/runtime instructions
2. Relevant cold state blocks (only if relevant)
3. Warm summaries in chronological order
4. Hot turns verbatim
5. Current user turn
6. Retrieved semantic memory
7. Tool context

Important:
- Do not always inject every cold block.
- Relevance filter cold state and semantic memory by current task.

---

## What gets extracted to long-term memory

Before compression, run a durable-memory extractor.

Store only information likely to matter later:
- stable user preferences
- project facts
- design decisions
- environment facts
- recurring workflows
- commitments still relevant

Do not store:
- transient chatter
- one-off filler
- outdated intermediate reasoning

Example extractor output:

```json
{
  "facts": [
    "User prefers concise answers.",
    "Project uses local-first architecture.",
    "Current repo path is ~/personal-agent."
  ],
  "decisions": [
    "Use semantic memory as canonical long-term store.",
    "Use structured JSON summaries instead of prose summaries."
  ],
  "open_threads": [
    "Implement dynamic context degradation in the agent memory stack."
  ]
}
```

---

## Hard compaction policy

If over hard threshold:

1. Summarize oldest raw turns immediately.
2. Merge level-1 summaries where possible.
3. Drop nonessential tool chatter already represented in summaries.
4. Keep pinned work and the most recent user-facing turns intact.

---

## Emergency trim policy

This is the only place hard dropping is allowed.

Rules:
- Never drop system instructions.
- Never drop the current task.
- Never drop pinned items.
- Drop oldest nonessential low-value material first.
- Prefer dropping verbose tool logs already captured elsewhere.

Log every emergency trim event so it is observable and debuggable.

---

## Summary generation rules

The summarizer should be instructed to:

- preserve factual commitments
- preserve unresolved tasks
- preserve concrete decisions
- preserve user preferences and constraints
- avoid stylistic fluff
- avoid inventing motivations or implications
- keep output compact and structured
- mark uncertainty explicitly

Suggested summarizer prompt contract:

```text
Summarize this conversation chunk into structured memory.
Preserve only durable facts, decisions, open threads, user preferences, important artifacts, and grounded tool outcomes.
Do not include filler.
Do not speculate.
Return strict JSON matching schema.
```

---

## Anti-drift safeguards

1. Each summary stores source turn ids.
2. Merged summaries store child block ids.
3. Periodically regenerate summaries from raw history if raw is still available.
4. Never let summaries become the only source of truth for facts that should be in semantic memory.

---

## Observability

Expose operator-facing state such as:

```json
{
  "working_memory": {
    "hot_tokens": 4200,
    "warm_tokens": 1600,
    "cold_tokens": 500,
    "total_tokens": 6300,
    "soft_threshold": 7000,
    "hard_threshold": 8200,
    "panic_threshold": 9000,
    "last_compaction_at": "...",
    "last_compaction_action": "summarized_raw_chunk",
    "emergency_trim_count": 0
  }
}
```

Also log:
- chunk ranges summarized
- summary levels merged
- durable facts extracted
- pinned items preserved
- emergency trims

---

## Integration points for your system

This should sit between:

1. **conversation ingestion**
2. **prompt assembly**

Suggested flow per turn:

```python
def process_turn(state, new_turn):
    state.hot_turns.append(new_turn)
    state = extract_and_store_durable_memory(state, new_turn)
    state = manage_working_memory(state, WORKING_MEMORY_BUDGET)
    prompt = assemble_prompt(state)
    return state, prompt
```

### Recommended architecture fit for your project

Given your stack already has semantic memory:

- **Hot turns** = immediate conversation buffer
- **Warm summaries** = rolling working-memory compression
- **Cold state** = durable compact state block
- **Semantic memory** = authoritative long-term retrieval layer

That means summaries are not your final memory system. They are only a working-memory pressure valve.

---

## Minimal implementation plan

### Phase 1
- Add token accounting for prompt assembly.
- Add hot/warm/cold state structures.
- Add chunk summarization at soft threshold.
- Add observability counters.

### Phase 2
- Add durable-memory extraction before eviction.
- Add pinned thread support.
- Add level-1 to level-2 summary merging.

### Phase 3
- Add relevance filtering for cold blocks.
- Add emergency trim safeguards.
- Add tests for drift, pinning, and threshold behavior.

---

## Test cases Codex should implement

### 1. No compaction under threshold
- Given token usage below soft threshold
- Expect no summarization

### 2. Oldest unpinned chunk summarized first
- Given enough raw turns to exceed threshold
- Expect oldest unpinned chunk replaced with summary

### 3. Pinned turns preserved
- Given old pinned turns and over-budget context
- Expect other material compacted first

### 4. Summary merge occurs only after raw chunks exhausted
- Given only level-1 summaries remain
- Expect merge to level 2

### 5. Durable facts extracted before raw eviction
- Given user preference present in old chunk
- Expect preference stored before raw chunk removed

### 6. Emergency trim never removes current task
- Given panic threshold exceeded
- Expect current task and pinned content remain

### 7. Prompt assembly keeps chronology
- Given warm summaries and hot turns
- Expect summaries before recent verbatim turns

### 8. Tool log trimming prefers low-value verbose outputs
- Given bulky tool chatter and compact summary exists
- Expect tool chatter removed before valuable turns

---

## Codex work order

Implement dynamic context degradation for the agent working-memory pipeline.

Requirements:

1. Add explicit working-memory token budgeting with soft, hard, and panic thresholds.
2. Introduce a 3-layer context model: hot verbatim turns, warm structured summaries, cold compact state.
3. Add progressive compaction of the oldest unpinned raw chunks into strict structured summaries.
4. Add summary merging with capped compression levels (1 -> 2 -> 3 only).
5. Extract durable memory before evicting detailed history.
6. Add pinning support for active threads so important work is not prematurely compressed.
7. Ensure prompt assembly preserves chronology and prefers relevant cold state only.
8. Add observability and tests covering thresholds, pinning, extraction-before-eviction, and emergency trim behavior.

Constraints:

- No prose-only summaries.
- No hard truncation except emergency trim.
- No silent loss of unresolved work.
- Keep implementation deterministic where practical.
- Prefer structured JSON outputs and canonical accounting.

Deliverables:

- code changes
- tests
- brief operator-facing docs
- updated project status notes if that file exists

---

## My recommendation

For your system specifically: treat this as a **working-memory compactor**, not the main memory system. The real durable memory should stay semantic/retrieval-backed. This layer only prevents catastrophic prompt overflow and sudden amnesia.


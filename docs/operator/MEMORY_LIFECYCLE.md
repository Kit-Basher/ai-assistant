# Memory Lifecycle

Personal Agent memory must be understandable and user-controlled. Users should
not need database knowledge to see what is remembered, opt out for a turn,
preview memory deletion, or understand memory scope.

## Scope Model

- Saved long-term memory: preferences, anchors, summaries, open loops, and
  durable user/project context.
- Current thread context: the visible conversation and short-term working
  context for the current thread.
- Pending confirmations/actions: Plan Mode and follow-up state waiting for a
  user reply.
- External tools/current facts: search, status, and other tools. These are not
  disabled by a saved-memory opt-out unless the user asks for that separately.

`do not use memory for this` means saved memory and prior conversation context
for the current turn only. It does not disable tools, search, or current facts.

## Current Product Flow

These installed `/chat` prompts are deterministic:

- `what do you remember about me?`
- `show memory status`
- `do not use memory for this`
- `disable memory for this thread`
- `enable memory for this thread`
- `disable memory globally`
- `enable memory globally`
- `forget what you remember about X`
- `delete all memory about me`
- `export my memory`
- `redact sensitive memory`
- `clean up duplicate memories`

Inspection and current-turn opt-out are read-only/non-destructive.

Thread/global enable/disable, forget-topic, delete-all, export, redact, and
dedupe currently return Plan Mode-style previews. The previews include scope,
resources, rollback policy, and an explicit confirmation requirement. The
executor behind those lifecycle previews is still partial; confirmation does
not silently delete, export, redact, or rewrite memory.

## Proof

Run:

```bash
python scripts/memory_lifecycle_smoke.py
```

The smoke talks to `http://127.0.0.1:8765` and verifies:

- memory inspection is understandable and does not invent sensitive specifics
- memory status distinguishes saved memory, thread context, pending actions,
  and external tools/current facts
- current-turn no-memory wording is deterministic and precise
- global/thread memory controls produce previews
- forget/delete/export/redact/dedupe produce previews
- cancel works
- stale confirmation does not delete memory
- the repo working tree status is unchanged by the smoke

## Remaining Gaps

Executor Authorization Migration v1 classifies memory mutation lanes as
`memory.forget`, `memory.export`, `memory.redact`, or `memory.compact` and
requires Universal Mutation Plan metadata for those lanes.

Do not claim full memory lifecycle execution completion yet. Bounded executors
for thread/global memory toggles, export, redaction, dedupe, forget-topic, and
delete-all are still future work where not already implemented; they must keep
Plan Mode confirmation and fail closed with `mutated=false` while preview-only.

Audit v2E centrally authorizes the shipping `/memory/reset` surface and
explicit assistant preference-memory store/forget paths. Reset remains
irreversible and single-use. Existing preview-only lifecycle variants remain
honestly preview-only; this audit does not invent missing restore/merge
executors or silently treat them as internal bookkeeping.

The reset component set is deliberately limited to continuity runtime keys,
Memory v2 items/events/bootstrap state, and semantic sources/chunks/vectors/
index state. It does not erase project/task graph records, general notes or
preferences, prior exports, external backups, or recovery artifacts; no
tombstone or automatic restoration contract is claimed. Restoring a backup can
therefore reintroduce previously erased material and requires separate review
and authorization. Indeterminate reset execution is never retried
automatically.

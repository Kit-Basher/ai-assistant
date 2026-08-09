# General Task Loop (Work Package 3)

Status: implementation design for the bounded native task coordinator introduced after the proven 0.2.11/WP2 runtime.

## Production trace and ownership

`POST /chat` in `agent/api_server.py` validates JSON, derives the actor/session/thread, builds one read-only unified-understanding preview, and calls `AgentRuntime.chat`. `AgentRuntime.chat` is a transport adapter: it supplies timing and runtime context, serializes the response, and records the display transcript. `Orchestrator.handle_message` owns conversation decisions. Its deterministic pending-state guards run before ordinary intent selection; the WP1 `RequestUnderstandingService` then owns ordinary single-capability selection. `CapabilityRegistry` validates the selected capability and inputs, its canonical invocation hook calls the native implementation, and its verifier decides whether the returned implementation result is structurally usable.

Mutation previews are currently produced by the native capability/controller path and represented as universal mutation plans. `ConfirmationStore`, `mutation_plan.py`, `confirmation_transactions.py`, and `ExecutorRegistry` bind confirmation to actor, thread/session, plan fingerprint, targets, expiry, and an exactly-once transaction. Those components remain the mutation authority. The task loop records and sequences their result; it does not mint confirmations, call raw executors, or turn an `approved` model field into authority.

The canonical SQLite database is `MemoryDB.db_path` (`agent.db`). It already owns durable chat, memory, audit, and continuity data. WP3 adds namespaced `agent_tasks`, `agent_task_steps`, `agent_task_events`, and `agent_task_approvals` tables through an additive migration. The pre-existing `tasks` table is a personal to-do feature and is not reused because its schema and semantics are unrelated. The JSON `MutationPlanStore` remains a compatibility cache for individual mutation previews; it is not a task-state store.

The Web UI posts ordinary messages to `/chat`, stores only session/thread identifiers in the browser, and renders confirmation cards from response data. WP3 adds task cards derived from the server task record and a small `/tasks` read/control API. Browser state is never authoritative.

`scripts/native_capability_proof.py` protects the WP2 inventory and `scripts/release_gate.py` runs its declared proofs. WP3 adds a task-proof manifest/report and runner; the canonical and extended gates execute it and fail closed on a missing category, stale candidate fingerprint, unproved task-composable capability, or policy mismatch.

Older `AssistantPlanner`, Plan Mode helpers, compatibility phrase classifiers, open loops, personal to-do tasks, setup flows, pack lifecycle flows, and lifecycle journals are not a second WP3 task loop. Where retained, they are bounded compatibility or low-level execution/approval adapters. They cannot select a WP3 action after a task plan has selected a registry capability.

## Authority boundaries

1. Unified understanding decides casual, direct single-capability, ambiguous, or candidate substantial-task handling. It remains authoritative for the fast path.
2. The task planner may propose only a versioned structured plan. Model output is untrusted data.
3. The plan validator resolves every executable step against the live `CapabilityRegistry`, validates its input contract, derives mode/approval/health from the registry, rejects output references that cannot be resolved, and enforces hard bounds.
4. The coordinator alone advances durable task state and dispatches the current validated step.
5. The registry is the only action authority. No task step names an endpoint, helper, executor, command line, or pack implementation.
6. Existing confirmation and executor components remain authoritative for actual mutations. A task approval is an additional exact binding to the current task/plan; it never weakens the native approval.
7. Registry and independent task verifiers, not prose or executor `ok`, govern completion.

## Versioned contracts

The public contract family is `personal-agent.task.v1`:

- `GoalV1`: bounded original goal, concise success criteria, actor/session/thread binding.
- `PlanV1`: task id, monotonically increasing version, canonical SHA-256 hash, ordered steps, creation source, ceilings, and explicit criteria.
- `StepV1`: stable step id, registered capability id, validated inputs, dependencies, allowed prior-output references, expected evidence, verifier kind, registry-derived mode/approval, retry class/limit, timeout, and optional declared compensation id.
- `EvidenceV1`: source capability/invocation, redacted result summary/hash, registry verifier result, optional independent observation, timestamp, and criterion links.
- `FailureV1`: validation/policy, missing information, dependency unavailable, missing capability, transient, deterministic, verification, cancelled, expired, or indeterminate classification with a bounded public explanation.
- `MissingCapabilityV1`: original outcome/criteria, missing user-goal ability, safe input/output shape, scopes/dependencies, considered capabilities, partial evidence, why completion is impossible, and a safe next-step category. It performs no pack search, fetch, import, install, enable, execution, or creation.
- `ProgressEventV1` and `OutcomeV1`: redacted durable progress and terminal/partial evidence.

Unknown fields are rejected at the plan boundary. Strings, collections, evidence, and stored events are size bounded and recursively redacted. Raw prompts, hidden reasoning, secrets, bearer/token values, private URLs, and unnecessary file contents are never persisted.

## Complexity and fast path

A greeting, casual fallback, clarification, task-control turn, or one selected capability remains on the existing deterministic path. A substantial-task candidate requires at least two independently understood registered capability goals, or an explicit outcome whose validated proposal contains multiple dependent steps. Risk alone does not make a one-capability request a task. Clause decomposition uses the same semantic request-understanding service; it is not a trigger list. A new model-planned version uses at most one full generation. Status, cancellation, approval, pause/resume, stored execution, and verification use no planning generation.

## State machine

States are `planning`, `proposed`, `ready`, `awaiting_information`, `awaiting_approval`, `running`, `verifying`, `paused`, `blocked`, `recovering`, `succeeded`, `partially_completed`, `failed`, `denied`, `cancelled`, `expired`, and `indeterminate`.

| From | Allowed next states |
| --- | --- |
| planning | proposed, awaiting_information, blocked, failed |
| proposed | ready, awaiting_information, awaiting_approval, cancelled, failed |
| ready | running, paused, cancelled, expired |
| running | awaiting_approval, verifying, paused, blocked, recovering, partially_completed, cancelled, failed, indeterminate |
| awaiting_information | proposed, cancelled, expired |
| awaiting_approval | running, denied, cancelled, expired, blocked |
| verifying | running, succeeded, partially_completed, recovering, blocked, failed, indeterminate |
| paused | ready, cancelled, expired, blocked |
| blocked | ready, partially_completed, cancelled, expired, failed |
| recovering | ready, running, awaiting_approval, blocked, partially_completed, failed, indeterminate |
| terminal states | no transitions |

Each transition is compare-and-swap guarded by task revision. Only one terminal event may be committed. One active mutation is allowed per actor/thread and per canonical resource key. Read-only tasks are bounded by the configured concurrent-task ceiling.

## Approval binding

The task approval digest covers schema version, task id, plan version/hash, exact ordered mutating step ids, canonical validated inputs and target/resource fingerprints, relevant preconditions, actor, session, thread, permission/mode snapshot, and expiry. It is single-use. A plan revision, changed target/input/order, health or Safe/Controlled Mode change, actor/thread/session mismatch, expiry, denial, cancellation, or replay invalidates it. Existing native preview/confirmation must also succeed; task approval is never substituted for it.

## Execution, evidence, and recovery

Read-only steps dispatch through `CapabilityRegistry.invoke`. Mutation steps first invoke the registered native preview boundary and stop at `awaiting_approval`; confirmed execution continues through the existing deterministic confirmation route. The coordinator records the canonical result, calls the registry verifier, and performs a declared independent observation when available. Overall success requires every criterion to cite passing evidence. Response wording is derived from state/evidence, so an unverified task cannot claim completion.

Retries are allowed only for registry-declared retry-safe read-only/transient work, with a small hard limit. A dispatched mutation whose result was not durably recorded becomes `indeterminate` on restart. It is reconciled against the confirmation transaction/idempotency ledger and an independent observation before any retry. Completed mutations are never replayed. Replanning creates a new version, retains valid evidence, invalidates approvals, and cannot add an unregistered capability. Compensation is available only when explicitly declared by the registry and separately verified; inverse commands are never synthesized.

Startup reconciliation leaves valid approval waits intact until expiry, returns resumable read-only work to `ready`, and marks in-flight mutations indeterminate. Terminal tasks never resume themselves. Retention bounds terminal task count, event count, and evidence bytes without deleting active records.

## API and UI

`GET /tasks` lists the caller's bounded recent tasks, `GET /tasks/{id}` returns one caller-bound task, and `POST /tasks/{id}/control` accepts only versioned `cancel`, `pause`, or `resume` controls with thread/session binding and compare-and-swap revision. Ordinary chat remains the primary creation/control surface. The Web UI renders a compact task card with goal, verified progress, approval preview, blocked/partial/final state, and Stop/Resume controls. Advanced details contain IDs and redacted evidence only.

## Security limits and exclusions

Hard ceilings cover steps, revisions, replans, retry attempts, capability calls, planning generations, wall time, input/result/event sizes, retained tasks, and concurrency. Untrusted model, file, web, pack, memory, tool, or UI content cannot introduce a capability, policy, approval, or evidence. The coordinator rejects raw shell/endpoint/URL proposals unless a selected registered capability contract explicitly accepts the field.

WP3 does not implement executable/declarative pack workers, remote pack acquisition, automatic pack discovery/install, assistant-created packs/tools, reference packs, arbitrary shell, or managed-service installation. The missing-capability record is only a safe handoff contract for later work packages.

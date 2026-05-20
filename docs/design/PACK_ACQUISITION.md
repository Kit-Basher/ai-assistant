# Pack Acquisition Coordinator

Status: current assistant-facing workflow.

`agent/packs/acquisition.py` is the assistant-facing coordinator for external skill acquisition after a user asks for a missing capability. It does not make external packs native abilities and does not bypass source trust, preview, quarantine, review, approval, enablement, configuration, permission, or managed-adapter gates.

The coordinator wraps the existing safety services:

- capability detection and recommendation identify the requested capability and search approved/trusted catalog sources only.
- registry discovery supplies catalog candidates and source trust status.
- scaffold preview/create supplies review-only generated text candidates when no trusted catalog candidate exists.
- `PackLifecycleService` reports current state, missing gate, and next safe step.
- `PackLifecycleActionController` advances exactly one confirmed gate at a time.
- `ManagedAdapterInvoker` invokes approved core-owned adapter operations only after lifecycle returns `usable=true`.

## Assistant Contract

For capability requests, the assistant must say what is missing and what the next safe step is. It must not dead-end at “I can sketch a helper” without a lifecycle action.

Allowed v1 outcomes:

- trusted catalog candidate found: show preview before fetch/import.
- remote source untrusted: stop at the source-trust gate and explain that approval is required before fetch/import.
- no candidate found: offer a preview-only scaffold path.
- imported for review: ask for review/approval.
- approved but disabled: ask to enable.
- enabled but missing config/permission: ask for that exact gate.
- usable: invoke only supported managed-adapter operations.

Each confirmation advances one gate only. Repeated `yes` must not skip approval, enablement, configuration, permission, or use gates.

## Safety Boundaries

Remote content remains hostile even when a source is approved. Source trust only permits preview/fetch into quarantine. Catalog metadata, archives, manifests, README, and `SKILL.md` content remain untrusted until normalized and reviewed.

External/generated packs must not run arbitrary code. They can request only approved managed adapters implemented in core runtime. V1 does not add internet-wide search, OAuth, browser scraping, transcript lookup, YouTube/browser parsing, dependency installs, `handler.py`, or arbitrary generated code execution.

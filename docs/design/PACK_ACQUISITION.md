# Pack Acquisition Coordinator

Status: current assistant-facing workflow.

`agent/packs/acquisition.py` is the assistant-facing coordinator for external skill acquisition after a user asks for a missing capability. It does not make external packs native abilities and does not bypass source trust, preview, quarantine, review, approval, enablement, configuration, permission, or managed-adapter gates.

Canonical package shape: [`docs/design/EXTERNAL_PACK_FORMAT.md`](/home/c/personal-agent/docs/design/EXTERNAL_PACK_FORMAT.md).

The coordinator wraps the existing safety services:

- capability detection and recommendation identify the requested capability and search approved/trusted catalog sources only.
- registry discovery supplies catalog candidates and source trust status.
- safe web search may supply untrusted source leads when approved/trusted catalogs have no candidate and search is available.
- scaffold preview/create supplies review-only generated text candidates when no trusted catalog candidate or source lead path is available.
- `PackLifecycleService` reports current state, missing gate, and next safe step.
- `PackLifecycleActionController` advances exactly one confirmed gate at a time.
- `ManagedAdapterInvoker` invokes approved core-owned adapter operations only after lifecycle returns `usable=true`.

## Assistant Contract

For capability requests, the assistant must say what is missing and what the next safe step is. It must not dead-end at “I can sketch a helper” without a lifecycle action.

Allowed v1 outcomes:

- trusted catalog candidate found: show preview before fetch/import.
- remote source untrusted: stop at the source-trust gate and explain that approval is required before fetch/import.
- safe web-search leads found: show untrusted source leads only. Leads are not source approval, are not trusted, and cannot be fetched/imported until the separate source approval gate is completed.
- source approval preview confirmed: record explicit trust for the source id only, then stop. Source approval does not approve pack content and does not fetch, download, import, install, approve, enable, configure, grant permissions, or use a pack.
- approved source fetch confirmed: fetch into quarantine and import for review only, then stop. The assistant must show the imported pack review state before any approval continuation. Quarantine fetch/import does not approve, enable, configure, grant permissions, or use a pack; review approval remains the next gate.
- no candidate found: offer a preview-only scaffold path.
- imported for review: ask for review/approval.
- approved but disabled: ask to enable.
- enabled but missing config/permission: ask for that exact gate.
- usable: invoke only supported managed-adapter operations.

Each confirmation advances one gate only. Repeated `yes` must not skip approval, enablement, configuration, permission, or use gates.

## Safety Boundaries

Remote content remains hostile even when a source is approved. Source trust only permits preview/fetch into quarantine. Catalog metadata, archives, manifests, README, and `SKILL.md` content remain untrusted until normalized and reviewed.

Safe web-search lead discovery is metadata-only. Result URLs, titles, snippets, and engine/source labels are untrusted search metadata. The assistant must not fetch result pages, download archives, call `/packs/install`, import packs, enable packs, or infer safety from GitHub or any other domain. Leads only point to the separate source approval gate.

Source approval is explicit user trust for a source id, not trust in the content. It may record a source catalog/policy entry that permits a future fetch or preview into quarantine, but fetched content remains hostile and must still pass catalog/source policy, quarantine, normalization, inspection, review approval, enablement, configuration, permissions, and managed-adapter gates before use.

Quarantine fetch/import-for-review is a separate gate after source approval. The assistant may fetch only from an approved source id through the existing hostile remote-fetch and `ExternalPackIngestor` paths. A successful fetch creates a review-only external pack candidate with no approval, no enablement, no configuration, no permission grant, and no use.

After any import-for-review result, the assistant must render a bounded review-state summary before asking for approval. That summary may include pack identity, lifecycle state, local review status, enabled=false, permissions/grants status, managed-adapter kinds, risk flags, import status, and safe source/provenance metadata. It must not expose raw `SKILL.md`, README, manifests, catalog listings, prompt text, secrets, private paths, or long hostile strings. Review state is not content trust; it is a truthfulness checkpoint that tells the user why the pack is still not usable and names the next safe gate as review/approval.

External/generated packs must not run arbitrary code. They can request only approved managed adapters implemented in core runtime. V1 does not add internet-wide search, OAuth, browser scraping, transcript lookup, YouTube/browser parsing, dependency installs, `handler.py`, or arbitrary generated code execution.

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

- trusted catalog candidate found: show untrusted metadata only; do not offer remote fetch/import.
- remote source found: explain that remote acquisition is unavailable. Source policy may allow metadata queries only.
- safe web-search leads found: show untrusted source leads only. Leads are not source approval, are not trusted, and cannot be fetched/imported until the separate source approval gate is completed.
- source approval preview: advisory only in the assistant compatibility flow. Source catalog creation and query-policy changes are separate centrally authorized operations.
- approved-source fetch confirmation: explicitly unavailable. Old approval/fetch confirmations cannot open a URL or import content.
- no candidate found: offer a preview-only scaffold path.
- imported for review: show review state, then show a review-approval preview before any approval mutation.
- approved but disabled: show an enablement preview before any enablement mutation.
- enabled but missing config/permission: ask for that exact gate.
- usable: ask for a specific managed-adapter operation, show an invocation preview, then invoke only after an explicit confirmation.

Each confirmation advances one gate only. Repeated `yes` must not skip approval, enablement, configuration, permission, or use gates.

## Safety Boundaries

Remote content remains hostile even when a source is allowlisted. Source policy
permits metadata queries only; it does not permit fetch into quarantine.
Catalog metadata, archives, manifests, README, and `SKILL.md` content remain
untrusted. A future fetch stage must be separately authorized and digest-bound.

Safe web-search lead discovery is metadata-only. Result URLs, titles, snippets, and engine/source labels are untrusted search metadata. The assistant must not fetch result pages, download archives, call `/packs/install`, import packs, enable packs, or infer safety from GitHub or any other domain. Leads only point to the separate source approval gate.

Source catalog and query policy are distinct centrally authorized state changes, not trust in pack content. They permit metadata queries only and do not authorize archive acquisition.

Quarantine fetch/import-for-review is not implemented at the product boundary. The retained hostile-fetch primitives are offline/test infrastructure and are not called by Web, assistant, Telegram, CLI, compatibility, or registered executor paths. A future implementation must add a separate authorized fetch stage before approval or installation.

After any import-for-review result, the assistant must render a bounded review-state summary before asking for approval. That summary may include pack identity, lifecycle state, local review status, enabled=false, permissions/grants status, managed-adapter kinds, risk flags, import status, and safe source/provenance metadata. It must not expose raw `SKILL.md`, README, manifests, catalog listings, prompt text, secrets, private paths, or long hostile strings. Review state is not content trust; it is a truthfulness checkpoint that tells the user why the pack is still not usable and names the next safe gate as review/approval.

Review approval continuation is also one gate only. After the review state, the next `yes` shows a review-approval preview and does not mutate approval. A second `yes` records review approval only. It does not enable the pack, configure it, grant permissions, execute code, invoke managed adapters, or use the pack. After approval, the assistant reports the current lifecycle state and names the next safe gate, normally enablement.

Enablement continuation follows the same preview-before-mutation rule. After review approval, the assistant may queue enablement only when `PackLifecycleService` says enablement is next. The first `yes` shows an enablement preview and does not mutate state. A second `yes` records enablement only. It does not configure adapters, grant permissions, invoke managed adapters, execute code, or use the pack. After enablement, the assistant reports the next lifecycle gate: configuration, permission, or ready-to-use state. Ready-to-use still does not mean automatic invocation.

Configuration and permission continuation is separate from enablement. If the enabled pack needs a managed-adapter permission, the assistant first previews the requirement: adapter kind, scope, allowed file types, whether a local path is needed, and the safety limits. A user-provided local path can then produce a scoped grant preview. Only the following confirmation records metadata/config. The grant does not invoke the adapter, use the pack, execute code, install dependencies, run shell commands, or read/parse private file contents. If the grant makes the pack usable, the assistant reports ready/usable and asks for the next specific input or action instead of running automatically.

Managed adapter invocation is a later explicit action after usability. The assistant must preview the pack, adapter kind, operation, redacted grant scope, read/write behavior, and disabled execution channels before running anything. The first supported operation set is `validate_grant`, `describe_capability`, and `dry_run`; `local_file_import` v1 can validate metadata and dry-run the selected-file grant only. Content read, search, parse, or indexing behavior is not implemented until a future core-owned adapter operation adds it.

External/generated packs must not run arbitrary code. They can request only approved managed adapters implemented in core runtime. V1 does not add internet-wide search, OAuth, browser scraping, transcript lookup, YouTube/browser parsing, dependency installs, `handler.py`, or arbitrary generated code execution.

## v2F authorization status

Local text-pack installation is centrally Plan/confirm authorized. Combined
remote fetch/install is explicitly unavailable: remote acquisition requires a
future separately authorized, connection-target-validated quarantine stage,
followed by a second digest- and normalized-manifest-bound approval/install
Plan. Source allowlisting alone never authorizes fetch, install, execution, or
permissions.

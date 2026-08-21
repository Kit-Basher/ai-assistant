# Safe Pack Acquisition and Useful Brokers (WP5)

Status: implementation design for the WP5 product path built above the WP1–WP4.5 authorities.

## Existing ownership and migration

`RequestUnderstandingService` remains the only ordinary-chat selector. `PackAcquisitionCoordinator` handles a structured missing-capability handoff, but before WP5 it can only query configured metadata and offer a text scaffold. `PackLifecycleService`, the universal mutation-plan/confirmation controller, `PackCapabilityStore`, `DynamicPackCapabilityRuntime`, `CapabilityRegistry`, and the WP3 task coordinator respectively remain the authorities for lifecycle truth, mutation authority, exact pack versions, action registration, invocation, and multi-step work.

WP4's `RemotePackFetcher` and `SourceFetchPreviewService` contain hostile-input test primitives, but the product controller intentionally denies fetch. WP5 promotes a hardened transport into the canonical acquisition controller; source policy remains metadata-query authority only. Legacy local-directory install and text-pack routes remain compatibility surfaces and cannot register or invoke an action outside the dynamic runtime.

The old `automatic_pack_action=false` missing-capability dead end is migrated to a proposal record with `automatic_discovery=true` and `automatic_fetch=false`. Automatic work is read-only metadata discovery only. Fetch, draft creation, review approval, grants, enablement, activation, update, rollback, disable, revocation, removal, and invocation remain distinct exact-confirmation transitions.

## Contract family

- `personal-agent.pack-acquisition.v1`: sanitized exact source, immutable provenance, fetch ceilings, actor/session/thread/plan binding, and quarantine outcome.
- `personal-agent.pack-review.v1`: exact archive, normalized content, capability, contract, broker and asset digests plus bounded static findings.
- `personal-agent.pack-broker.v1`: core-owned broker kind, exact pack/version/contract binding, scopes, limits, data-flow class and grant digest.
- `personal-agent.pack-draft.v1`: bounded assistant proposal using only supported pack classes, transforms, brokers and registered capabilities.
- `personal-agent.pack-update.v1`: old/new exact-version comparison, changed authority, activation preconditions and rollback eligibility.
- `personal-agent.pack-private-store.v1`: namespaced structured data schema, quotas, retention and compatibility version.
- `personal-agent.pack-visualizer.v1`: raster asset digest, decoded bounds, frame grid and mapping from the fixed core state vocabulary.
- `personal-agent.pack-wp5-proof.v1`: commit/diff-bound proof categories and sensitivity results.

Unknown authority-bearing fields fail validation. All contracts are canonically serialized before SHA-256 binding. Authority binds pack id, version, normalized content digest, capability contract digest, broker declaration digest, actor, session, thread, plan/revision, target/scope and expiry.

## Acquisition and quarantine

Supported sources are an exact GitHub repository/ref, a GitHub archive, a configured catalog entry resolving to one of those sources, and an explicitly supplied generic HTTPS zip/tar archive. A commit-pinned Git source retains that commit as provenance. For product GitHub traffic, a mutable ref is visibly unpinned in the preview and is resolved through the bounded core transport to a 40-character commit before the archive is fetched; the resolved commit, archive SHA-256, and normalized content digest become the immutable review identity. Generic archives use the archive and normalized-content digests as their immutable review identity.

`SafeHttpsTransport` owns certificate-verified GET/HEAD requests with no ambient proxy, cookies, credentials or caller headers. It validates normalized hostname, port, DNS answers and the connected peer at every redirect. Loopback, private, link-local, unspecified, carrier-grade NAT, multicast, reserved, ULA, IPv4-mapped local/private and metadata targets fail closed. Redirects, headers, body bytes, connect/read/total time and content types are bounded.

Bytes stream into a mode-0700 runtime-owned staging directory and are atomically finalized under quarantine. Archive inspection rejects traversal/absolute paths, links, special files, executable modes, hidden control files, nested archives, unsupported code/binaries, duplicate/case-fold/Unicode-normalized collisions, format/magic mismatch and count/size/depth/ratio/time exhaustion. Failure and cancellation remove partial staging data. Quarantine never implies review approval, grant, enablement, registration or invocation.

## Core-owned brokers

Pack code never receives host handles. Each broker validates the exact current pack/grant immediately before work and returns tainted bounded data for core verification.

1. `selected_local_data`: reads one exact confirmed regular file under an allowed root using descriptor-relative, no-follow checks and a stable stat fingerprint. Supported bounded input is UTF-8 text, JSON, CSV and HTML. It cannot scan a directory or read archives, executables, devices, databases or credential formats.
2. `pack_private_store`: transactional JSON records namespaced by pack/version and actor, with schema, item, byte, value, write-rate and retention ceilings. Cross-pack access fails. Raw input is not retained; the local-search workflow stores only bounded derived fields. Revocation blocks access immediately.
3. `scoped_https`: core-owned GET/HEAD only to reviewed origins, path templates and parameter names. It shares acquisition SSRF/redirect/proxy/timeout/byte enforcement. Responses are tainted. A plan may not combine private local data/private-store values with outbound network use in WP5. Authenticated requests remain unavailable.
4. `presence_visualizer`: core UI rendering of a structurally decoded, CRC-checked reviewed PNG plus declarative animation metadata. Allowed states are `idle`, `listening`, `thinking`, `acting`, `success`, `warning`, and `error`. WebP is intentionally not accepted until an equally bounded full decoder is part of the runtime. SVG/HTML/CSS/JS, URLs, fonts, audio, shaders and executable UI are rejected. Reduced motion selects a single stable frame.

Broker-before/guest/broker-after is mandatory: core obtains granted inputs, pure computation receives bounded values only, core validates output, and any later mutation requires its own native confirmation. WP4 Wasm remains no-WASI/no-import Bubblewrap+Wasmtime and receives no broker handles.

## Creation, lifecycle and updates

Assistant creation uses at most one model proposal per draft revision. The proposal is untrusted and validated against supported deterministic templates: portable reference, declarative registered-capability composition, local-data search, and sprite visualizer. It cannot name endpoints, shell, code, dependencies, dynamic capabilities or unknown brokers. Confirmation writes only a quarantine/review candidate. The assistant cannot approve, grant, enable, activate, invoke or publish it.

Usability requires exact normalized/artifact digests, review approval, enablement, required configuration, exact broker grants, healthy dependencies, self-test and verifier. Dynamic registration occurs only through `DynamicPackCapabilityRuntime` into the existing registry. WP3 may name only the resulting registry id and binds the exact version/content/contract/broker digests.

An update is a new quarantine/version record. A bounded diff covers source, archive/content/contract/broker/grant/schema/verifier/example/asset/risk changes. The old active version remains authoritative until atomic activation of a fully reviewed, granted, enabled and self-tested new version. Changed authority invalidates pending approvals and tasks. Rollback reactivates only a previously reviewed exact version whose current dependencies, grants and self-test still pass. Disable/revoke removes authority immediately; removal may retain or delete only that pack's bounded private data and audit tombstone.

## API, chat, UI and Telegram

The canonical pack status API exposes candidate classification, sanitized provenance, lifecycle checklist, access/data flow, health, self-test, verifier, last bounded invocation and one next safe action. Mutation endpoints are thin adapters to the central preview/confirmation controller. Chat uses unified understanding and the acquisition coordinator; it never adds a phrase router. Telegram uses the same actor/session/thread-bound plans or reports the mutation unavailable. The Web UI renders the same records, escaped and bounded, with candidate search, quarantine/review, grants, enablement, update diff, rollback/revoke/remove, and optional visualizer controls. Browser state is never authority.

The completion audit makes those claims executable end to end. Once
`packs.manage` is selected, structured lifecycle inputs resolve an exact live
pack/version; they do not reclassify ordinary intent. Inspection, update status,
and version comparison are read-only. Approval, grants, enable/disable,
activation, rollback, removal, and broker revocation produce exactly one
actor/session/thread-bound preview and one confirmation. A short same-thread
follow-up such as “is there an update?” retains the preceding pack context, but
an unrelated message neither advances nor cancels a pending mutation. Remote
fetch responses expose only a bounded dynamic-record identity so the next review
gate can be completed without leaking quarantine paths or hostile documents.

The normal-user Skills view queries only enabled, policy-allowed configured
metadata sources and labels cache age/staleness. It separately supports an exact
HTTPS/GitHub source, assistant draft preview, version diff, activation/rollback,
broker revocation, removal, and reduced-motion visualizer preview. No listing or
status render performs a remote fetch.

Exact-candidate proof uses an immutable minimal reference artifact at Git commit
`96e290cf0981b6986da4b5c7cc0b35b9a8db1949`. The reference commit contains only
one declarative manifest and is test evidence, not a bundled or live-installed
capability. The isolated workflow removes it and every generated pack/grant/index
before exit.

## Security, recovery and explicit exclusions

Foreign content cannot approve itself, select capabilities, expand grants, alter plans, supply verifier truth or claim completion. Broker results and remote content are tainted and never become prompts or authority automatically. In-flight mutations are reconcile-first; unknown outcomes do not retry. Revocation/update/restart causes exact binding revalidation before any queued task continues.

WP5 does not add arbitrary Python/JavaScript/native/shell/dependencies, directory crawling, broad writes, unrestricted HTTP/LAN/browser/OAuth, pack-visible secrets, Docker/Podman/systemd/package control, pack UI code, automatic lifecycle mutations, model changes, or self-modifying core code. SearXNG remains optional and is not installed or started.

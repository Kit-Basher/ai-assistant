# Safe Pack Capability Runtime (WP4)

Status: implemented release design. This document describes the authority and
containment boundaries for locally supplied external packs. It does not define
automatic acquisition or assistant-created packs; those remain WP5 work.

## Ownership before WP4

The existing `PackLifecycleService`, external ingestion/store, pack review UI,
permission controllers, and managed-adapter path own the portable text-pack
lifecycle and compatibility APIs. They do not make foreign content executable.
WP1 request understanding owns ordinary `/chat` selection, the WP2
`CapabilityRegistry` owns action authority and proof contracts, and the WP3
task coordinator owns plans, approvals, execution evidence, and completion.
WP4 extends those authorities; it does not add another router or executor.

Legacy pack recommendation, catalog, text import, and managed-adapter routes
remain compatibility/lifecycle surfaces. They cannot register an action and
cannot invoke a WP4 capability. Only `DynamicPackCapabilityRuntime` may derive
an external `CapabilityDefinition`, and every invocation then crosses the same
registry used by native chat and tasks.

## Versioned contracts

- `personal-agent.pack.v1`: normalized pack identity, class, version and exact
  content digest.
- `personal-agent.pack-capability.v1`: bounded semantics, JSON-shaped input and
  output schemas, mode, permissions, invocation, verifier, ceilings and a
  deterministic self-test input.
- `personal-agent.pack-worker.v1`: one signed integer input to one signed
  integer output from a named WebAssembly export. Worker output is untrusted.
- `personal-agent.pack-proof.v1`: commit/diff-bound release evidence.

Unknown authority-bearing fields fail validation. External IDs are always
`pack.<canonical-pack-id>.<capability-name>` and cannot shadow native IDs.
Manifests are limited to 64 KiB, eight capabilities, 24 properties per object,
five schema levels, 12 examples, and bounded strings. Wasm artifacts are
limited to 2 MiB. Normalized JSON is sorted and hashed before preview/review.

## Pack classes

Portable text packs have no capability contracts and are never executable or
globally inserted into prompts. Existing bounded, enabled text-pack retrieval
remains unchanged.

Declarative packs have one fixed invocation per capability. WP4 supports only
static invocation of an already registered native capability with literal
values or `$input.<field>` mappings. There are no loops, branches, expressions,
dynamic capability names, endpoints, URLs, shell, imports, or hidden prompts.
Mode, permissions and approval can only become stricter. Because WP4 adds no
general mutation broker, external mutating declarations report
`unsupported_effect_broker` and cannot register.

Sandboxed executable packs are pure computation only. They cannot call the
registry, task store, database, models, secrets, services, native executors, or
host brokers. The initial ABI deliberately supports only `i32 -> i32`; richer
host effects require a future core-owned broker and a new ABI.

## Executable isolation

Each invocation starts a new Bubblewrap process with user, PID, network, IPC,
and UTS namespaces, a cleared environment, no inherited application handles,
no host home/repository/state mounts, and only read-only runtime/code/module
mounts plus an invocation scratch tmpfs. Within it, Wasmtime runs with no WASI
and rejects every module import. Therefore filesystem, environment, sockets,
DNS, clocks, processes, shell, dynamic libraries and dependency installation
are structurally absent from guest authority.

Wasmtime fuel is at most 10,000,000; guest linear memory at most 32 MiB; wall
time at most five seconds; request/response at most 64 KiB; module size at most
2 MiB. The host child has file-descriptor, CPU, output-file and core-dump
limits. Timeout/cancellation kills the process group. Scratch is deleted and
the worker PID is checked after every invocation. Missing Bubblewrap, missing
Wasmtime, failed namespace setup, malformed protocol, imported functions,
fuel exhaustion, timeout or a changed artifact fails closed.

The API process never imports or evaluates pack code. A sanitized subprocess
alone is not treated as containment; both namespace and Wasmtime layers must be
healthy.

## Lifecycle and dynamic registration

Local import first creates a mutation preview bound to exact source digest,
actor, session, thread and a five-minute expiry. Review, exact permission
grants, enable/disable, block/revoke and removal are separate one-use mutation
plans. Replays, wrong bindings and content/state changes fail without advancing
a gate.

Registration requires intact normalized/artifact digests, explicit review,
exact enablement, all derived permissions, healthy isolation when applicable,
a valid verifier, and a passing real self-test. Startup reconstructs only those
entries. A corrupt entry is isolated from native and other external entries.

Changing content produces a new record and immediately clears approval,
enablement, grants and self-test evidence from older versions. Disable,
revocation, block, removal, corruption or health failure removes the capability
from the next atomic registry reconstruction. Closures bind invocation and
health to exact record, content, contract and ABI digests, so a stored WP3 step
cannot silently substitute a newer version.

## Chat, tasks, verification and truth

Enabled external definitions contribute only bounded descriptions/examples to
the existing offline semantic matcher. Explicit pack provenance helps resolve
native/pack ambiguity; names and payload paths are not authority. The selected
ID is schema validated and invoked by the registry without phrase
reclassification.

WP3 validates the external definition exactly like a native step. The result
contains an audit-safe pack/version/content/contract/ABI binding. Foreign output
cannot add steps or emit authoritative approval, verifier, evidence or task
completion fields. Core output-schema validation, registry verification and
WP3 task-level evidence decide success.

The capability and pack status views distinguish imported, reviewed, enabled,
permissioned, healthy and usable. They expose no artifact path, raw documents,
module bytes, worker stdout/stderr, prompts or secrets.

## Release enforcement and security proof

`scripts/pack_capability_proof.py` checks required contracts and proof
categories, runs the production `/chat`/task/worker suite, records the exact
commit plus tracked-diff fingerprint, and exercises release-model sensitivity.
The canonical release gate runs it alongside retained WP1–WP3 proof. Actual
worker tests reject WASI imports and exhaust an infinite loop under fuel while
checking orphan cleanup. The isolated candidate corpus provides lifecycle and
live API/UI evidence; reference packs are never installed in live user state.

## Explicit exclusions

WP4 performs no remote search/download, automatic install/review/enable/grant,
background update, assistant-created code, self-modification, arbitrary HTTP,
browser/OAuth access, shell, Docker/Podman/systemctl control, YouTube access, or
sprite rendering. Missing-capability handoff continues to set
`automatic_pack_action=false`.

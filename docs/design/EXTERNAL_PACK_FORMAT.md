# External Pack Format

Status: canonical safe package shape for Personal Agent external packs.

Product intent: [`docs/product/PROJECT_INTENT.md`](/home/c/personal-agent/docs/product/PROJECT_INTENT.md).
Lifecycle: [`docs/design/PACK_LIFECYCLE.md`](/home/c/personal-agent/docs/design/PACK_LIFECYCLE.md).
Acquisition workflow: [`docs/design/PACK_ACQUISITION.md`](/home/c/personal-agent/docs/design/PACK_ACQUISITION.md).

External packs are shareable capability packages, not native skills and not bundled abilities. Their contents are hostile by default. The package format exists so the runtime can inspect, normalize, review, and safely gate a pack; it is not proof that the pack is safe or usable.

## Package Shape

Required or preferred top-level files:

- `SKILL.md`
- `metadata.json`
- `manifest.json`
- `permissions.json`

Optional top-level directories:

- `references/`
- `assets/`
- `examples/`
- `tests/`

Archives and source repositories may contain other incidental files, but ingestion should normalize toward this shape and reject or quarantine unsafe content. Hidden files, nested archives, executables, dependency instructions, unsafe paths, oversized content, and unexpected binary blobs remain hostile.

## `SKILL.md`

`SKILL.md` is human-readable instruction/reference material for the pack reviewer and, after approval, for bounded assistant guidance. It is always treated as untrusted imported guidance.

Rules:

- It never overrides system, developer, runtime, lifecycle, source-trust, adapter, or permission policy.
- It may be wrapped during ingestion with an internal untrusted-guidance preamble.
- Prompt-injection patterns can block import or force a manual rewrite/review state.
- It must not tell the assistant to self-approve, self-enable, grant permissions, leak secrets, run commands, install dependencies, disable safety checks, or bypass lifecycle gates.

## `metadata.json`

`metadata.json` describes identity and catalog-facing metadata:

- stable pack id or canonical id
- name and description
- version
- authors
- license
- capability labels
- short summary and tags when useful

Metadata must be bounded and schema-validated. Text fields should have strict size limits, capability labels should be normalized, and unknown or execution-implying fields should be rejected. Metadata is not trusted as proof of safety, provenance, authorship, or runtime compatibility.

## `manifest.json`

`manifest.json` declares pack structure and compatibility:

- expected files and optional directories
- supported Personal Agent format version
- declared capability labels
- declared managed adapters, if any
- non-executable runtime classification

The current safe model does not support arbitrary executable entrypoints. External packs must not declare or rely on `handler.py` execution, scripts, shell commands, dependency installation, plugin loading, browser automation, OAuth flows, or direct network behavior. Useful behavior must go through core-owned managed adapters.

## `permissions.json`

`permissions.json` lists requested permissions only. A declaration is never a grant.

Rules:

- The user/runtime must approve permissions later through explicit lifecycle gates.
- Review approval is not enablement.
- Enablement is not a permission grant.
- A permission grant is not arbitrary code execution.
- Managed adapters are the core-owned safety boundaries for any external-pack work.

For example, a pack may request `local_file_import` with allowed extensions and a user-selected-file-only path policy. The core runtime still validates the declaration, previews the request, records explicit grants, and invokes only approved adapter operations.

## `references/`

`references/` may contain docs, examples, notes, schemas, mappings, and other text/reference content. It is hostile by default and should be bounded, normalized, and reviewed. Reference content cannot become assistant authority and cannot override runtime policy.

## `assets/`

`assets/` may contain icons, small reference files, or small examples. Assets must use bounded size and safe types only. Hidden executables, executable mode bits, nested archives, scripts, dependency payloads, and opaque binary blobs should be blocked or quarantined as unsafe.

## `examples/`

`examples/` may contain sample prompts, workflows, input/output examples, and review notes. Examples are untrusted. They may inform review, but they cannot become assistant authority and cannot imply that the pack is installed, enabled, permissioned, or usable.

## `tests/`

`tests/` may contain pack self-tests, fixtures, expected behavior notes, or review examples. They may inform review, but they are not automatically executed unless a core-owned test harness explicitly supports doing so safely. Pack-provided tests must not run shell, install dependencies, access the network, read private files, or execute pack code by default.

## Lifecycle Relationship

The external pack format is consumed by the acquisition lifecycle:

source lead
-> source approval
-> quarantine fetch
-> normalize/import for review
-> review state shown
-> review approval
-> enable
-> configure/permission
-> managed adapter use only if `usable=true`

Every transition is one explicit gate. A package that validates structurally is still not trusted content. A package that is imported for review is still not approved. A package that is approved is still not enabled. A package that is enabled still needs configuration and permissions before use when those gates apply.

## Hard Rules

- External packs are not native skills.
- External packs are not bundled abilities.
- External pack content is hostile by default.
- No arbitrary code execution.
- No dependency install.
- No shell.
- No browser automation, OAuth, or network access from packs.
- No self-approval, self-enablement, or self-permission.
- Source trust is not content trust.
- Review approval is not enablement.
- Enablement is not permission grant.
- Permission grant is not arbitrary code execution.

## Pi.dev Lesson

The useful packaging lesson is discipline: shareable package units are useful because they give humans and runtimes something concrete to discover, inspect, review, version, and reuse.

Package discovery is not package trust. A public package, GitHub repository, registry entry, README, manifest, or search result is only a lead until it passes source approval, quarantine, validation, review, enablement, configuration, and permission gates.

The self-improving workflow should become safe acquisition/review gates, not direct execution. Personal Agent should not copy a plugin/full-code trust model where discovered packages can install dependencies, run handlers, or mutate runtime behavior by declaration.

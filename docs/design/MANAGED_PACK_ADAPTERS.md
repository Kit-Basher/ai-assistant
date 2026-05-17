# Managed Pack Adapters

Status: current implemented direction for external/generated pack safety.

Product intent: [`docs/product/PROJECT_INTENT.md`](/home/c/personal-agent/docs/product/PROJECT_INTENT.md).

External and generated skill packs do not run arbitrary code. Generated packs remain text-only review artifacts unless the core runtime implements and gates a managed adapter.

## External Pack Lifecycle Contract

External packs are not bundled active abilities. Starter catalogs are discoverable sources only; they are not installed capabilities and must not be described as built-in skills.

Missing capability flow must not dead-end. The assistant should tell the user what is missing and the next safe step: preview a discovered pack, preview a scaffold, create a review-only candidate, or explain why the capability is blocked.

The intended lifecycle is discover -> preview -> scaffold/create -> quarantine -> inspect -> approve -> configure -> permission -> enable -> use. A pack is usable only after the relevant approval, enablement, configuration, and permission gates are complete.

## Phase 3 Contract

The first enabled adapter is `local_file_import`.

Packs may declare:

```json
{
  "kind": "local_file_import",
  "purpose": "Import a user-selected Google Takeout YouTube watch-history file.",
  "allowed_extensions": [".json", ".html"],
  "max_file_size_mb": 50,
  "path_policy": "user_selected_file_only",
  "stores_local_index": true,
  "network_allowed": false
}
```

The runtime rejects unknown or disabled adapter kinds, network access for `local_file_import`, wildcard or executable extensions, directory-scanning policies, dependency installs, and executable generated pack files.

## Permission Flow

When a generated pack needs local-file access, chat asks the user for a local path. A provided path first renders a permission preview: what data would be accessed, what would be stored, what is blocked, and that raw content is not logged or added to support context.

In this phase, confirming the preview records grant metadata only. The assistant validates path metadata after confirmation: exists, is a file, extension is allowed, and size is within the declared limit. It does not parse, index, upload, or execute anything.

## Future Adapters

Placeholders such as `local_directory_import`, `google_takeout_youtube_history`, `transcript_lookup`, and `network_fetch` are disabled until they have explicit core-runtime implementations, review, and tests.

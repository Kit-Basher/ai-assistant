# Known Limits

This project is intentionally capable, but it is not a general-purpose agent
runtime. These limits are deliberate.

## Not Supported

- arbitrary shell execution
- unrestricted filesystem mutation
- in-process foreign code, arbitrary Python/JavaScript/shell/native plugins, or
  host-effectful executable packs (WP4 permits only no-import pure Wasm in its
  isolated worker)
- silent install/enable/switch behavior
- automatic adoption of discovery proposals
- legacy root/system install scripts as the supported shipping path
- background full-disk indexing

## Still Limited by Design

- continuity memory does not do merge-on-write
- cross-key atomic memory snapshots are not supported
- discovery metadata is advisory, not authoritative
- pack discovery can be degraded or unavailable without making the core
  runtime invalid
- supported generated drafts start with a preview, then may create a portable,
  declarative local-data-search, native-wrapper, or PNG-visualizer review
  candidate in quarantine; generated packs cannot approve, enable, permission,
  execute, or publish themselves
- private-history capabilities, including YouTube history search, are not live
  capabilities yet; the current scaffold starts from a future user-selected
  local Google Takeout import and explicitly defers OAuth, browser scraping,
  transcript fetching, network lookup, and video/audio downloads
- useful host effects remain core-owned brokers: one exact selected file can be
  indexed into bounded per-pack structured storage, public HTTPS is exact
  GET/HEAD only, and private local data cannot be combined with outbound network
- WP4 executable pack ABI v1 is intentionally `i32 -> i32` pure computation;
  filesystem, network, model, service, database and secret brokers are absent
- authenticated pack HTTP, WebP visualizers, OAuth/browser automation, broad
  filesystem access, directory crawling, and pack-supplied UI/code remain unsupported
- release and recovery diagnostics are deterministic, but they are not a full
  observability stack

## Environment Assumptions

- supported install paths are the stable release bundle, the optional Debian
  package, and the checkout/dev install for repo work
- canonical mutable state lives under `~/.local/share/personal-agent`
- canonical operator config lives under `~/.config/personal-agent`
- the runtime is expected to be started and managed by the user service

## What Operators Should Expect

- `system is initializing` means wait and retry
- `system is blocked` means a real blocker must be fixed
- `pack installed` does not mean `pack usable`
- `discovery unavailable` does not mean the assistant is broken
- stale confirmation tokens should be retried from a fresh preview
- `/ready`, `/state`, and `/packs/state` are the normal diagnosis surface

## Support Boundary

If a situation cannot be diagnosed from the state surfaces plus `python -m
agent doctor`, that is a product gap, not an expected operator workflow.

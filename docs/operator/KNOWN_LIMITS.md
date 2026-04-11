# Known Limits

This project is intentionally capable, but it is not a general-purpose agent
runtime. These limits are deliberate.

## Not Supported

- arbitrary shell execution
- unrestricted filesystem mutation
- foreign code or plugin pack execution
- silent install/enable/switch behavior
- automatic adoption of discovery proposals
- Debian/system packaging as the supported shipping path
- background full-disk indexing

## Still Limited by Design

- continuity memory does not do merge-on-write
- cross-key atomic memory snapshots are not supported
- discovery metadata is advisory, not authoritative
- pack discovery can be degraded or unavailable without making the core
  runtime invalid
- release and recovery diagnostics are deterministic, but they are not a full
  observability stack

## Environment Assumptions

- supported install path is the repo checkout plus user-level systemd service
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

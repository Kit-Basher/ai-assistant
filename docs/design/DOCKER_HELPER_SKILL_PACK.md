# Docker Helper Skill Pack Plan

The Docker helper skill pack is a possible future external skill pack for Docker education, review, and troubleshooting. It is intentionally separate from native Docker or managed local service control.

## Purpose

A Docker helper pack can help users understand and work with Docker concepts. It may provide guidance for:

- explaining containers, images, volumes, and networks
- troubleshooting common Docker errors
- drafting Dockerfiles or Compose files as text
- reviewing Dockerfiles or Compose files for risk
- explaining Personal-Agent-managed service behavior
- inspecting safe Docker metadata through native controllers if those controllers exist

## Boundary

The Docker helper skill pack is not the Docker executor.

It cannot run arbitrary Docker commands, grant itself Docker permissions, approve images, approve containers, approve services, or bypass managed local service confirmation gates. It can request native managed-service actions, but those actions remain controlled by core runtime policy, approved service definitions, and explicit user confirmations.

This means the pack may explain or review Docker material, but native code owns any actual Docker/Podman interaction.

## Possible Pack Format

A Docker helper pack should follow the external pack format:

- `SKILL.md`
- `metadata.json`
- `permissions.json`
- `examples/`
- `references/`
- `tests/`

All pack content remains untrusted imported guidance. It cannot override runtime policy, self-approve, self-enable, grant permissions, or execute code.

## Possible Capabilities

A future Docker helper pack might declare capability labels such as:

- `docker_explain`
- `dockerfile_review`
- `compose_review`
- `container_troubleshooting`
- `managed_service_guidance`

These labels describe guidance or review capabilities. They do not imply execution authority.

## Safe Future Native Interfaces

If the core runtime later implements Docker-related managed interfaces, the pack may request them through normal pack lifecycle and permission gates. Possible native interfaces include:

- `docker_status_readonly`
- `docker_container_list_readonly`
- `docker_logs_readonly` for Personal-Agent-managed containers only
- `managed_service_setup` for approved services only

Each interface must be core-owned, schema-validated, previewed, permissioned, audited, and bounded to the approved operation.

## Explicitly Blocked

The Docker helper pack must not provide or trigger:

- arbitrary `docker run`
- arbitrary `docker exec`
- arbitrary `docker compose up`
- privileged containers
- host filesystem mounts
- host networking
- arbitrary image pulls
- Dockerfile builds from untrusted content
- Compose execution from untrusted content
- using Docker as an escape hatch for pack code execution
- granting itself Docker permissions
- approving itself, enabling itself, or granting itself managed-service permissions

## Review Expectations

Review should treat Docker guidance as high-impact operational advice. The assistant should distinguish text help from execution. If a user asks to run Docker commands, the assistant should route through native managed local service controls when the action is an approved service action, or explain that arbitrary Docker execution is not available.

A future Docker helper pack should be tested with hostile instructions that ask it to bypass safety policy, run arbitrary commands, mount host paths, use host networking, or execute untrusted Compose files. Those requests must remain blocked by core runtime policy.

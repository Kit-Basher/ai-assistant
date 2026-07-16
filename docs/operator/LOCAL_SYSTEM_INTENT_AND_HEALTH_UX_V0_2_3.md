# Local System Intent and Health UX v0.2.3

## Reported Failure

The request:

```text
can you take a look at my pc and tell me what is using the most memory?
```

was incorrectly routed to web search. A follow-up that explicitly asked for a
local check succeeded, but the response was a broad telemetry dump and did not
answer which local program was using the most memory.

## Routing Policy

Current-device grounding now beats broad web-search fallback. High-confidence
local signals include:

- `my PC`, `my computer`, `my machine`, `this machine`, `this system`;
- `locally`, `on this computer`;
- process/resource wording such as memory, RAM, CPU, GPU, VRAM, disk, process,
  program, application, service, running, or slow.

General instruction questions still remain informational or web-routed when
appropriate, for example “How do I check memory usage in Windows?” or “Search
for Linux system monitors.”

## Process Inspection

The system-health collector now includes bounded process-level information from
`/proc`:

- PID;
- process name;
- friendly display name;
- RSS memory;
- memory percent;
- sampled CPU percent where available;
- user scope;
- deterministic process group counts.

Command lines and environment variables are omitted by default so tokens,
secret arguments, and private paths are not exposed in ordinary answers.

## Grouping and Names

Related processes are grouped by deterministic names where safe. Examples:

- Ollama -> `Ollama`, local AI model service;
- Firefox/Firefox ESR -> `Firefox`;
- Chromium/Chrome -> `Chromium`;
- GNOME Shell -> desktop environment;
- Docker daemon/runtime -> Docker services.

Unknown processes fall back to their process name with a generic local-process
description.

## Response Shape

When the user asks what is using memory, the response starts with RAM pressure
and top memory users. Disk warnings, network state, and unrelated telemetry are
secondary.

Friendly output is the default. Technical details such as PID and raw
percentages remain available in structured data and can be exposed by explicit
detail-oriented surfaces.

## RAM and VRAM

System RAM and GPU VRAM are distinct. A memory question primarily answers
system RAM, but high GPU VRAM usage is mentioned separately when visible.

## Boundaries

Local inspection remains read-only. It may list processes, CPU/memory totals,
disk usage, GPU status, service status, network status, uptime, and similar
telemetry. It does not kill processes, restart services, clear caches, delete
files, change priority, stop containers, or mutate startup behavior. Those
actions require the normal Universal Plan and confirmation boundary.

## Proof

- `tests/fixtures/local_system_intent_cases.json`
- `tests/test_local_system_intent_and_health.py`
- `scripts/local_intent_routing_smoke.py`
- `scripts/local_system_inspection_smoke.py`

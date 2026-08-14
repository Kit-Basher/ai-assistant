# Model Runtime Truth and Latency Closure (WP4.5)

## Authority and production flow

The structured provider observation is the authority for what is physically installed. For Ollama this is the bounded JSON `GET /api/tags` adapter, not decorative CLI output. The canonical LLM registry remains the authority for configured/routable identities; default/temporary/effective selection remains owned by defaults and the existing exact-confirmation controller. Model Manager records are lifecycle history. Model Scout is advisory. Neither history nor a remote catalog row can become installed truth.

`RuntimeTruthService.model_runtime_truth()` reconciles these sources into `personal-agent.model-runtime-truth.v1`. `/models`, `/llm/models/truth`, conversational `models.inventory`, Model Scout, and the Web UI consume that view. Installed, registered-not-observed, remote catalog, and history-only rows are separate collections. Case, omitted `:latest`, provider prefixes, and equal digests are aliases of one authority identity. A timeout yields `unknown` and retains the aged prior observation; it never means absent.

Evaluation evidence uses `personal-agent.installed-model-eval.v1` and the production OpenAI-compatible Ollama adapter. Models run sequentially with identical prompts, deterministic judges, one bounded metrics probe, no registry/default mutation, no download, and no deletion. The recommendation is advisory and is attached to the running build commit at read time. A live default change remains a `models.switch` mutation with exact preview, actor/session/thread binding, confirmation, verification, and rollback.

## Latency root cause and closure

Before WP4.5, every ordinary `/chat` request called `assistant_frontdoor_active()`, which synchronously called `ready_status()`. On cache expiry that path performed provider/model readiness probes before unified routing. Presence therefore spent roughly five seconds before the handler began its 2–4 ms deterministic work; system status paid that preflight plus another readiness aggregation and approached ten seconds. HTTP framing already included a correct `Content-Length`; the delay was before first byte, not connection-close body delimiting.

Front-door selection now checks configured architecture only and never probes dependencies. Deterministic runtime status consumes the latest observed readiness snapshot with its age/staleness. Explicit readiness and model refresh surfaces retain bounded probing authority. Presence/status never launch a model or pack worker. The release latency investigation treats loopback presence p95 over 250 ms or warm status p95 over 1,000 ms as blockers instead of printing zero blockers unconditionally.

## Security and exclusions

Provider metadata is size/count/string bounded. Evidence exposes no credentials, private provider URLs, prompts, unique hardware serials, or model files. Benchmark outputs are bounded previews and hashes. Scout cannot switch; evaluation cannot switch; this work does not install/remove models or change remote fallback. WP5 pack acquisition and assistant-created packs are excluded.

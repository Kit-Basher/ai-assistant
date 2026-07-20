# Local Model Provider Support

Date: 2026-06-11

This is an operator boundary note based on the current code, not a roadmap
promise.

## Current Status

| Backend | Status | What is supported now |
|---|---|---|
| Ollama | supported | Built-in local provider entry, `/api/tags` discovery, OpenAI-compatible chat path, allowlisted pull endpoint, and GGUF import into Ollama through existing model-manager flows. Users may install Ollama themselves; Personal Agent does not require Ollama globally. |
| Generic OpenAI-compatible local server | supported | Operators can add an `openai_compat` provider with a local `base_url`, `chat_path`, optional API key source, and model rows. Catalog probing can use `/v1/models` for non-Ollama providers. |
| llama.cpp direct binary/library | absent | There is no core adapter that starts `llama-server`, loads GGUF directly, manages llama.cpp processes, or imports GGUF without Ollama. Do not claim direct llama.cpp support. |
| llama.cpp OpenAI-compatible server | supported through OpenAI-compatible provider path | The registry accepts `llama_cpp_openai_compatible`, maps it to the llama.cpp backend/probe label, and uses the OpenAI-compatible chat/probe path. The user/operator must run and configure the server endpoint. |
| LM Studio | supported through OpenAI-compatible provider path | No LM Studio-specific adapter exists. If LM Studio exposes an OpenAI-compatible local endpoint, configure it as an `openai_compat` local provider and add/probe model rows. |
| vLLM | supported through OpenAI-compatible provider path | No vLLM-specific lifecycle adapter exists. If vLLM exposes an OpenAI-compatible endpoint, configure it as an `openai_compat` provider and add/probe model rows. |
| GGUF discovery/import | partially supported | Model scout and Hugging Face planning can identify GGUF-ready repos/files. Existing import paths create an Ollama Modelfile and run `ollama create`, or import from a user-provided Modelfile. Direct GGUF execution outside Ollama is absent. |

## Model Scout Expectation

On a Debian desktop with an RTX 2060 6GB VRAM and 64GB RAM, model scout must not
present huge local models as easy/default. Guidance should distinguish:

- easy local path: Ollama with small/quantized models that fit the machine;
- advanced local path: GGUF plus a user-managed runner such as llama.cpp server,
  LM Studio, or vLLM through an OpenAI-compatible endpoint;
- Ollama GGUF import path: existing Personal Agent-managed import into Ollama,
  with explicit confirmation and persistent journal rows;
- cloud/API path: OpenAI/OpenRouter or another configured API provider when the
  local machine is not suitable.

Large MoE or high-parameter models may be discoverable, but they should be
framed as advanced/manual and hardware-sensitive, not as a default local setup
for a 6GB VRAM desktop.

## Safety Boundary

Current model acquisition/import rollback removes only owned generated files,
such as Personal Agent-created Modelfiles or download markers. It does not
delete unrelated Ollama models, Ollama cache data, user-provided GGUF files,
user-provided Modelfiles, or broad filesystem directories.

Public acquisition, pull, switch, default, registry, cleanup, and maintenance
writes require a Universal Mutation Plan and durable scoped confirmation. SAFE
MODE denies acquisition/removal/remote-switch equivalents while inventory,
health, and advisory discovery remain available.

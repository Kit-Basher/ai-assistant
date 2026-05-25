# Capability Setup UX

Personal Agent should present setup in terms of user capabilities, not backend names. Normal users ask for web search, Telegram, local chat, better local models, or extra skills. The assistant translates those requests into safe setup flows.

## Required And Optional Components

Required components are the minimum needed for the assistant to answer in the current runtime, such as a usable chat model or configured provider.

Optional components add capabilities. Docker, Podman, SearXNG, llama.cpp, model runners, Telegram, and external skill packs are optional backends. They must not be presented as required unless the user asks for a capability that depends on them.

Examples:

- Web search: optional capability. Technical dependency is a local SearXNG service, often managed through Docker or Podman.
- Better local models or MoE models: optional capability. Technical dependency may be llama.cpp or a compatible local model runner in a future setup lane.
- Local chat model: baseline capability. Technical dependency is Ollama or another configured provider.
- Telegram: optional capability. Technical dependency is a bot token and service.
- Extra skills: optional capability. Technical dependency is the external pack acquisition lifecycle.

Docker and llama.cpp are optional. Do not make either one a global prerequisite for using the assistant.

## Normal-User Wording

Lead with the capability and next step:

- "Web search needs one extra local component. I can set it up for you using Docker, if Docker is available. It will run only on this computer. Say yes to continue."
- "Web search needs Docker or Podman. I can show the install command, but I won’t install system software automatically."
- "This model needs an extra local model runner. I can show the setup plan before changing anything."
- "Telegram needs a bot token. Paste one when you are ready."
- "I can look for a skill, preview it, and walk through review before it becomes usable."

Avoid leading with backend details such as image names, binds, volumes, ports, env vars, registry IDs, lifecycle internals, or provider implementation names. Show those details only when the user asks, or in Advanced/operator surfaces and structured payloads.

## Assistant-Guided Setup

The assistant owns the user-facing flow:

1. Detect the missing capability.
2. Explain the missing piece in plain language.
3. Show a safe preview.
4. Ask for confirmation.
5. Run only the approved bounded action.
6. Report what changed and what still needs setup.

If setup cannot proceed, explain the plain reason and one next step. For example, if approved web-search ports are busy, say that another local app is using them and do not attempt setup.

## Preview, Confirm, Action

All mutating setup follows preview -> confirm -> action. A confirmation applies only to the previewed action and must not silently continue into later gates.

Examples:

- Web search service setup: preview the local service plan, confirm, then run only the approved bounded service action.
- External skills: preview, source approval, quarantine import, review, enable, configure/permission, then use.
- Model runner setup: future preview and confirmation before any install, download, import, or config change.

## No Silent System Installs

The assistant must not silently install Docker, Podman, llama.cpp, Ollama, system packages, or dependencies. If a system dependency is missing, offer guidance or a command for the user to run, unless a future core-owned setup path has its own explicit preview and confirmation gate.

## Technical Details

Normal surfaces should say "Web search", "Chat", "Telegram", "Local models", and "Skills". Advanced/operator surfaces may show SearXNG, Docker/Podman, Ollama, llama.cpp, ports, binds, volumes, env vars, source IDs, and pack lifecycle details.

Technical details remain in structured payloads for audit and diagnostics, but they should not dominate the first visible response.

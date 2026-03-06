# Telegram Extraction Checklist

This inventory tracks Telegram adapter responsibilities and the target ownership boundary.

| function/path | current responsibility | keep in adapter? | target runtime helper/module |
|---|---|---|---|
| `telegram_adapter/bot.py::build_app/register_handlers/main` | PTB app lifecycle, polling startup, handler registration | yes | n/a (transport-only) |
| `telegram_adapter/bot.py::acquire_telegram_poll_lock/release_telegram_poll_lock` | singleton poller lock enforcement | yes | n/a (transport-only) |
| `telegram_adapter/bot.py::_send_reply` | Telegram-safe send (truncate, BadRequest retry, parse fallback), `telegram.out` logging | yes | n/a (transport-only) |
| `telegram_adapter/bot.py::_handle_message` (input extraction + tracing + audit wrappers) | inbound update extraction, trace/audit wrapper, bridge dispatch, transport send | yes | `agent/telegram_bridge.py` for business routing |
| `telegram_adapter/bot.py::_handle_status/_handle_doctor/_handle_help/_handle_health/_handle_brief/_handle_memory` | slash-command transport wrappers | yes | `agent/telegram_bridge.py::handle_telegram_command` |
| `telegram_adapter/bot.py::classify_model_provider_intent + setup wizard state machine` | provider/model setup wizard transport flow | partial | keep temporarily in adapter; converge toward runtime-owned setup contract |
| `telegram_adapter/bot.py::maybe_handle_llm_fixit_reply` | active fix-it reply interception + state transition persistence | partial | `agent/ux/llm_fixit_wizard.py` (source of truth), adapter as thin mapper |
| `agent/telegram_bridge.py::classify_telegram_text_command` | canonical NL->command mapping for help/setup/status/health/doctor/brief/memory | no | `agent/telegram_bridge.py` (runtime bridge) |
| `agent/telegram_bridge.py::handle_telegram_text` | canonical text dispatch to runtime command handlers | no | `agent/telegram_bridge.py` |
| `agent/telegram_bridge.py::handle_telegram_command` | canonical command handling and deterministic result envelope | no | `agent/telegram_bridge.py` |
| `agent/telegram_bridge.py::build_telegram_help` | canonical help rendering with onboarding-aware behavior | no | `agent/setup_wizard.py`, `agent/onboarding_contract.py` |
| `agent/telegram_bridge.py::build_telegram_setup` | canonical setup summary rendering/error path | no | `agent/setup_wizard.py`, `agent/recovery_contract.py` |
| `agent/telegram_bridge.py::build_telegram_status` | runtime status rendering from `/ready` + `/llm/status` truth | no | `agent/runtime_contract.py` |
| `agent/telegram_bridge.py::build_telegram_memory` | continuity summary via orchestrator/runtime | no | `agent/memory_runtime.py`, `agent/memory_contract.py` |
| `agent/telegram_bridge.py::build_telegram_error` | canonical deterministic error block formatting | no | `agent/error_response_ux.py` |

## Immediate extraction result
- Canonical help/setup/status/health/doctor/brief/memory wording and routing now go through `agent/telegram_bridge.py`.
- Adapter keeps transport mechanics and wizard interception while delegating canonical product UX generation to the bridge.

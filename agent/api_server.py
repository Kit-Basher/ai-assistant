from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import os
from pathlib import Path
from typing import Any

from agent.config import Config, load_config
from agent.llm.router import LLMRouter
from agent.secret_store import SecretStore


@dataclass
class RuntimeSettings:
    routing_mode: str


class SettingsStore:
    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def load(self, default_mode: str) -> RuntimeSettings:
        if not self._path.is_file():
            return RuntimeSettings(routing_mode=default_mode)
        try:
            parsed = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                return RuntimeSettings(routing_mode=default_mode)
            mode = str(parsed.get("routing_mode") or default_mode).strip().lower() or default_mode
            if mode not in {"auto", "prefer_cheap", "prefer_best"}:
                mode = default_mode
            return RuntimeSettings(routing_mode=mode)
        except Exception:
            return RuntimeSettings(routing_mode=default_mode)

    def save(self, settings: RuntimeSettings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"routing_mode": settings.routing_mode}
        self._path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


class AgentRuntime:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.router = LLMRouter(config, log_path=config.log_path)
        self.secret_store = SecretStore(path=os.getenv("AGENT_SECRET_STORE_PATH", "").strip() or None)
        settings_path = os.getenv("AGENT_UI_CONFIG_PATH", "").strip() or str(
            Path(config.db_path).resolve().parent / "ui_config.json"
        )
        self.settings_store = SettingsStore(settings_path)
        self.settings = self.settings_store.load(default_mode=self.router.policy.mode)
        self.router.set_routing_mode(self.settings.routing_mode)

        self._request_log: deque[dict[str, Any]] = deque(maxlen=100)

        # Load previously saved provider keys into the live runtime.
        for provider_name in ("openai", "openrouter"):
            key = self.secret_store.get_provider_api_key(provider_name)
            if key:
                self.router.set_provider_api_key(provider_name, key)

    def _log_request(self, endpoint: str, ok: bool, payload: dict[str, Any]) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "ok": bool(ok),
            "payload": payload,
        }
        self._request_log.appendleft(record)

    def health(self) -> dict[str, Any]:
        snapshot = self.router.doctor_snapshot()
        return {
            "ok": True,
            "service": "personal-agent-api",
            "time": datetime.now(timezone.utc).isoformat(),
            "routing_mode": snapshot.get("routing_mode"),
            "configured_providers": [item.get("name") for item in snapshot.get("providers") or []],
        }

    def models(self) -> dict[str, Any]:
        snapshot = self.router.doctor_snapshot()
        return {
            "providers": snapshot.get("providers") or [],
            "models": snapshot.get("models") or [],
            "routing_mode": snapshot.get("routing_mode"),
            "circuits": snapshot.get("circuits") or {},
        }

    def get_config(self) -> dict[str, Any]:
        return {
            "routing_mode": self.router.policy.mode,
            "retry_attempts": self.router.policy.retry_attempts,
            "timeout_seconds": self.router.policy.default_timeout_seconds,
            "secret_storage": self.secret_store.backend_name,
        }

    def update_config(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        mode = str((payload or {}).get("routing_mode") or "").strip().lower()
        if mode not in {"auto", "prefer_cheap", "prefer_best"}:
            return False, {"ok": False, "error": "routing_mode must be auto, prefer_cheap, or prefer_best"}
        self.router.set_routing_mode(mode)
        self.settings = RuntimeSettings(routing_mode=mode)
        self.settings_store.save(self.settings)
        return True, {"ok": True, "routing_mode": mode}

    @staticmethod
    def _normalize_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
        raw = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(raw, list):
            return []
        messages: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip() or "user"
            content = str(item.get("content") or "")
            messages.append({"role": role, "content": content})
        return messages

    def chat(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        messages = self._normalize_messages(payload)
        if not messages:
            return False, {"ok": False, "error": "messages must be a non-empty list"}

        model_override = (payload.get("model") or "").strip() or None
        provider_override = (payload.get("provider") or "").strip().lower() or None

        result = self.router.chat(
            messages,
            purpose=str(payload.get("purpose") or "chat"),
            provider_override=provider_override,
            model_override=model_override,
            require_tools=bool(payload.get("require_tools")),
            require_json=bool(payload.get("require_json")),
            require_vision=bool(payload.get("require_vision")),
            timeout_seconds=float(payload.get("timeout_seconds") or 0) or None,
        )

        response = {
            "ok": bool(result.get("ok")),
            "assistant": {
                "role": "assistant",
                "content": result.get("text") or "",
            },
            "meta": {
                "provider": result.get("provider"),
                "model": result.get("model"),
                "fallback_used": bool(result.get("fallback_used")),
                "attempts": result.get("attempts") or [],
                "duration_ms": int(result.get("duration_ms") or 0),
                "error": result.get("error_class"),
            },
        }
        self._log_request("/chat", bool(result.get("ok")), response["meta"])
        return bool(result.get("ok")), response

    def _model_for_provider(self, provider: str, preferred_model: str | None = None) -> str | None:
        if preferred_model:
            return preferred_model
        for model in self.router.registry.sorted_models():
            if model.provider == provider and model.enabled:
                return model.id
        return None

    def test_provider(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        provider = str((payload or {}).get("provider") or "").strip().lower()
        if not provider:
            return False, {"ok": False, "error": "provider is required"}

        model_override = self._model_for_provider(provider, str(payload.get("model") or "").strip() or None)
        if not model_override:
            return False, {"ok": False, "error": "No enabled model found for provider"}

        previous_key = self.secret_store.get_provider_api_key(provider) or ""
        candidate_key = str(payload.get("api_key") or "").strip()

        if candidate_key:
            if not self.router.set_provider_api_key(provider, candidate_key):
                return False, {"ok": False, "error": "Unknown provider"}

        result = self.router.chat(
            [
                {"role": "system", "content": "Reply with PONG."},
                {"role": "user", "content": "ping"},
            ],
            purpose="diagnostics",
            provider_override=provider,
            model_override=model_override,
            timeout_seconds=float(payload.get("timeout_seconds") or 5.0),
        )

        if result.get("ok"):
            if candidate_key:
                self.secret_store.set_provider_api_key(provider, candidate_key)
            response = {
                "ok": True,
                "provider": provider,
                "model": result.get("model"),
                "duration_ms": int(result.get("duration_ms") or 0),
            }
            self._log_request("/providers/test", True, response)
            return True, response

        if candidate_key:
            self.router.set_provider_api_key(provider, previous_key)
        response = {
            "ok": False,
            "provider": provider,
            "model": result.get("model"),
            "error": result.get("error_class") or "provider_error",
            "message": result.get("error") or "connection test failed",
        }
        self._log_request("/providers/test", False, response)
        return False, response


class APIServerHandler(BaseHTTPRequestHandler):
    runtime: AgentRuntime

    def log_message(self, format: str, *args) -> None:  # pragma: no cover - avoid noisy stdout in tests
        _ = format
        _ = args

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
        return {}

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, self.runtime.health())
            return
        if self.path == "/models":
            self._send_json(200, self.runtime.models())
            return
        if self.path == "/config":
            self._send_json(200, self.runtime.get_config())
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json()
        if self.path == "/chat":
            ok, body = self.runtime.chat(payload)
            self._send_json(200 if ok else 400, body)
            return
        if self.path == "/providers/test":
            ok, body = self.runtime.test_provider(payload)
            self._send_json(200 if ok else 400, body)
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_PUT(self) -> None:  # noqa: N802
        if self.path != "/config":
            self._send_json(404, {"ok": False, "error": "not_found"})
            return
        payload = self._read_json()
        ok, body = self.runtime.update_config(payload)
        self._send_json(200 if ok else 400, body)


def build_runtime(config: Config | None = None) -> AgentRuntime:
    loaded = config or load_config(require_telegram_token=False)
    return AgentRuntime(loaded)


def run_server(host: str, port: int) -> None:
    runtime = build_runtime()

    class _Handler(APIServerHandler):
        pass

    _Handler.runtime = runtime

    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"Personal Agent API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Personal Agent HTTP API")
    parser.add_argument("--host", default=os.getenv("AGENT_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENT_API_PORT", "8765")))
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

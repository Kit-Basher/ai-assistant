#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BASE_URL = os.environ.get("AGENT_API_BASE_URL") or "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("PROVIDER_MATRIX_SMOKE_TIMEOUT_SECONDS", "45"))
DEFAULT_CHAT_PROMPT = "Write one short sentence about a bicycle."
DEFAULT_TEST_TIMEOUT_SECONDS = float(os.environ.get("PROVIDER_MATRIX_TEST_TIMEOUT_SECONDS", "15"))
TARGET_PROVIDER_IDS = ("ollama", "openrouter")


@dataclass(frozen=True)
class ProviderTarget:
    provider_id: str
    model_id: str
    source: str
    health_status: str
    configured: bool
    active: bool
    required: bool


def _first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _normalized(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _json_from_response(body: str) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload_dict(document: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(document, dict) and isinstance(document.get("payload"), dict):
        return dict(document.get("payload"))
    return dict(document) if isinstance(document, dict) else {}


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
            ok = status < 400
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = int(getattr(exc, "code", 500))
        ok = False
    except urllib.error.URLError as exc:
        return {"ok": False, "status": 0, "payload": {}, "raw": "", "error": f"transport error: {exc.reason}"}
    except Exception as exc:  # pragma: no cover - defensive live-smoke guard
        return {"ok": False, "status": 0, "payload": {}, "raw": "", "error": f"transport error: {exc}"}
    payload_json = _json_from_response(raw)
    return {"ok": ok, "status": status, "payload": payload_json, "raw": raw}


def _request_payload(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None, *, timeout: float) -> dict[str, Any]:
    result = _request_json(base_url, method, path, payload, timeout=timeout)
    response_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    if not isinstance(response_payload, dict):
        response_payload = {}
    return {
        "ok": bool(result.get("ok")),
        "status": int(result.get("status") or 0),
        "payload": dict(response_payload),
        "raw": str(result.get("raw") or ""),
        "error": str(result.get("error") or "").strip() or None,
    }


def _provider_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _payload_dict(document)
    rows = payload.get("providers")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _status_models(document: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _payload_dict(document)
    rows = payload.get("models")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _defaults_snapshot(document: dict[str, Any]) -> dict[str, Any]:
    payload = _payload_dict(document)
    return {
        "default_provider": str(payload.get("default_provider") or "").strip() or None,
        "default_model": str(payload.get("default_model") or payload.get("chat_model") or "").strip() or None,
        "routing_mode": str(payload.get("routing_mode") or "").strip() or None,
        "allow_remote_fallback": bool(payload.get("allow_remote_fallback", True)),
    }


def _provider_health_status(provider_row: dict[str, Any]) -> str:
    health = provider_row.get("health") if isinstance(provider_row.get("health"), dict) else {}
    return str(health.get("status") or provider_row.get("health_status") or "unknown").strip().lower() or "unknown"


def _row_model_id(model_row: dict[str, Any]) -> str:
    return str(model_row.get("model_id") or model_row.get("id") or "").strip()


def _provider_row(rows: list[dict[str, Any]], provider_id: str) -> dict[str, Any]:
    provider_key = str(provider_id or "").strip().lower()
    for row in rows:
        if str(row.get("id") or row.get("provider") or "").strip().lower() == provider_key:
            return dict(row)
    return {}


def _provider_model_rows(provider_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = provider_row.get("models")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _pick_usable_model_id(provider_row: dict[str, Any]) -> str | None:
    model_rows = _provider_model_rows(provider_row)
    usable_rows = [row for row in model_rows if bool(row.get("usable_now", False))]
    if not usable_rows:
        return None
    current_model = str(provider_row.get("current_model_id") or provider_row.get("model_id") or "").strip()
    if current_model:
        for row in usable_rows:
            if _row_model_id(row) == current_model:
                return current_model
    return _row_model_id(usable_rows[0]) or None


def _provider_target_gate(provider_row: dict[str, Any], *, required: bool) -> tuple[str, str | None, str | None]:
    provider_id = str(provider_row.get("id") or provider_row.get("provider") or "").strip().lower()
    if not provider_id:
        return ("fail" if required else "skip", None, "provider id is missing")
    enabled = bool(provider_row.get("enabled", True))
    configured = bool(provider_row.get("configured", False))
    local = bool(provider_row.get("local", False))
    health_status = _provider_health_status(provider_row)
    health = provider_row.get("health") if isinstance(provider_row.get("health"), dict) else {}
    connection_state = str(
        provider_row.get("connection_state")
        or health.get("connection_state")
        or ""
    ).strip().lower() or "unknown"
    selection_state = str(
        provider_row.get("selection_state")
        or health.get("selection_state")
        or connection_state
    ).strip().lower() or connection_state
    auth_required = bool(provider_row.get("auth_required", False))
    secret_present = bool(provider_row.get("secret_present", False))
    if required and not local:
        return "fail", None, f"{provider_id} is not marked local"
    if not required and local:
        return "skip", None, f"{provider_id} is a local provider, not cloud"
    if not enabled or not configured:
        status = "fail" if required else "skip"
        return status, None, f"{provider_id} is not configured and enabled"
    if health_status != "ok":
        status = "fail" if required else "skip"
        return status, None, f"{provider_id} health is {health_status}"
    if connection_state != "configured_and_usable" or selection_state != "configured_and_usable":
        status = "fail" if required else "skip"
        return status, None, f"{provider_id} state is {connection_state}/{selection_state}"
    if auth_required and not secret_present:
        return "skip", None, f"{provider_id} is missing credentials"
    model_id = _pick_usable_model_id(provider_row)
    if not model_id:
        status = "fail" if required else "skip"
        return status, None, f"{provider_id} has no usable_now model"
    source = "provider_row"
    return "ok", model_id, source


def _build_provider_target(provider_row: dict[str, Any], *, required: bool) -> tuple[str, ProviderTarget | None, str]:
    provider_id = str(provider_row.get("id") or provider_row.get("provider") or "").strip().lower()
    if not provider_id:
        return ("fail" if required else "skip", None, "provider id is missing")
    status, model_id, reason = _provider_target_gate(provider_row, required=required)
    if status != "ok" or not model_id:
        return status, None, reason or "provider is not usable"
    target = ProviderTarget(
        provider_id=provider_id,
        model_id=model_id,
        source="provider_row",
        health_status=_provider_health_status(provider_row),
        configured=bool(provider_row.get("configured", False)),
        active=bool(provider_row.get("active", False)),
        required=required,
    )
    return "ok", target, ""


def _chat_probe(
    base_url: str,
    prompt: str,
    *,
    user_id: str,
    thread_id: str,
    timeout: float,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": str(prompt or "")}],
        "purpose": "chat",
        "task_type": "chat",
        "source_surface": "operator_smoke",
        "user_id": user_id,
        "thread_id": thread_id,
        "trace_id": f"provider-matrix-smoke-{user_id}-{int(time.time())}",
    }
    if provider:
        payload["provider"] = str(provider).strip().lower()
    if model:
        payload["model"] = str(model).strip()
    result = _request_json(base_url, "POST", "/chat", payload, timeout=timeout)
    response_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    assistant = response_payload.get("assistant") if isinstance(response_payload.get("assistant"), dict) else {}
    meta = response_payload.get("meta") if isinstance(response_payload.get("meta"), dict) else {}
    text = str(assistant.get("content") or response_payload.get("message") or response_payload.get("error") or "").strip()
    return {
        "ok": bool(result.get("ok")),
        "status": int(result.get("status") or 0),
        "text": text,
        "first_line": _first_line(text),
        "route": str(meta.get("route") or "").strip().lower() or "unknown",
        "error": str(result.get("error") or "").strip() or None,
        "payload": dict(response_payload),
    }


def _dead_end_warning(text: str) -> str | None:
    lowered = _normalized(text)
    if not lowered:
        return "empty response"
    if text.lstrip().startswith("{") or text.lstrip().startswith("["):
        return "raw json response"
    if any(token in lowered for token in ("need more context", "does not exist", "i can't", "i cannot", "couldn't complete")):
        return "dead-end wording"
    return None


def _chat_contract_check(chat_result: dict[str, Any], target: ProviderTarget) -> tuple[bool, str]:
    if not bool(chat_result.get("ok")):
        return False, f"chat probe failed: {chat_result}"
    if int(chat_result.get("status") or 0) >= 400:
        return False, f"chat probe returned HTTP {chat_result.get('status')}"
    text = str(chat_result.get("text") or "")
    if warning := _dead_end_warning(text):
        return False, f"chat probe warning: {warning} ({chat_result.get('first_line')})"
    payload = chat_result.get("payload") if isinstance(chat_result.get("payload"), dict) else {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    response_text = str(assistant.get("content") or payload.get("message") or payload.get("error") or "").strip()
    if not response_text:
        return False, "chat probe returned an empty assistant response"
    if str(meta.get("used_llm") if "used_llm" in meta else payload.get("used_llm")).strip().lower() not in {"true", "1"}:
        return False, f"chat probe did not use the LLM: {meta}"
    provider = str(meta.get("provider") or payload.get("provider") or "").strip().lower()
    model = str(meta.get("model") or payload.get("model") or "").strip()
    if provider != target.provider_id:
        if not target.required:
            return True, f"provider={provider or 'unknown'} model={model or 'unknown'} chat={chat_result.get('first_line')} (override not selected)"
        return False, f"chat provider mismatch: {provider} != {target.provider_id}"
    if target.required and not model:
        return False, f"chat model missing for required provider {target.provider_id}"
    timing = meta.get("chat_timing_ms") if isinstance(meta.get("chat_timing_ms"), dict) else {}
    llm_request_ms = int(timing.get("llm_request_ms") or 0)
    if llm_request_ms <= 0:
        return False, f"chat timing missing llm_request_ms: {timing}"
    route = str(meta.get("route") or payload.get("route") or "").strip().lower()
    if route in {
        "status",
        "runtime_status",
        "model_status",
        "setup",
        "memory",
        "social",
        "files",
        "filesystem",
        "pack",
        "packs",
        "discovery",
        "skill",
        "assistant_capabilities",
    }:
        return False, f"unexpected special-purpose route: {route}"
    return True, _first_line(response_text)


def _run_target(base_url: str, target: ProviderTarget, defaults: dict[str, Any], *, timeout: float) -> tuple[bool, str]:
    status_after = _request_payload(base_url, "GET", "/llm/status", timeout=timeout)
    status_payload = _payload_dict(status_after)
    runtime_mode = str(status_payload.get("runtime_mode") or "").strip().upper()
    if runtime_mode != "READY":
        return False, f"runtime not ready after switch: {status_payload}"
    active_provider_health = status_payload.get("active_provider_health") if isinstance(status_payload.get("active_provider_health"), dict) else {}
    active_model_health = status_payload.get("active_model_health") if isinstance(status_payload.get("active_model_health"), dict) else {}
    if str(active_provider_health.get("status") or "").strip().lower() != "ok":
        return False, f"active provider health is not ok: {active_provider_health}"
    if str(active_model_health.get("status") or "").strip().lower() != "ok":
        return False, f"active model health is not ok: {active_model_health}"
    visible_counts = status_payload.get("visible_counts") if isinstance(status_payload.get("visible_counts"), dict) else {}
    if int(visible_counts.get("total") or 0) <= 0:
        return False, f"no visible models after switch: {visible_counts}"
    if not bool(status_payload.get("compat_only", True)):
        return False, f"compat_only not set on operator surface: {status_payload}"
    if not bool(status_payload.get("non_canonical_for_assistant", True)):
        return False, f"assistant contract flag missing: {status_payload}"

    chat_result = _chat_probe(
        base_url,
        DEFAULT_CHAT_PROMPT,
        user_id=f"provider-matrix-{target.provider_id}",
        thread_id=f"provider-matrix-{target.provider_id}:thread",
        timeout=timeout,
        provider=target.provider_id,
        model=target.model_id,
    )
    chat_ok, chat_detail = _chat_contract_check(chat_result, target)
    if not chat_ok:
        return False, chat_detail
    return True, f"provider={target.provider_id} model={target.model_id} chat={chat_detail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a live provider matrix smoke against configured local and cloud provider backends."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the live API server.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Per-provider timeout in seconds.")
    args = parser.parse_args(argv)

    providers_response = _request_payload(str(args.base_url), "GET", "/providers", timeout=float(args.timeout))
    defaults_response = _request_payload(str(args.base_url), "GET", "/defaults", timeout=float(args.timeout))
    status_response = _request_payload(str(args.base_url), "GET", "/llm/status", timeout=float(args.timeout))

    original_defaults = _defaults_snapshot(defaults_response)
    initial_status = _payload_dict(status_response)
    if not initial_status:
        print(f"/llm/status preflight failed: {status_response}", flush=True)
        return 1
    provider_rows = _provider_rows(providers_response)
    provider_lookup = {str(row.get("id") or row.get("provider") or "").strip().lower(): dict(row) for row in provider_rows}
    local_row = provider_lookup.get("ollama") or {}
    cloud_row = provider_lookup.get("openrouter") or {}

    exit_code = 0
    for provider_id, provider_row, required in (
        ("ollama", local_row, True),
        ("openrouter", cloud_row, False),
    ):
        if not provider_row:
            if required:
                print(f"[{provider_id}] missing from /providers; this is required for the local assistant path.", flush=True)
                exit_code = 1
            else:
                print(f"[{provider_id}] skipped: not configured.", flush=True)
            continue
        provider_row["models"] = _provider_model_rows(provider_row)
        gate_status, target, reason = _build_provider_target(provider_row, required=required)
        if gate_status == "skip":
            print(f"[{provider_id}] skipped: {reason}", flush=True)
            continue
        if gate_status != "ok" or target is None:
            print(f"[{provider_id}] failed preflight: {reason}", flush=True)
            exit_code = 1
            continue
        ok = False
        detail = ""
        ok, detail = _run_target(str(args.base_url), target, original_defaults, timeout=float(args.timeout))
        print(
            f"[{target.provider_id}] health={target.health_status} configured={target.configured} active={target.active} "
            f"required={target.required} model={target.model_id} result={'ok' if ok else 'failed'}"
        )
        if detail:
            print(f"[{target.provider_id}] detail: {detail}", flush=True)
        if not ok:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

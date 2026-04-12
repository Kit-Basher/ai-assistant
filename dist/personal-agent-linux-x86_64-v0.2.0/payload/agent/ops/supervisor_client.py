from __future__ import annotations

import datetime as dt
import hmac
import json
import os
import socket
import uuid
from hashlib import sha256
from typing import Any


DEFAULT_SOCKET_PATH = "/run/personal-agent/supervisor.sock"
DEFAULT_LOG_LINES_MAX = 200


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_ops_config() -> dict[str, Any]:
    env_path = os.getenv("OPS_CONFIG_PATH")
    config_path = env_path or os.path.join(_repo_root(), "ops", "ops_config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError:
        return {}


def _utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sign_request(data: dict[str, Any], key: str) -> str:
    payload = _canonical_payload(data)
    digest = hmac.new(key.encode("utf-8"), payload.encode("utf-8"), sha256)
    return digest.hexdigest()


def build_request(op: str, payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    base = {
        "op": op,
        "ts": _utc_iso(),
        "nonce": uuid.uuid4().hex,
        "payload": payload or {},
    }
    base["sig"] = _sign_request(
        {
            "op": base["op"],
            "ts": base["ts"],
            "nonce": base["nonce"],
            "payload": base["payload"],
        },
        key,
    )
    return base


def _socket_path() -> str:
    env_path = os.getenv("SUPERVISOR_SOCKET_PATH")
    if env_path:
        return env_path
    config = load_ops_config()
    return config.get("socket_path") or DEFAULT_SOCKET_PATH


def _log_lines_max() -> int:
    env_val = os.getenv("SUPERVISOR_LOG_LINES_MAX")
    if env_val:
        try:
            return max(1, int(env_val))
        except ValueError:
            return DEFAULT_LOG_LINES_MAX
    config = load_ops_config()
    value = config.get("log_lines_max")
    if isinstance(value, int) and value > 0:
        return value
    return DEFAULT_LOG_LINES_MAX


def _hmac_key() -> str | None:
    return os.getenv("SUPERVISOR_HMAC_KEY")


def send_request(op: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    key = _hmac_key()
    if not key:
        return {"ok": False, "op": op, "error": "missing_hmac_key"}
    request = build_request(op, payload, key)
    path = _socket_path()
    response: dict[str, Any]
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(2.0)
        client.connect(path)
        client.sendall((json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8"))
        data = b""
        while b"\n" not in data and len(data) < 64 * 1024:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk
        line = data.split(b"\n", 1)[0].decode("utf-8") if data else "{}"
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            response = {"ok": False, "op": op, "error": "invalid_response"}
    return response


def clamp_log_lines(value: int | None) -> int:
    max_lines = _log_lines_max()
    if value is None:
        return min(50, max_lines)
    try:
        lines = int(value)
    except (TypeError, ValueError):
        return min(50, max_lines)
    return max(1, min(max_lines, lines))

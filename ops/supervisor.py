from __future__ import annotations

import argparse
import datetime as dt
import hmac
import json
import os
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from hashlib import sha256
from typing import Any


MAX_REQUEST_BYTES = 16 * 1024
DEFAULT_SKEW_SECONDS = 120
DEFAULT_NONCE_CACHE = 1024
DEFAULT_LOG_LINES_MAX = 200
DEFAULT_SOCKET_PATH = "/run/personal-agent/supervisor.sock"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_ts(value: str) -> dt.datetime | None:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _canonical_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sign_request(data: dict[str, Any], key: str) -> str:
    payload = _canonical_payload(data)
    digest = hmac.new(key.encode("utf-8"), payload.encode("utf-8"), sha256)
    return digest.hexdigest()


def verify_signature(data: dict[str, Any], key: str, sig: str) -> bool:
    expected = sign_request(data, key)
    return hmac.compare_digest(expected, sig or "")


class NonceCache:
    def __init__(self, max_size: int = DEFAULT_NONCE_CACHE) -> None:
        self._max_size = max(1, int(max_size))
        self._seen: OrderedDict[str, float] = OrderedDict()

    def check_and_store(self, nonce: str) -> bool:
        if not nonce:
            return False
        if nonce in self._seen:
            return False
        self._seen[nonce] = time.time()
        while len(self._seen) > self._max_size:
            self._seen.popitem(last=False)
        return True


class Supervisor:
    def __init__(
        self,
        *,
        socket_path: str,
        hmac_key: str,
        agent_unit_name: str,
        systemctl_mode: str = "system",
        log_lines_max: int = DEFAULT_LOG_LINES_MAX,
        skew_seconds: int = DEFAULT_SKEW_SECONDS,
        nonce_cache_size: int = DEFAULT_NONCE_CACHE,
    ) -> None:
        self.socket_path = socket_path
        self.hmac_key = hmac_key
        self.agent_unit_name = agent_unit_name
        self.systemctl_mode = systemctl_mode
        self.log_lines_max = max(1, int(log_lines_max))
        self.skew_seconds = max(1, int(skew_seconds))
        self.nonces = NonceCache(nonce_cache_size)

    def _systemctl(self, args: list[str]) -> list[str]:
        cmd = ["systemctl"]
        if self.systemctl_mode == "user":
            cmd.append("--user")
        cmd.extend(args)
        return cmd

    def _run(self, args: list[str]) -> dict[str, Any]:
        result = subprocess.run(args, check=False, capture_output=True, text=True)
        return {
            "exit_code": result.returncode,
            "stdout": (result.stdout or "").strip(),
            "stderr": (result.stderr or "").strip(),
        }

    def _validate_unit(self) -> bool:
        if not self.agent_unit_name:
            return False
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.@-"
        return all(ch in allowed for ch in self.agent_unit_name)

    def _verify_request(self, request: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
        op = request.get("op")
        ts = request.get("ts")
        nonce = request.get("nonce")
        payload = request.get("payload") or {}
        sig = request.get("sig")

        if not isinstance(op, str) or not isinstance(payload, dict):
            return False, "invalid_request", {}
        if not isinstance(ts, str) or not isinstance(nonce, str):
            return False, "invalid_request", {}
        if not isinstance(sig, str):
            return False, "missing_signature", {}

        ts_parsed = _parse_ts(ts)
        if not ts_parsed:
            return False, "invalid_timestamp", {}
        if abs((_utc_now() - ts_parsed).total_seconds()) > self.skew_seconds:
            return False, "timestamp_out_of_range", {}

        signed = {"op": op, "ts": ts, "nonce": nonce, "payload": payload}
        if not verify_signature(signed, self.hmac_key, sig):
            return False, "invalid_signature", {}
        if not self.nonces.check_and_store(nonce):
            return False, "replayed_nonce", {}
        return True, None, signed

    def _handle_restart(self) -> dict[str, Any]:
        if not self._validate_unit():
            return {"ok": False, "error": "invalid_unit_name"}
        result = self._run(self._systemctl(["restart", self.agent_unit_name]))
        status = self._run(self._systemctl(["is-active", self.agent_unit_name]))
        return {
            "ok": True,
            "result": {
                "restart": result,
                "status": status.get("stdout"),
            },
        }

    def _handle_status(self) -> dict[str, Any]:
        if not self._validate_unit():
            return {"ok": False, "error": "invalid_unit_name"}
        active = self._run(self._systemctl(["is-active", self.agent_unit_name]))
        show = self._run(
            self._systemctl(
                [
                    "show",
                    self.agent_unit_name,
                    "-p",
                    "ActiveEnterTimestamp",
                    "-p",
                    "SubState",
                    "-p",
                    "ActiveState",
                ]
            )
        )
        return {
            "ok": True,
            "result": {
                "active": active.get("stdout"),
                "show": show.get("stdout"),
            },
        }

    def _handle_logs(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._validate_unit():
            return {"ok": False, "error": "invalid_unit_name"}
        lines = payload.get("lines", 50)
        try:
            lines_int = int(lines)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid_lines"}
        lines_int = max(1, min(self.log_lines_max, lines_int))
        result = self._run(
            [
                "journalctl",
                "-u",
                self.agent_unit_name,
                "-n",
                str(lines_int),
                "--no-pager",
                "--output=short-iso",
            ]
        )
        return {"ok": True, "result": {"lines": result.get("stdout", "")}}

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        ok, error, signed = self._verify_request(request)
        if not ok:
            return {"ok": False, "op": request.get("op"), "error": error}
        op = signed["op"]
        payload = signed.get("payload") or {}
        if op == "restart":
            response = self._handle_restart()
        elif op == "status":
            response = self._handle_status()
        elif op == "logs":
            response = self._handle_logs(payload)
        else:
            return {"ok": False, "op": op, "error": "unknown_operation"}
        response.setdefault("op", op)
        return response

    def serve(self) -> None:
        os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen(5)
        try:
            while True:
                conn, _ = server.accept()
                with conn:
                    data = b""
                    while b"\n" not in data and len(data) < MAX_REQUEST_BYTES:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    if not data:
                        continue
                    line = data.split(b"\n", 1)[0]
                    try:
                        request = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        response = {"ok": False, "op": None, "error": "invalid_json"}
                    else:
                        response = self.handle_request(request)
                    conn.sendall((json.dumps(response, ensure_ascii=True) + "\n").encode("utf-8"))
        finally:
            server.close()
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)


def _read_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def main() -> int:
    parser = argparse.ArgumentParser(description="Personal Agent Ops Supervisor")
    parser.add_argument("--socket-path", default=_read_env("SUPERVISOR_SOCKET_PATH", DEFAULT_SOCKET_PATH))
    parser.add_argument("--agent-unit", default=_read_env("AGENT_UNIT_NAME"))
    parser.add_argument("--systemctl-mode", default=_read_env("SUPERVISOR_SYSTEMCTL_MODE", "system"))
    parser.add_argument("--log-lines-max", type=int, default=int(_read_env("SUPERVISOR_LOG_LINES_MAX", "200") or 200))
    parser.add_argument("--skew-seconds", type=int, default=int(_read_env("SUPERVISOR_SKEW_SECONDS", "120") or 120))
    parser.add_argument("--nonce-cache", type=int, default=int(_read_env("SUPERVISOR_NONCE_CACHE", "1024") or 1024))
    args = parser.parse_args()

    hmac_key = _read_env("SUPERVISOR_HMAC_KEY")
    if not hmac_key:
        print("SUPERVISOR_HMAC_KEY is required.", file=sys.stderr)
        return 1
    if not args.agent_unit:
        print("AGENT_UNIT_NAME is required.", file=sys.stderr)
        return 1

    supervisor = Supervisor(
        socket_path=args.socket_path,
        hmac_key=hmac_key,
        agent_unit_name=args.agent_unit,
        systemctl_mode=args.systemctl_mode,
        log_lines_max=args.log_lines_max,
        skew_seconds=args.skew_seconds,
        nonce_cache_size=args.nonce_cache,
    )
    supervisor.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

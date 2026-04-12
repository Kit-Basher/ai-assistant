from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any


_ALLOWED_COMMANDS = {"ollama", "sh"}


@dataclass(frozen=True)
class SafeCommandResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


class SafeRunner:
    def __init__(self, installer_script_path: str, output_limit: int = 16000) -> None:
        self.installer_script_path = str(Path(installer_script_path).resolve())
        self.output_limit = max(1024, int(output_limit))

    def run(self, command: list[str], *, timeout_seconds: float = 600.0) -> SafeCommandResult:
        if not command:
            raise ValueError("command is required")
        normalized = [str(part) for part in command]
        binary = os.path.basename(normalized[0])

        if binary not in _ALLOWED_COMMANDS:
            raise ValueError("command not allowed")
        if binary == "sh":
            if len(normalized) < 2:
                raise ValueError("installer script path is required")
            script_path = str(Path(normalized[1]).resolve())
            if script_path != self.installer_script_path:
                raise ValueError("only bundled installer script is allowed")

        env = self._safe_env()
        try:
            completed = subprocess.run(
                normalized,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
                env=env,
                check=False,
            )
            stdout, stdout_trunc = self._truncate(completed.stdout)
            stderr, stderr_trunc = self._truncate(completed.stderr)
            return SafeCommandResult(
                ok=completed.returncode == 0,
                returncode=int(completed.returncode),
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                truncated=stdout_trunc or stderr_trunc,
            )
        except subprocess.TimeoutExpired as exc:
            stdout, stdout_trunc = self._truncate(exc.stdout if isinstance(exc.stdout, str) else "")
            stderr, stderr_trunc = self._truncate(exc.stderr if isinstance(exc.stderr, str) else "")
            return SafeCommandResult(
                ok=False,
                returncode=124,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                truncated=stdout_trunc or stderr_trunc,
            )

    def _truncate(self, value: str) -> tuple[str, bool]:
        if len(value) <= self.output_limit:
            return value, False
        return value[: self.output_limit], True

    @staticmethod
    def _safe_env() -> dict[str, str]:
        allowed_base = {"PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "OLLAMA_HOST"}
        output: dict[str, str] = {}
        for key, value in os.environ.items():
            key_upper = key.upper()
            if any(token in key_upper for token in ("TOKEN", "SECRET", "KEY", "PASS", "AUTH")):
                continue
            if key_upper in allowed_base:
                output[key] = value
        if "PATH" not in output:
            output["PATH"] = os.environ.get("PATH", "")
        return output

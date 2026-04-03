from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    stdout: str
    stderr: str
    returncode: int | None
    error: str | None
    permission_denied: bool
    not_available: bool


def run_command(args: Iterable[str], timeout_s: float = 2.0) -> CommandResult:
    argv = list(args)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            args=argv,
            stdout="",
            stderr="",
            returncode=None,
            error=str(exc),
            permission_denied=False,
            not_available=True,
        )
    except PermissionError as exc:
        return CommandResult(
            args=argv,
            stdout="",
            stderr="",
            returncode=None,
            error=str(exc),
            permission_denied=True,
            not_available=True,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            args=argv,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            returncode=None,
            error="timeout",
            permission_denied=False,
            not_available=True,
        )

    stderr_lower = (completed.stderr or "").lower()
    permission_denied = "permission denied" in stderr_lower or "access denied" in stderr_lower
    not_available = "system has not been booted with systemd" in stderr_lower
    return CommandResult(
        args=argv,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        returncode=completed.returncode,
        error=None,
        permission_denied=permission_denied,
        not_available=not_available,
    )


_TOKEN_RE = re.compile(r"\b\d+:[A-Za-z0-9_-]{20,}\b")
_ENV_SECRET_RE = re.compile(r"(\b[A-Z0-9_]*(?:TOKEN|API_KEY)[A-Z0-9_]*=)([^\s]+)", re.IGNORECASE)


def redact_secrets(text: str) -> str:
    if not text:
        return text
    redacted_lines: list[str] = []
    for line in text.splitlines():
        line = _TOKEN_RE.sub("[REDACTED]", line)
        line = _ENV_SECRET_RE.sub(r"\1[REDACTED]", line)
        redacted_lines.append(line)
    return "\n".join(redacted_lines)

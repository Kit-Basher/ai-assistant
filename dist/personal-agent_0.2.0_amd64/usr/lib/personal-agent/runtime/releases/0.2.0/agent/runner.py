from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RunnerResult:
    mode: str
    allowlist_tag: str | None
    would_run: str | None
    blocked_reason: str | None
    returncode: int | None
    stdout: str | None
    stderr: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


class Runner:
    def __init__(self, mode: str = "sandbox") -> None:
        self.mode = mode
        self.last_result: RunnerResult | None = None

    def run(self, *_args: Any, **_kwargs: Any) -> RunnerResult:
        result = RunnerResult(
            mode=self.mode,
            allowlist_tag=None,
            would_run=None,
            blocked_reason="runner_not_configured",
            returncode=1,
            stdout="",
            stderr="",
            error="runner_not_configured",
        )
        self.last_result = result
        return result

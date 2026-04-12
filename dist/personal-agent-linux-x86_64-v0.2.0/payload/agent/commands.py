from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Command:
    name: str
    args: str


def parse_command(text: str) -> Command | None:
    if not text.startswith("/"):
        return None
    parts = text.strip().split(" ", 1)
    name = parts[0][1:]
    args = parts[1] if len(parts) > 1 else ""
    return Command(name=name, args=args)


def split_pipe_args(args: str, expected: int) -> list[str]:
    parts = [part.strip() for part in args.split("|")]
    while len(parts) < expected:
        parts.append("")
    return parts[:expected]

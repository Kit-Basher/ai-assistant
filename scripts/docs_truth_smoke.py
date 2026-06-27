#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = [ROOT / "README.md", ROOT / "docs"]

FORBIDDEN_PATTERNS = (
    re.compile(r"\bfinished agent\b", re.IGNORECASE),
    re.compile(r"\bagent is finished\b", re.IGNORECASE),
    re.compile(r"\bproduction[- ]ready\b", re.IGNORECASE),
    re.compile(r"\bfully complete\b", re.IGNORECASE),
    re.compile(r"\bvm proof complete\b", re.IGNORECASE),
    re.compile(r"\bvm[- ]proven\b", re.IGNORECASE),
)

NEGATION_HINTS = (
    "do not",
    "don't",
    "not ",
    "not-",
    "not_",
    "never ",
    "no ",
    "pending",
    "deferred",
    "has not",
    "have not",
    "without",
    "must not",
    "does not",
    "is not",
    "are not",
)


def _markdown_files() -> list[Path]:
    files: list[Path] = []
    for path in DOC_PATHS:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
    return sorted(set(files))


def _allowed_context(line: str) -> bool:
    lowered = line.lower()
    return any(hint in lowered for hint in NEGATION_HINTS)


def main() -> int:
    failures: list[str] = []
    checked = 0
    for path in _markdown_files():
        rel = path.relative_to(ROOT)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(errors="replace").splitlines()
        for lineno, line in enumerate(lines, start=1):
            checked += 1
            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line) and not _allowed_context(line):
                    failures.append(f"{rel}:{lineno}: {line.strip()}")

    print("# Personal Agent Docs Truth Smoke")
    print(f"Checked markdown lines: {checked}")
    if failures:
        print("DOCS_TRUTH_SMOKE: fail")
        for item in failures:
            print(f"- {item}")
        return 1
    print("Forbidden unqualified release claims: 0")
    print("DOCS_TRUTH_SMOKE: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


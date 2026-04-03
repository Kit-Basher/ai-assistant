from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_dump_routes() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "dump_routes.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_dump_routes_contains_core_routes() -> None:
    output = _run_dump_routes()
    assert "## Active Endpoints" in output
    assert "### GET" in output
    assert "### POST" in output

    for route in ("/health", "/chat", "/ask", "/llm/models/check"):
        assert f"- {route}" in output


def test_dump_routes_method_order_is_stable() -> None:
    output = _run_dump_routes()
    get_index = output.index("### GET")
    post_index = output.index("### POST")
    put_index = output.index("### PUT")
    delete_index = output.index("### DELETE")
    assert get_index < post_index < put_index < delete_index

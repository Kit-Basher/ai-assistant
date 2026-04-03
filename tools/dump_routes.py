#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.bootstrap.routes import METHOD_ORDER, extract_routes_from_api_server, format_routes_markdown


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    api_server_path = repo_root / "agent" / "api_server.py"
    routes = extract_routes_from_api_server(api_server_path)
    print(format_routes_markdown(routes), end="")


if __name__ == "__main__":
    main()

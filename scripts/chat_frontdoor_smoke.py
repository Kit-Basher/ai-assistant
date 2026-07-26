from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("Running isolated end-user /chat front-door smoke", flush=True)
    return int(pytest.main(["-q", str(ROOT / "tests" / "test_chat_frontdoor_contract.py")]))


if __name__ == "__main__":
    raise SystemExit(main())

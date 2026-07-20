#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1
MANIFEST_NAME = ".build-manifest.json"


def _source_files(repo_root: Path) -> list[Path]:
    desktop = repo_root / "desktop"
    roots = [desktop / "index.html", desktop / "package.json", desktop / "package-lock.json", desktop / "vite.config.js"]
    roots.extend(sorted((desktop / "src").rglob("*")))
    return [path for path in roots if path.is_file()]


def _output_files(repo_root: Path) -> list[Path]:
    dist = repo_root / "agent" / "webui" / "dist"
    return [path for path in sorted(dist.rglob("*")) if path.is_file() and path.name != MANIFEST_NAME]


def _tree_digest(repo_root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def build_payload(repo_root: Path) -> dict[str, object]:
    sources = _source_files(repo_root)
    outputs = _output_files(repo_root)
    if not sources:
        raise RuntimeError("web UI source files are missing under desktop/")
    if not outputs or not (repo_root / "agent" / "webui" / "dist" / "index.html").is_file():
        raise RuntimeError("web UI build output is missing under agent/webui/dist/")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": _tree_digest(repo_root, sources),
        "output_sha256": _tree_digest(repo_root, outputs),
        "source_files": [path.relative_to(repo_root).as_posix() for path in sources],
        "output_files": [path.relative_to(repo_root).as_posix() for path in outputs],
    }


def write_manifest(repo_root: Path) -> Path:
    payload = build_payload(repo_root)
    path = repo_root / "agent" / "webui" / "dist" / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_manifest(repo_root: Path) -> Path:
    path = repo_root / "agent" / "webui" / "dist" / MANIFEST_NAME
    if not path.is_file():
        raise RuntimeError("web UI build manifest is missing; run bash scripts/build_webui.sh")
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("web UI build manifest is unreadable; rebuild the web UI") from exc
    expected = build_payload(repo_root)
    for field in ("schema_version", "source_sha256", "output_sha256", "source_files", "output_files"):
        if recorded.get(field) != expected[field]:
            raise RuntimeError(
                f"web UI build is stale or modified ({field} mismatch); run bash scripts/build_webui.sh"
            )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or verify the Personal Agent web UI build manifest.")
    parser.add_argument("mode", choices=("write", "verify"))
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    try:
        path = write_manifest(repo_root) if args.mode == "write" else verify_manifest(repo_root)
    except RuntimeError as exc:
        parser.exit(1, f"Personal Agent web UI: {exc}\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

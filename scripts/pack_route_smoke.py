#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = os.environ.get("AGENT_API_BASE_URL") or "http://127.0.0.1:8765"


def _first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _json_from_response(body: str) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
            return {
                "ok": status < 400,
                "status": status,
                "payload": _json_from_response(raw),
                "raw": raw,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": int(getattr(exc, "code", 500)),
            "payload": _json_from_response(raw),
            "raw": raw,
            "error": str(exc),
        }
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "status": 0,
            "payload": {},
            "raw": "",
            "error": f"transport error: {exc.reason}",
        }
    except Exception as exc:  # pragma: no cover - live smoke defensive guard
        return {
            "ok": False,
            "status": 0,
            "payload": {},
            "raw": "",
            "error": f"transport error: {exc}",
        }


def _make_local_fixture(root: Path) -> dict[str, str]:
    pack_dir = root / "pack-smoke-demo"
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "references").mkdir(parents=True, exist_ok=True)
    (pack_dir / "SKILL.md").write_text(
        "---\n"
        "id: pack-smoke-demo\n"
        "name: Pack Smoke Demo\n"
        "version: 0.1.0\n"
        "description: safe text pack for live smoke testing\n"
        "---\n"
        "# Pack Smoke Demo\n\n"
        "Use the reference notes only.\n",
        encoding="utf-8",
    )
    (pack_dir / "references" / "guide.md").write_text(
        "# Guide\n\n"
        "This pack is intentionally simple and safe.\n",
        encoding="utf-8",
    )

    catalog_path = root / "registry-catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "id": "pack-smoke-demo",
                        "name": "Pack Smoke Demo",
                        "summary": "A safe text skill pack for live smoke testing.",
                        "author": "Codex",
                        "source_url": str(pack_dir),
                        "has_skill_md": True,
                        "tags": ["smoke", "demo"],
                    }
                ]
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "source_id": f"pack-smoke-{int(time.time())}",
        "remote_id": "pack-smoke-demo",
        "pack_dir": str(pack_dir),
        "catalog_path": str(catalog_path),
    }


def _make_native_fixture(root: Path) -> str:
    pack_dir = root / "pack-smoke-native"
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "package.json").write_text(
        json.dumps({"name": "pack-smoke-native", "version": "1.0.0"}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (pack_dir / "handler.js").write_text(
        "export function run() {\n"
        "  return true;\n"
        "}\n",
        encoding="utf-8",
    )
    return str(pack_dir)


def _is_dead_end(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        phrase in lowered
        for phrase in (
            "need more context",
            "i need more context",
            "can't help",
            "cannot help",
            "couldn't read",
            "could not read",
            "did not preview",
            "did not query",
            "no discovery sources are enabled",
            "runtime state",
        )
    )


def _status_text(result: dict[str, Any]) -> str:
    status = result.get("status")
    if isinstance(status, int) and status:
        return str(status)
    return "ok" if bool(result.get("ok")) else "error"


def _step_result(route: str, result: dict[str, Any], text: str, *, warnings: list[str] | None = None) -> None:
    print(f"route: {route}")
    print(f"status: {_status_text(result)}")
    print(f"first_line: {_first_line(text)}")
    print(f"dead_end_warnings: {', '.join(warnings or []) if warnings else 'none'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a live, non-blocking smoke across pack discovery, preview, and install."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the live API server.")
    parser.add_argument("--keep-source", action="store_true", help="Leave the temporary catalog source in place.")
    args = parser.parse_args(argv)

    exit_code = 0
    created_source_id = ""
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture = _make_local_fixture(Path(tmpdir))
        source_id = fixture["source_id"]
        created_source_id = source_id
        catalog_path = fixture["catalog_path"]
        pack_dir = fixture["pack_dir"]
        native_pack_dir = _make_native_fixture(Path(tmpdir))
        remote_id = fixture["remote_id"]

        create_payload = {
            "source_id": source_id,
            "name": "Pack Smoke Local Source",
            "kind": "local_catalog",
            "base_url": catalog_path,
            "enabled": True,
            "supports_search": True,
            "supports_preview": True,
            "supports_compare_hint": False,
            "notes": "temporary operator smoke source",
        }
        create_result = _request_json(str(args.base_url), "POST", "/pack_sources/catalog", create_payload, timeout=8.0)
        create_body = create_result.get("payload") if isinstance(create_result.get("payload"), dict) else {}
        source_payload = create_body.get("source") if isinstance(create_body.get("source"), dict) else {}
        source_name = str(source_payload.get("name") or create_payload["name"]).strip()
        create_text = str(
            create_body.get("message")
            or create_body.get("summary")
            or source_name
            or create_result.get("error")
            or ""
        )
        create_warnings: list[str] = []
        if not create_result.get("ok"):
            create_warnings.append("source creation failed")
            exit_code = 1
        elif str(source_payload.get("id") or "").strip() != source_id:
            create_warnings.append("source id mismatch")
            exit_code = 1
        _step_result(f"POST /pack_sources/catalog ({source_id})", create_result, create_text, warnings=create_warnings)

        list_result = _request_json(str(args.base_url), "GET", "/pack_sources", timeout=8.0)
        list_body = list_result.get("payload") if isinstance(list_result.get("payload"), dict) else {}
        sources = list_body.get("sources") if isinstance(list_body.get("sources"), list) else []
        source_row = next((row for row in sources if isinstance(row, dict) and str(row.get("id") or "").strip() == source_id), None)
        list_text = str(
            source_row.get("name")
            if isinstance(source_row, dict)
            else list_body.get("summary")
            or list_result.get("error")
            or ""
        )
        list_warnings: list[str] = []
        if not list_result.get("ok"):
            list_warnings.append("pack source listing failed")
            exit_code = 1
        elif source_row is None:
            list_warnings.append("temporary source not visible in /pack_sources")
            exit_code = 1
        _step_result("GET /pack_sources", list_result, list_text, warnings=list_warnings)

        packs_result = _request_json(
            str(args.base_url),
            "GET",
            f"/pack_sources/{urllib.parse.quote(source_id)}/packs",
            timeout=8.0,
        )
        packs_body = packs_result.get("payload") if isinstance(packs_result.get("payload"), dict) else {}
        packs = packs_body.get("packs") if isinstance(packs_body.get("packs"), list) else []
        pack_row = next((row for row in packs if isinstance(row, dict) and str(row.get("remote_id") or "").strip() == remote_id), None)
        pack_text = str(
            pack_row.get("name")
            if isinstance(pack_row, dict)
            else packs_body.get("message")
            or packs_body.get("summary")
            or packs_result.get("error")
            or ""
        )
        packs_warnings: list[str] = []
        if not packs_result.get("ok"):
            packs_warnings.append("pack list failed")
            exit_code = 1
        elif pack_row is None:
            packs_warnings.append("expected pack not present in list")
            exit_code = 1
        _step_result(f"GET /pack_sources/{source_id}/packs", packs_result, pack_text, warnings=packs_warnings)

        search_result = _request_json(
            str(args.base_url),
            "GET",
            f"/pack_sources/{urllib.parse.quote(source_id)}/search?q=smoke",
            timeout=8.0,
        )
        search_body = search_result.get("payload") if isinstance(search_result.get("payload"), dict) else {}
        search_payload = search_body.get("search") if isinstance(search_body.get("search"), dict) else {}
        search_items = search_payload.get("results") if isinstance(search_payload.get("results"), list) else []
        search_row = next((row for row in search_items if isinstance(row, dict) and str(row.get("remote_id") or "").strip() == remote_id), None)
        search_text = str(
            search_row.get("name")
            if isinstance(search_row, dict)
            else search_payload.get("summary")
            or search_body.get("message")
            or search_result.get("error")
            or ""
        )
        search_warnings: list[str] = []
        if not search_result.get("ok"):
            search_warnings.append("pack search failed")
            exit_code = 1
        elif search_row is None:
            search_warnings.append("expected pack not present in search results")
            exit_code = 1
        _step_result(f"GET /pack_sources/{source_id}/search?q=smoke", search_result, search_text, warnings=search_warnings)

        preview_result = _request_json(
            str(args.base_url),
            "GET",
            f"/pack_sources/{urllib.parse.quote(source_id)}/packs/{urllib.parse.quote(remote_id)}/preview",
            timeout=8.0,
        )
        preview_body = preview_result.get("payload") if isinstance(preview_result.get("payload"), dict) else {}
        preview = preview_body.get("preview") if isinstance(preview_body.get("preview"), dict) else {}
        preview_text = str(
            preview.get("summary")
            or preview_body.get("message")
            or preview_result.get("error")
            or ""
        )
        preview_warnings: list[str] = []
        lowered_preview = preview_text.lower()
        if not preview_result.get("ok"):
            preview_warnings.append("preview failed")
            exit_code = 1
        if "read-only" not in lowered_preview:
            preview_warnings.append("preview is missing explicit read-only wording")
            exit_code = 1
        if "already installed" in lowered_preview or "has been installed" in lowered_preview or "was installed" in lowered_preview:
            preview_warnings.append("preview sounds like action already happened")
            exit_code = 1
        if _is_dead_end(preview_text):
            preview_warnings.append("preview dead-end wording")
            exit_code = 1
        _step_result(
            f"GET /pack_sources/{source_id}/packs/{remote_id}/preview",
            preview_result,
            preview_text,
            warnings=preview_warnings,
        )

        install_target = str((preview.get("install_handoff") or {}).get("source") or pack_dir).strip()
        install_payload = {"path": install_target}
        install_result = _request_json(str(args.base_url), "POST", "/packs/install", install_payload, timeout=20.0)
        install_body = install_result.get("payload") if isinstance(install_result.get("payload"), dict) else {}
        install_text = str(
            install_body.get("message")
            or install_body.get("why")
            or (install_body.get("review") or {}).get("summary")
            or install_result.get("error")
            or ""
        )
        install_status = str((install_body.get("normalization_result") or {}).get("status") or "").strip().lower()
        install_warnings: list[str] = []
        lowered_install = install_text.lower()
        if not install_result.get("ok"):
            install_warnings.append("install failed")
            exit_code = 1
        if install_status not in {"normalized", "partial_safe_import", "blocked"}:
            install_warnings.append(f"unexpected install status: {install_status or 'missing'}")
            exit_code = 1
        if install_status == "normalized" and "normalized" not in lowered_install:
            install_warnings.append("normalized install wording missing")
            exit_code = 1
        if install_status == "partial_safe_import" and not any(token in lowered_install for token in ("safe parts", "quarantined")):
            install_warnings.append("partial import wording missing")
            exit_code = 1
        if install_status == "blocked" and "blocked" not in lowered_install:
            install_warnings.append("blocked install wording missing")
            exit_code = 1
        if _is_dead_end(install_text):
            install_warnings.append("install dead-end wording")
            exit_code = 1
        _step_result("/packs/install", install_result, install_text, warnings=install_warnings)

        native_install_result = _request_json(
            str(args.base_url),
            "POST",
            "/packs/install",
            {"source": native_pack_dir},
            timeout=20.0,
        )
        native_install_body = native_install_result.get("payload") if isinstance(native_install_result.get("payload"), dict) else {}
        native_install_text = str(
            native_install_body.get("message")
            or native_install_body.get("why")
            or (native_install_body.get("review") or {}).get("summary")
            or native_install_result.get("error")
            or ""
        )
        native_install_status = str((native_install_body.get("normalization_result") or {}).get("status") or "").strip().lower()
        native_install_warnings: list[str] = []
        lowered_native_install = native_install_text.lower()
        if not native_install_result.get("ok"):
            native_install_warnings.append("blocked native install failed")
            exit_code = 1
        if native_install_status != "blocked":
            native_install_warnings.append(f"unexpected blocked-native status: {native_install_status or 'missing'}")
            exit_code = 1
        if "blocked" not in lowered_native_install:
            native_install_warnings.append("blocked install wording missing")
            exit_code = 1
        if _is_dead_end(native_install_text):
            native_install_warnings.append("blocked install dead-end wording")
            exit_code = 1
        _step_result(
            "/packs/install (blocked native)",
            native_install_result,
            native_install_text,
            warnings=native_install_warnings,
        )

        remote_source_url = str(os.environ.get("PACK_ROUTE_SMOKE_REMOTE_URL") or "").strip()
        if remote_source_url:
            remote_source_kind = str(os.environ.get("PACK_ROUTE_SMOKE_REMOTE_SOURCE_KIND") or "github_archive").strip() or "github_archive"
            remote_ref = str(os.environ.get("PACK_ROUTE_SMOKE_REMOTE_REF") or "main").strip() or "main"
            remote_install_result = _request_json(
                str(args.base_url),
                "POST",
                "/packs/install",
                {
                    "source": remote_source_url,
                    "source_kind": remote_source_kind,
                    "ref": remote_ref,
                },
                timeout=30.0,
            )
            remote_install_body = (
                remote_install_result.get("payload") if isinstance(remote_install_result.get("payload"), dict) else {}
            )
            remote_install_text = str(
                remote_install_body.get("message")
                or remote_install_body.get("why")
                or (remote_install_body.get("review") or {}).get("summary")
                or remote_install_result.get("error")
                or ""
            )
            remote_install_status = str((remote_install_body.get("normalization_result") or {}).get("status") or "").strip().lower()
            remote_install_warnings: list[str] = []
            lowered_remote_install = remote_install_text.lower()
            if not remote_install_result.get("ok"):
                remote_install_warnings.append("remote install failed")
                exit_code = 1
            if remote_install_status not in {"normalized", "partial_safe_import", "blocked"}:
                remote_install_warnings.append(f"unexpected remote status: {remote_install_status or 'missing'}")
                exit_code = 1
            if remote_install_status == "normalized" and "normalized" not in lowered_remote_install:
                remote_install_warnings.append("normalized install wording missing")
                exit_code = 1
            if remote_install_status == "partial_safe_import" and not any(
                token in lowered_remote_install for token in ("safe parts", "quarantined")
            ):
                remote_install_warnings.append("partial import wording missing")
                exit_code = 1
            if remote_install_status == "blocked" and "blocked" not in lowered_remote_install:
                remote_install_warnings.append("blocked install wording missing")
                exit_code = 1
            if _is_dead_end(remote_install_text):
                remote_install_warnings.append("remote install dead-end wording")
                exit_code = 1
            _step_result(
                "/packs/install (remote)",
                remote_install_result,
                remote_install_text,
                warnings=remote_install_warnings,
            )

        if not args.keep_source:
            cleanup_result = _request_json(
                str(args.base_url),
                "DELETE",
                f"/pack_sources/catalog/{urllib.parse.quote(created_source_id)}",
                timeout=8.0,
            )
            if not cleanup_result.get("ok"):
                print(f"cleanup_warning: could not delete temporary source {created_source_id}")

    if exit_code == 0:
        print("dead_end_warnings: none")
    else:
        print("dead_end_warnings: one or more pack-route checks failed")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

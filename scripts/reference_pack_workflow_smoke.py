#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures" / "reference_packs"
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
    timeout: float = 8.0,
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
            return {"ok": status < 400, "status": status, "payload": _json_from_response(raw), "raw": raw}
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
        return {"ok": False, "status": 0, "payload": {}, "raw": "", "error": f"transport error: {exc.reason}"}
    except Exception as exc:  # pragma: no cover - defensive live smoke guard
        return {"ok": False, "status": 0, "payload": {}, "raw": "", "error": f"transport error: {exc}"}


def _status_text(result: dict[str, Any]) -> str:
    status = result.get("status")
    if isinstance(status, int) and status:
        return str(status)
    return "ok" if bool(result.get("ok")) else "error"


def _dead_end_warnings(text: str, *, ok: bool) -> list[str]:
    lowered = str(text or "").lower()
    warnings: list[str] = []
    if not ok:
        warnings.append("transport not ok")
    if not str(text or "").strip():
        warnings.append("empty first line")
    if any(
        phrase in lowered
        for phrase in (
            "need more context",
            "i need more context",
            "could not read",
            "couldn't read",
            "can't help",
            "cannot help",
            "no discovery sources are enabled",
            "need to know",
        )
    ):
        warnings.append("dead-end wording")
    return warnings


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog_entry(*, fixture_name: str, source_manifest: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    label_map = {
        "anthropic_clean_skill": "Repo Helper",
        "vercel_structured_skill": "React View Transitions",
        "fragmented_prompt_repo": "Fragmented Prompt Repo",
        "openclaw_mixed_skill": "PAI Mixed Skill",
        "vercel_blocked_native_skill": "Skills CLI Bundle",
    }
    name = label_map.get(fixture_name, fixture_name.replace("_", " ").title())
    return {
        "id": fixture_name,
        "name": name,
        "summary": str(source_manifest.get("why_this_exists") or expected.get("outcome_category") or "").strip(),
        "author": str(source_manifest.get("upstream_repo") or "").strip() or None,
        "source_url": str(source_manifest.get("source_url") or "").strip() or None,
        "latest_ref_hint": str(source_manifest.get("pinned_ref") or "").strip() or None,
        "artifact_type_hint": "native_code_pack" if expected.get("outcome_category") == "blocked_install" else "portable_text_skill",
        "tags": [fixture_name, str(expected.get("outcome_category") or "").strip()],
        "has_skill_md": expected.get("outcome_category") != "blocked_install",
        "requires_execution": expected.get("outcome_category") == "blocked_install",
        "is_plugin": expected.get("outcome_category") == "blocked_install",
    }


def _chat_summary(base_url: str, prompt: str, *, user_id: str, trace_id: str) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "purpose": "chat",
        "task_type": "chat",
        "source_surface": "operator_smoke",
        "user_id": user_id,
        "thread_id": f"{user_id}:thread",
        "trace_id": trace_id,
    }
    result = _request_json(base_url, "POST", "/chat", payload, timeout=45.0)
    payload_json = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    assistant = payload_json.get("assistant") if isinstance(payload_json.get("assistant"), dict) else {}
    meta = payload_json.get("meta") if isinstance(payload_json.get("meta"), dict) else {}
    text = str(assistant.get("content") or payload_json.get("message") or payload_json.get("error") or "").strip()
    return {
        "result": result,
        "route": str(meta.get("route") or "").strip().lower() or "unknown",
        "text": text,
        "first_line": _first_line(text),
        "warnings": _dead_end_warnings(text, ok=bool(result.get("ok"))),
    }


def _step(route: str, result: dict[str, Any], text: str, *, warnings: list[str] | None = None) -> None:
    print(f"route: {route}")
    print(f"status: {_status_text(result)}")
    print(f"first_line: {_first_line(text)}")
    print(f"dead_end_warnings: {', '.join(warnings or []) if warnings else 'none'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a live end-to-end pack workflow smoke against the canonical runtime."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the live API server.")
    parser.add_argument("--keep-sources", action="store_true", help="Leave temporary catalog sources in place.")
    args = parser.parse_args(argv)

    exit_code = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        for fixture_name in (
            "anthropic_clean_skill",
            "vercel_structured_skill",
            "fragmented_prompt_repo",
            "openclaw_mixed_skill",
            "vercel_blocked_native_skill",
        ):
            fixture_root = FIXTURES_ROOT / fixture_name
            source_dir = fixture_root / "source"
            source_manifest = _read_json(fixture_root / "source_manifest.json")
            expected = _read_json(fixture_root / "expected.json")
            source_id = f"workflow-{fixture_name}"
            catalog_path = tmp_root / f"{fixture_name}-catalog.json"
            catalog_path.write_text(
                json.dumps({"packs": [_catalog_entry(fixture_name=fixture_name, source_manifest=source_manifest, expected=expected)]}, ensure_ascii=True, indent=2)
                + "\n",
                encoding="utf-8",
            )

            create_payload = {
                "source_id": source_id,
                "name": f"{source_manifest.get('upstream_repo', fixture_name)} source",
                "kind": "local_catalog",
                "base_url": str(catalog_path),
                "enabled": True,
                "supports_search": True,
                "supports_preview": True,
                "supports_compare_hint": False,
                "notes": f"workflow smoke for {fixture_name}",
            }
            create_result = _request_json(args.base_url, "POST", "/pack_sources/catalog", create_payload, timeout=8.0)
            create_body = create_result.get("payload") if isinstance(create_result.get("payload"), dict) else {}
            create_text = str(
                ((create_body.get("source") or {}) if isinstance(create_body.get("source"), dict) else {}).get("name")
                or create_body.get("message")
                or create_body.get("summary")
                or create_result.get("error")
                or ""
            )
            create_warnings: list[str] = []
            if not create_result.get("ok"):
                create_warnings.append("source creation failed")
                exit_code = 1
            _step(f"POST /pack_sources/catalog ({source_id})", create_result, create_text, warnings=create_warnings)

            list_result = _request_json(args.base_url, "GET", "/pack_sources", timeout=8.0)
            list_body = list_result.get("payload") if isinstance(list_result.get("payload"), dict) else {}
            sources = list_body.get("sources") if isinstance(list_body.get("sources"), list) else []
            source_row = next((row for row in sources if isinstance(row, dict) and str(row.get("id") or "").strip() == source_id), None)
            list_warnings: list[str] = []
            if not list_result.get("ok"):
                list_warnings.append("pack source listing failed")
                exit_code = 1
            if source_row is None:
                list_warnings.append("source not visible in /pack_sources")
                exit_code = 1
            _step("GET /pack_sources", list_result, str((source_row or {}).get("name") or list_body.get("summary") or ""), warnings=list_warnings)

            pack_name = str(source_manifest.get("source_kind") or fixture_name).strip() or fixture_name
            packs_result = _request_json(args.base_url, "GET", f"/pack_sources/{urllib.parse.quote(source_id)}/packs", timeout=8.0)
            packs_body = packs_result.get("payload") if isinstance(packs_result.get("payload"), dict) else {}
            packs = packs_body.get("packs") if isinstance(packs_body.get("packs"), list) else []
            pack_row = next((row for row in packs if isinstance(row, dict) and str(row.get("name") or "").strip() == _catalog_entry(
                fixture_name=fixture_name, source_manifest=source_manifest, expected=expected
            )["name"]), None)
            pack_warnings: list[str] = []
            if not packs_result.get("ok"):
                pack_warnings.append("pack list failed")
                exit_code = 1
            if pack_row is None:
                pack_warnings.append("expected pack not present in list")
                exit_code = 1
            _step(f"GET /pack_sources/{source_id}/packs", packs_result, str((pack_row or {}).get("name") or ""), warnings=pack_warnings)

            search_query = {
                "anthropic_clean_skill": "repo helper",
                "vercel_structured_skill": "react",
                "fragmented_prompt_repo": "prompt",
                "openclaw_mixed_skill": "pai",
                "vercel_blocked_native_skill": "skills",
            }[fixture_name]
            search_result = _request_json(
                args.base_url,
                "GET",
                f"/pack_sources/{urllib.parse.quote(source_id)}/search?q={urllib.parse.quote(search_query)}",
                timeout=8.0,
            )
            search_body = search_result.get("payload") if isinstance(search_result.get("payload"), dict) else {}
            search_payload = search_body.get("search") if isinstance(search_body.get("search"), dict) else {}
            search_items = search_payload.get("results") if isinstance(search_payload.get("results"), list) else []
            search_row = next((row for row in search_items if isinstance(row, dict) and str(row.get("name") or "").strip() == _catalog_entry(
                fixture_name=fixture_name, source_manifest=source_manifest, expected=expected
            )["name"]), None)
            search_warnings: list[str] = []
            if not search_result.get("ok"):
                search_warnings.append("search failed")
                exit_code = 1
            if search_row is None:
                search_warnings.append("expected pack not present in search results")
                exit_code = 1
            _step(f"GET /pack_sources/{source_id}/search?q={search_query}", search_result, str((search_row or {}).get("name") or ""), warnings=search_warnings)

            preview_result = _request_json(
                args.base_url,
                "GET",
                f"/pack_sources/{urllib.parse.quote(source_id)}/packs/{urllib.parse.quote(str((pack_row or {}).get('remote_id') or fixture_name))}/preview",
                timeout=8.0,
            )
            preview_body = preview_result.get("payload") if isinstance(preview_result.get("payload"), dict) else {}
            preview = preview_body.get("preview") if isinstance(preview_body.get("preview"), dict) else {}
            preview_text = str(preview.get("summary") or preview_body.get("message") or preview_result.get("error") or "")
            preview_warnings: list[str] = []
            if not preview_result.get("ok"):
                preview_warnings.append("preview failed")
                exit_code = 1
            if "read-only" not in preview_text.lower():
                preview_warnings.append("preview missing read-only wording")
                exit_code = 1
            if any(phrase in preview_text.lower() for phrase in ("already installed", "has been installed", "was installed")):
                preview_warnings.append("preview sounds like mutation already happened")
                exit_code = 1
            if _dead_end_warnings(preview_text, ok=True):
                preview_warnings.append("preview dead-end wording")
                exit_code = 1
            _step(
                f"GET /pack_sources/{source_id}/packs/{str((pack_row or {}).get('remote_id') or fixture_name)}/preview",
                preview_result,
                preview_text,
                warnings=preview_warnings,
            )

            install_result = _request_json(args.base_url, "POST", "/packs/install", {"source": str(source_dir)}, timeout=25.0)
            install_body = install_result.get("payload") if isinstance(install_result.get("payload"), dict) else {}
            install_text = str(
                install_body.get("message")
                or install_body.get("why")
                or (install_body.get("review") or {}).get("summary")
                or install_result.get("error")
                or ""
            )
            install_payload = install_body.get("normalization_result") if isinstance(install_body.get("normalization_result"), dict) else {}
            install_status = str(install_payload.get("status") or "").strip().lower()
            install_outcome_map = {
                "normalized": "normalized_safe_text",
                "partial_safe_import": "partial_safe_import",
                "blocked": "blocked_install",
            }
            actual_outcome = install_outcome_map.get(install_status, "unknown")
            install_warnings: list[str] = []
            if not install_result.get("ok"):
                install_warnings.append("install failed")
                exit_code = 1
            if actual_outcome != str(expected.get("outcome_category") or ""):
                install_warnings.append(
                    f"unexpected outcome: expected {expected.get('outcome_category')} got {actual_outcome}"
                )
                exit_code = 1
            if install_status == "normalized" and "normalized" not in install_text.lower():
                install_warnings.append("normalized install wording missing")
                exit_code = 1
            if install_status == "partial_safe_import" and not any(token in install_text.lower() for token in ("partial", "safe parts", "quarantined")):
                install_warnings.append("partial import wording missing")
                exit_code = 1
            if install_status == "blocked" and "blocked" not in install_text.lower():
                install_warnings.append("blocked install wording missing")
                exit_code = 1
            _step("/packs/install", install_result, install_text, warnings=install_warnings)

            pack_row_live = None
            packs_after_install = _request_json(args.base_url, "GET", "/packs", timeout=8.0)
            packs_after_install_body = packs_after_install.get("payload") if isinstance(packs_after_install.get("payload"), dict) else {}
            packs_after_install_rows = packs_after_install_body.get("packs") if isinstance(packs_after_install_body.get("packs"), list) else []
            canonical_id = str((install_body.get("pack") or {}).get("canonical_id") or "").strip()
            pack_row_live = next((row for row in packs_after_install_rows if isinstance(row, dict) and str(row.get("pack_id") or row.get("canonical_id") or "").strip() == canonical_id), None)
            if pack_row_live is None:
                exit_code = 1
                print("pack_verification: missing from /packs")
            else:
                normalized_path = str(pack_row_live.get("normalized_path") or "").strip()
                review_path = Path(normalized_path) / "metadata" / "review.json"
                normalization_path = Path(normalized_path) / "metadata" / "normalization.json"
                source_path = Path(normalized_path) / "metadata" / "source.json"
                for check_path in (review_path, normalization_path, source_path):
                    if not check_path.is_file():
                        exit_code = 1
                        print(f"pack_verification: missing {check_path}")

            chat_prompt_map = {
                "anthropic_clean_skill": "What is Repo Helper for?",
                "vercel_structured_skill": "How should I use React view transitions safely?",
                "fragmented_prompt_repo": "Fragmented Prompt Repo: how should I turn a vague request into clear steps?",
                "openclaw_mixed_skill": "PAI bundle: summarize the safe text guidance you kept and what you stripped.",
                "vercel_blocked_native_skill": "Can I use this skills CLI bundle directly?",
            }
            chat_summary = _chat_summary(
                args.base_url,
                chat_prompt_map[fixture_name],
                user_id=f"workflow-smoke-{fixture_name}",
                trace_id=f"workflow-smoke-{fixture_name}-{int(os.getpid())}",
            )
            chat_text = str(chat_summary.get("text") or "")
            usable = False
            if actual_outcome in {"normalized_safe_text", "partial_safe_import"}:
                usable = any(
                    token.lower() in chat_text.lower()
                    for token in (
                        "repo helper",
                        "react view transitions",
                        "fragmented prompt repo",
                        "pai",
                        "skills cli bundle",
                    )
                )
            chat_warnings = list(chat_summary.get("warnings") or [])
            if actual_outcome in {"normalized_safe_text", "partial_safe_import"} and not usable:
                chat_warnings.append("installed-but-not-usable-in-chat")
                exit_code = 1
            _step(f"POST /chat ({fixture_name})", chat_summary["result"], chat_text, warnings=chat_warnings)

            removal_result = {"ok": False, "status": 0, "payload": {}, "error": "skipped"}
            removal_text = ""
            removal_warnings: list[str] = []
            if canonical_id:
                removal_result = _request_json(
                    args.base_url,
                    "DELETE",
                    f"/packs/{urllib.parse.quote(canonical_id)}",
                    timeout=8.0,
                )
                removal_body = removal_result.get("payload") if isinstance(removal_result.get("payload"), dict) else {}
                removal_text = str(
                    removal_body.get("message")
                    or removal_body.get("why")
                    or removal_body.get("summary")
                    or removal_result.get("error")
                    or ""
                )
                if not removal_result.get("ok"):
                    removal_warnings.append("remove failed")
                    exit_code = 1
            else:
                removal_warnings.append("no canonical pack id to remove")
                exit_code = 1
            _step(f"DELETE /packs/{canonical_id or fixture_name}", removal_result, removal_text, warnings=removal_warnings)

            removed_pack_check = _request_json(args.base_url, "GET", "/packs", timeout=8.0)
            removed_pack_body = removed_pack_check.get("payload") if isinstance(removed_pack_check.get("payload"), dict) else {}
            removed_pack_rows = removed_pack_body.get("packs") if isinstance(removed_pack_body.get("packs"), list) else []
            still_visible = any(
                isinstance(row, dict) and str(row.get("pack_id") or row.get("canonical_id") or "").strip() == canonical_id
                for row in removed_pack_rows
            )
            if still_visible:
                exit_code = 1

            cleanup_ok = bool(removal_result.get("ok")) and not still_visible
            if cleanup_ok and pack_row_live is not None:
                normalized_path = Path(str(pack_row_live.get("normalized_path") or "").strip())
                quarantine_path = Path(str(pack_row_live.get("quarantine_path") or "").strip())
                if normalized_path and normalized_path.exists():
                    cleanup_ok = False
                if quarantine_path and quarantine_path.exists():
                    cleanup_ok = False
            if not cleanup_ok:
                exit_code = 1

            post_remove_chat = _chat_summary(
                args.base_url,
                chat_prompt_map[fixture_name],
                user_id=f"workflow-smoke-{fixture_name}-post-remove",
                trace_id=f"workflow-smoke-{fixture_name}-{int(os.getpid())}-post-remove",
            )
            post_remove_text = str(post_remove_chat.get("text") or "")
            post_remove_warnings = list(post_remove_chat.get("warnings") or [])
            lowered_post_remove = post_remove_text.lower()
            removed_notice_ok = any(token in lowered_post_remove for token in ("was removed", "was deleted", "reinstall it"))
            if actual_outcome in {"normalized_safe_text", "partial_safe_import"} and not removed_notice_ok and any(
                token.lower() in lowered_post_remove
                for token in (
                    "repo helper",
                    "react view transitions",
                    "fragmented prompt repo",
                    "pai",
                    "skills cli bundle",
                )
            ):
                post_remove_warnings.append("removed pack still active in chat")
            _step(f"POST /chat after remove ({fixture_name})", post_remove_chat["result"], post_remove_text, warnings=post_remove_warnings)

            cleanup_result = _request_json(args.base_url, "DELETE", f"/pack_sources/catalog/{urllib.parse.quote(source_id)}", timeout=8.0)
            cleanup_text = str(
                (cleanup_result.get("payload") or {}).get("message")
                or (cleanup_result.get("payload") or {}).get("summary")
                or cleanup_result.get("error")
                or ""
            )
            cleanup_warnings: list[str] = []
            if not cleanup_result.get("ok"):
                cleanup_warnings.append("source cleanup failed")
                exit_code = 1
            _step(f"DELETE /pack_sources/catalog/{source_id}", cleanup_result, cleanup_text, warnings=cleanup_warnings)

            post_cleanup = _request_json(args.base_url, "GET", "/pack_sources", timeout=8.0)
            post_cleanup_body = post_cleanup.get("payload") if isinstance(post_cleanup.get("payload"), dict) else {}
            post_cleanup_sources = post_cleanup_body.get("sources") if isinstance(post_cleanup_body.get("sources"), list) else []
            still_present = any(isinstance(row, dict) and str(row.get("id") or "").strip() == source_id for row in post_cleanup_sources)
            if still_present:
                exit_code = 1

            print(f"[{fixture_name}] discover: {'yes' if create_result.get('ok') and list_result.get('ok') and packs_result.get('ok') and search_result.get('ok') else 'no'}")
            print(f"[{fixture_name}] preview: {'yes' if preview_result.get('ok') else 'no'}")
            print(f"[{fixture_name}] install: {actual_outcome}")
            print(f"[{fixture_name}] usable_in_agent: {'yes' if usable else 'no'}")
            print(f"[{fixture_name}] remove_uninstall: {'yes' if removal_result.get('ok') else 'no'}")
            print(f"[{fixture_name}] cleanup_clean: {'yes' if cleanup_ok and not still_present else 'no'}")

            if not args.keep_sources:
                # The catalog source is removed via the live API; the installed pack is intentionally left in-place
                # because the current runtime exposes no safe uninstall path for external packs.
                pass

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from agent.packs.managed_adapters import ADAPTER_LOCAL_FILE_IMPORT, USER_SELECTED_FILE_ONLY


_YOUTUBE_HISTORY_CAPABILITIES = {
    "youtube_history_search",
    "private_history_search",
}


def _stable_scaffold_id(capability: str, title: str) -> str:
    payload = json.dumps(
        {"capability": str(capability or "").strip().lower(), "title": str(title or "").strip()},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "scaffold-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_scaffold_preview(capability: str | None, *, user_goal: str | None = None) -> dict[str, Any] | None:
    capability_key = str(capability or "").strip().lower()
    if capability_key not in _YOUTUBE_HISTORY_CAPABILITIES:
        return None

    title = "YouTube History Search"
    summary = (
        "A draft local skill for reviewing how a future safe adapter could work with a user-selected "
        "Google Takeout YouTube watch-history export."
    )
    proposed_skill_doc = "\n\n".join(
        [
            "# YouTube History Search",
            "## Purpose\nSearch a local, user-provided YouTube watch-history export without uploading private history.",
            "## When to Use\nUse when the user wants to find a previously watched YouTube video from their own history.",
            "## Inputs\nA local Google Takeout watch-history file selected by the user in a future create/import phase.",
            "## Behavior\nCurrent safe runtime can validate a selected Takeout file grant and dry-run access only. Future core-owned adapter operations would be required before reading, parsing, indexing, or searching history contents.",
            "## Constraints\nNo OAuth in v1. No browser scraping in v1. No transcript fetching in v1. No uploads of watch history. No video or audio downloads.",
            "## Privacy\nRaw watch-history rows, full URLs, account identifiers, and search terms from the export must be excluded from logs and support context by default.",
            "## Example Prompts\n- Find a YouTube video I watched about a specific topic.\n- Search my local Takeout history for a video from a channel or time period.",
        ]
    )
    files_to_create = [
        {
            "path": "SKILL.md",
            "purpose": "Human-readable skill guidance and privacy constraints.",
        },
        {
            "path": "metadata.json",
            "purpose": "Capability labels, phase marker, and non-execution safety metadata.",
        },
    ]
    managed_adapters = [
        {
            "kind": ADAPTER_LOCAL_FILE_IMPORT,
            "purpose": "Import a user-selected Google Takeout YouTube watch-history file.",
            "allowed_extensions": [".json", ".html"],
            "max_file_size_mb": 50,
            "path_policy": USER_SELECTED_FILE_ONLY,
            "stores_local_index": True,
            "network_allowed": False,
        }
    ]
    proposed_manifest = {
        "schema_version": 1,
        "kind": "skill",
        "id": "youtube-history-search",
        "title": title,
        "capabilities": [
            "youtube_history_search",
            "private_history_search",
            "google_takeout_import",
        ],
        "phase": "preview_only",
        "creates_files": False,
        "executes_code": False,
        "permissions_granted": [],
        "managed_adapters": managed_adapters,
    }
    permissions_requested = [
        {
            "permission": "local_file_read",
            "scope": "A user-selected Google Takeout YouTube watch-history export in a future phase.",
            "status": "deferred",
        },
        {
            "permission": "local_index_storage",
            "scope": "A local derived search index in a future phase.",
            "status": "deferred",
        },
    ]
    privacy_notes = [
        "This preview does not read YouTube history.",
        "The v1 design starts with a local Google Takeout import selected by the user.",
        "Raw history is not uploaded.",
        "Raw history rows, full URLs, account identifiers, and private search terms are excluded from logs and support context by default.",
    ]
    blocked_actions = [
        "No arbitrary executable pack code is created or run.",
        "No packs are approved, enabled, or executed by the scaffold flow.",
        "No OAuth or Google account connection in v1.",
        "No browser history scraping in v1.",
        "No transcript fetching or network lookup in v1.",
        "No video or audio downloads.",
    ]
    return {
        "type": "skill_scaffold_preview",
        "scaffold_id": _stable_scaffold_id("youtube_history_search", title),
        "capability": "youtube_history_search",
        "title": title,
        "summary": summary,
        "user_goal": str(user_goal or "").strip() or None,
        "files_to_create": files_to_create,
        "proposed_manifest": proposed_manifest,
        "proposed_skill_doc": proposed_skill_doc,
        "permissions_requested": permissions_requested,
        "privacy_notes": privacy_notes,
        "blocked_actions": blocked_actions,
        "next_step": "After preview, a later confirmation can create a text-only review draft. Permission only records the selected file choice in this phase.",
        "creates_files": False,
        "executes_code": False,
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug or "scaffold"


def create_generated_scaffold_source(preview: dict[str, Any], *, storage_root: str | Path) -> dict[str, Any]:
    if not isinstance(preview, dict):
        raise ValueError("scaffold preview is required")
    title = str(preview.get("title") or "Generated Skill Scaffold").strip()
    scaffold_id = str(preview.get("scaffold_id") or _stable_scaffold_id(str(preview.get("capability") or ""), title)).strip()
    root = Path(storage_root).expanduser().resolve()
    quarantine_root = root / "quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    source_dir = quarantine_root / f"generated-{_slugify(title)}-{int(time.time() * 1000)}"
    source_dir.mkdir(parents=False, exist_ok=False)

    manifest = preview.get("proposed_manifest") if isinstance(preview.get("proposed_manifest"), dict) else {}
    managed_adapters = manifest.get("managed_adapters") if isinstance(manifest.get("managed_adapters"), list) else []
    manifest_payload = {
        **manifest,
        "phase": "generated_review_candidate",
        "generated_from_scaffold_id": scaffold_id,
        "creates_files": False,
        "executes_code": False,
        "permissions_granted": [],
        "review_required": True,
    }
    metadata_payload = {
        "generated": True,
        "generated_from_scaffold_id": scaffold_id,
        "title": title,
        "capability": str(preview.get("capability") or "").strip() or None,
        "summary": str(preview.get("summary") or "").strip() or None,
        "phase": "generated_review_candidate",
        "text_only": True,
        "creates_files": False,
        "executes_code": False,
        "approved": False,
        "enabled": False,
        "permissions_granted": [],
        "managed_adapters": list(managed_adapters),
        "privacy_notes": list(preview.get("privacy_notes") if isinstance(preview.get("privacy_notes"), list) else []),
        "blocked_actions": list(preview.get("blocked_actions") if isinstance(preview.get("blocked_actions"), list) else []),
        "support_context_policy": "exclude_raw_history_rows_full_urls_account_identifiers_and_private_search_terms",
    }
    skill_doc = str(preview.get("proposed_skill_doc") or "").strip()
    if not skill_doc:
        skill_doc = f"# {title}\n\nPreview-only generated text skill scaffold.\n"
    (source_dir / "SKILL.md").write_text(skill_doc + "\n", encoding="utf-8")
    (source_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (source_dir / "metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "source_path": str(source_dir),
        "files_created": ["SKILL.md", "manifest.json", "metadata.json"],
        "creates_files": True,
        "executes_code": False,
        "approved": False,
        "enabled": False,
        "permissions_granted": [],
    }


def render_scaffold_offer(preview: dict[str, Any] | None) -> str:
    if not isinstance(preview, dict):
        return "That skill is not active yet. Say yes to preview a safe draft skill."
    title = str(preview.get("title") or "skill scaffold").strip()
    return (
        "I do not have private YouTube history access active yet, so I cannot read or search your history today. "
        "Browser automation would not safely solve private watch-history access. "
        f"I can preview a safe draft skill called {title}. Say yes to preview it."
    ).strip()


def render_scaffold_preview(preview: dict[str, Any] | None) -> str:
    if not isinstance(preview, dict):
        return "I cannot show that draft skill preview because the preview data is missing. No files were created."
    title = str(preview.get("title") or "Draft Skill").strip()
    files = preview.get("files_to_create") if isinstance(preview.get("files_to_create"), list) else []
    file_names = ", ".join(str(row.get("path") if isinstance(row, dict) else row).strip() for row in files if str(row.get("path") if isinstance(row, dict) else row).strip())
    return (
        f"I can create a draft skill for review: {title}. "
        "This will not read your history yet. It will not install, run code, or connect to Google. "
        "The current safe runtime can only validate a selected file grant or dry-run access; it cannot read, parse, index, or search the file contents yet. "
        f"If you continue, the review draft would include {file_names or 'SKILL.md and metadata.json'}. "
        "No files were created yet, and no code was executed. Say yes to create the review draft."
    ).strip()

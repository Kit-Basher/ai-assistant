from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.error_response_ux import compose_actionable_message
from agent.filesystem_skill import FileSystemSkill
from agent.packs.remote_fetch import RemoteFetchError, RemotePackFetcher, RemotePackSource

PACK_TYPE_SKILL = "skill"
PACK_TYPE_EXPERIENCE = "experience"
PACK_TYPE_NATIVE = "native"

CLASS_PORTABLE_TEXT_SKILL = "portable_text_skill"
CLASS_EXPERIENCE_PACK = "experience_pack"
CLASS_NATIVE_CODE_PACK = "native_code_pack"
CLASS_UNKNOWN_PACK = "unknown_pack"

STATUS_NORMALIZED = "normalized"
STATUS_REVIEW_NEEDED = "review_needed"
STATUS_BLOCKED = "blocked"
STATUS_PARTIAL_SAFE_IMPORT = "partial_safe_import"

SAFE_TEXT_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
}
STATIC_ASSET_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".css",
}
EXECUTABLE_FILE_NAMES = {
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "setup.py",
    "pyproject.toml",
    "requirements.txt",
    "cargo.toml",
    "cargo.lock",
    "plugin.json",
    "manifest.toml",
    "manifest.yaml",
    "manifest.yml",
    "go.mod",
}
EXECUTABLE_EXTENSIONS = {
    ".py",
    ".pyc",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".rb",
    ".php",
    ".pl",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".class",
    ".jar",
}
HIGH_RISK_PATH_PARTS = {
    ".git",
    ".github",
    ".vscode",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
}
PROMPT_INJECTION_PATTERNS = (
    r"ignore (all|any|the) previous instructions",
    r"ignore the system prompt",
    r"reveal (the )?(system|developer) prompt",
    r"exfiltrat(e|ion)",
    r"print .*secret",
    r"override .*policy",
    r"bypass .*safety",
)
INSTALL_PATTERNS = (
    r"\bpip(?:3)? install\b",
    r"\bnpm install\b",
    r"\bpnpm install\b",
    r"\byarn install\b",
    r"\bapt(?:-get)? install\b",
    r"\bbrew install\b",
    r"\bcargo install\b",
)
SHELL_REQUIREMENT_PATTERNS = (
    r"\bbash\b",
    r"\bsh\b",
    r"\bchmod \+x\b",
    r"\./[^\s`]+",
)
NETWORK_PATTERNS = (
    r"https?://",
    r"\bfetch\(",
    r"\brequests\.",
    r"\burllib\.",
    r"\bhttpx\.",
)
DOWNLOAD_EXECUTE_PATTERNS = (
    r"curl[^|\n]+[|][^\n]*(sh|bash)",
    r"wget[^|\n]+[|][^\n]*(sh|bash)",
    r"urlretrieve\(",
)
PRIVILEGE_PATTERNS = (
    r"\bsudo\b",
    r"\bsystemctl\b",
    r"/etc/",
    r"~/.ssh",
    r"~/.gnupg",
)
ENCODED_EXECUTION_PATTERNS = (
    r"[A-Za-z0-9+/]{200,}={0,2}",
    r"base64",
    r"eval\(",
)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slugify(value: str) -> str:
    lowered = str(value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "pack"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_text_file(path: Path, *, max_bytes: int = 256 * 1024) -> tuple[str | None, str | None]:
    try:
        raw = path.read_bytes()[:max_bytes]
    except OSError:
        return None, None
    return FileSystemSkill._decode_text(raw)


def _parse_simple_metadata_text(text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if not normalized_key:
            continue
        if normalized_value.startswith("[") and normalized_value.endswith("]"):
            items = [
                item.strip().strip("'\"")
                for item in normalized_value[1:-1].split(",")
                if item.strip().strip("'\"")
            ]
            meta[normalized_key] = items
            continue
        if "," in normalized_value and normalized_key in {"capabilities", "permissions", "tags"}:
            meta[normalized_key] = [
                item.strip().strip("'\"")
                for item in normalized_value.split(",")
                if item.strip().strip("'\"")
            ]
            continue
        meta[normalized_key] = normalized_value.strip("'\"")
    return meta


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    frontmatter = text[4:end]
    body = text[end + 4 :]
    return _parse_simple_metadata_text(frontmatter), body.lstrip("\n")


def _infer_first_heading(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip() or None
    return None


def _normalize_source_key(
    *,
    source_origin: str,
    source_url: str | None,
    source_path: str | None,
) -> str:
    payload = {
        "origin": str(source_origin or "").strip(),
        "url": str(source_url or "").strip() or None,
        "source_path": str(source_path or "").strip() or None,
    }
    return _sha256_text(_safe_json(payload))


def _normalize_source_fingerprint(
    *,
    source_url: str | None,
    resolved_commit: str | None,
    archive_sha256: str | None,
    source_path: str | None,
) -> str:
    payload = {
        "url": str(source_url or "").strip() or None,
        "resolved_commit": str(resolved_commit or "").strip() or None,
        "archive_sha256": str(archive_sha256 or "").strip() or None,
        "source_path": str(source_path or "").strip() or None,
    }
    return _sha256_text(_safe_json(payload))


def _component_identity_payload(
    components: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    *,
    classification: str,
) -> dict[str, Any]:
    normalized_components = [
        {
            "path": str(item.get("path") or ""),
            "component_type": str(item.get("component_type") or ""),
            "included": bool(item.get("included", False)),
            "executable": bool(item.get("executable", False)),
            "sha256": str(item.get("sha256") or "").strip() or None,
        }
        for item in components
    ]
    normalized_assets = [
        {
            "path": str(item.get("path") or ""),
            "asset_type": str(item.get("asset_type") or ""),
            "included": bool(item.get("included", False)),
            "executable": bool(item.get("executable", False)),
            "sha256": str(item.get("sha256") or "").strip() or None,
        }
        for item in assets
    ]
    return {
        "classification": str(classification or ""),
        "components": sorted(normalized_components, key=lambda item: (item["path"], item["component_type"])),
        "assets": sorted(normalized_assets, key=lambda item: (item["path"], item["asset_type"])),
    }


def _canonical_outcome_category(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == STATUS_NORMALIZED:
        return "normalized_safe_text"
    if normalized == STATUS_PARTIAL_SAFE_IMPORT:
        return "partial_safe_import"
    return "blocked_install"


def _squash_blank_lines(text: str) -> str:
    lines = [line.rstrip() for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    squashed: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank:
            if previous_blank:
                continue
            squashed.append("")
            previous_blank = True
            continue
        squashed.append(line)
        previous_blank = False
    return "\n".join(squashed).strip()


def _text_excerpt(text: str, *, max_chars: int = 1200) -> str:
    cleaned = _squash_blank_lines(text)
    if len(cleaned) <= max_chars:
        return cleaned
    excerpt = cleaned[:max_chars].rstrip()
    cut = excerpt.rfind("\n\n")
    if cut > 400:
        excerpt = excerpt[:cut].rstrip()
    return excerpt + "\n\n..."


def _is_example_source(rel_path: str, text: str) -> bool:
    lowered_path = str(rel_path or "").lower()
    lowered_text = str(text or "").lower()
    if any(token in lowered_path for token in ("example", "examples", "sample", "demo")):
        return True
    if "```" in lowered_text:
        return True
    if re.search(r"(?mi)^\s*(example|examples|sample|prompt examples?)\s*[:\-]", lowered_text):
        return True
    return False


def _looks_like_strong_skill_markdown(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = (
        "purpose",
        "when to use",
        "inputs",
        "behavior",
        "constraints",
        "response style",
        "example prompts",
    )
    score = 0
    for marker in markers:
        if re.search(rf"(?mi)^#{1,3}\s+{re.escape(marker)}\b", lowered):
            score += 1
    return score >= 4


def _default_skill_section(title: str, body: str) -> str:
    return f"## {title}\n\n{_squash_blank_lines(body).strip()}\n"


def _path_is_metadata(rel_path: str) -> bool:
    lowered = str(rel_path or "").lower()
    return lowered in {"metadata.json", "metadata.yaml", "metadata.yml"}


def _has_prompt_like_text_layout(file_records: list["PackFileRecord"]) -> bool:
    strong_tokens = (
        "prompt",
        "instruction",
        "instructions",
        "example",
        "examples",
        "guide",
        "workflow",
        "task",
    )
    strong_names = {
        "prompt.md",
        "prompts.md",
        "instruction.md",
        "instructions.md",
        "example.md",
        "examples.md",
        "guide.md",
        "workflow.md",
        "task.md",
    }
    for record in file_records:
        rel_lower = str(record.rel_path or "").lower()
        name_lower = Path(rel_lower).name.lower()
        if name_lower in strong_names:
            return True
        if any(token in rel_lower for token in strong_tokens):
            return True
    return False


@dataclass(frozen=True)
class PackComponent:
    path: str
    component_type: str
    included: bool
    executable: bool = False
    sha256: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PackRiskReport:
    score: float
    level: str
    flags: tuple[str, ...] = ()
    hard_block_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 4)
        return payload


@dataclass(frozen=True)
class CanonicalPack:
    id: str
    name: str
    version: str
    type: str
    pack_identity: dict[str, Any]
    source: dict[str, Any]
    integrity: dict[str, Any]
    trust: dict[str, Any]
    trust_anchor: dict[str, Any]
    capabilities: dict[str, Any]
    permissions: dict[str, Any]
    components: tuple[dict[str, Any], ...]
    assets: tuple[dict[str, Any], ...]
    source_history: tuple[dict[str, Any], ...]
    versions: tuple[dict[str, Any], ...]
    runtime: dict[str, Any]
    adaptation: dict[str, Any]
    audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PackNormalizationResult:
    classification: str
    status: str
    pack: CanonicalPack
    risk_report: PackRiskReport
    blocked_reasons: tuple[str, ...] = ()
    stripped_components: tuple[str, ...] = ()
    quarantine_path: str | None = None
    normalized_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "status": self.status,
            "pack": self.pack.to_dict(),
            "risk_report": self.risk_report.to_dict(),
            "blocked_reasons": list(self.blocked_reasons),
            "stripped_components": list(self.stripped_components),
            "quarantine_path": self.quarantine_path,
            "normalized_path": self.normalized_path,
        }


@dataclass(frozen=True)
class PackUserDecisionEnvelope:
    pack_name: str
    appears_to_do: str
    found_inside: tuple[str, ...]
    risk_level: str
    why_risk: tuple[str, ...]
    safe_options: tuple[str, ...]
    review_required: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_name": self.pack_name,
            "appears_to_do": self.appears_to_do,
            "found_inside": list(self.found_inside),
            "risk_level": self.risk_level,
            "why_risk": list(self.why_risk),
            "safe_options": list(self.safe_options),
            "review_required": bool(self.review_required),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class PackFileRecord:
    rel_path: str
    absolute_path: str
    file_type: str
    size: int
    sha256: str | None = None
    is_text: bool = False
    text_preview: str | None = None


class ExternalPackIngestor:
    def __init__(self, storage_root: str, *, remote_fetcher: RemotePackFetcher | None = None) -> None:
        self.storage_root = Path(storage_root).expanduser().resolve()
        self.quarantine_root = self.storage_root / "quarantine"
        self.normalized_root = self.storage_root / "normalized"
        self.quarantine_root.mkdir(parents=True, exist_ok=True)
        self.normalized_root.mkdir(parents=True, exist_ok=True)
        self._remote_fetcher = remote_fetcher

    def ingest_from_path(
        self,
        source_path: str,
        *,
        source_origin: str = "local_path",
        source_url: str | None = None,
        commit_hash: str | None = None,
        created_by: str = "packs_install",
    ) -> tuple[PackNormalizationResult, PackUserDecisionEnvelope]:
        source = Path(source_path).expanduser().resolve()
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"pack source not found: {source}")
        quarantined_path, file_records, integrity = self._quarantine_source(source)
        return self._finalize_ingestion(
            snapshot_path=quarantined_path,
            quarantine_path=str(quarantined_path),
            file_records=file_records,
            integrity=integrity,
            source_origin=source_origin,
            source_url=source_url,
            commit_hash=commit_hash,
            created_by=created_by,
        )

    def ingest_from_remote_source(
        self,
        source: RemotePackSource,
        *,
        created_by: str = "packs_install",
    ) -> tuple[PackNormalizationResult, PackUserDecisionEnvelope]:
        fetcher = self._remote_fetcher or RemotePackFetcher(str(self.storage_root))
        try:
            fetched = fetcher.fetch(source)
        except RemoteFetchError as exc:
            blocked = self._build_fetch_blocked_result(exc, created_by=created_by)
            return blocked, self._build_user_envelope(blocked)
        file_records = self._inventory(Path(fetched.snapshot_path))
        integrity = self._aggregate_integrity(file_records, archive_sha256=fetched.source.archive_sha256)
        integrity["raw_archive_path"] = fetched.raw_archive_path
        return self._finalize_ingestion(
            snapshot_path=Path(fetched.snapshot_path),
            quarantine_path=fetched.quarantine_path,
            file_records=file_records,
            integrity=integrity,
            source_origin=fetched.source.kind,
            source_url=fetched.source.url,
            commit_hash=fetched.source.commit_hash_resolved,
            created_by=created_by,
            remote_source=fetched.source,
            raw_archive_path=fetched.raw_archive_path,
        )

    def _finalize_ingestion(
        self,
        *,
        snapshot_path: Path,
        quarantine_path: str,
        file_records: list[PackFileRecord],
        integrity: dict[str, Any],
        source_origin: str,
        source_url: str | None,
        commit_hash: str | None,
        created_by: str,
        remote_source: RemotePackSource | None = None,
        raw_archive_path: str | None = None,
    ) -> tuple[PackNormalizationResult, PackUserDecisionEnvelope]:
        classification = self._classify(file_records)
        risk_report = self._scan_risk(file_records, classification, remote_source=remote_source)
        normalized = self._normalize(
            quarantined_path=snapshot_path,
            file_records=file_records,
            classification=classification,
            risk_report=risk_report,
            integrity=integrity,
            source_origin=source_origin,
            source_url=source_url,
            commit_hash=commit_hash,
            created_by=created_by,
            remote_source=remote_source,
            raw_archive_path=raw_archive_path,
        )
        normalized = PackNormalizationResult(
            classification=normalized.classification,
            status=normalized.status,
            pack=normalized.pack,
            risk_report=normalized.risk_report,
            blocked_reasons=normalized.blocked_reasons,
            stripped_components=normalized.stripped_components,
            quarantine_path=quarantine_path,
            normalized_path=normalized.normalized_path,
        )
        envelope = self._build_user_envelope(normalized)
        return normalized, envelope

    def _quarantine_source(self, source: Path) -> tuple[Path, list[PackFileRecord], dict[str, Any]]:
        suffix = f"{_slugify(source.name)}-{int(time.time() * 1000)}"
        quarantine_dir = self.quarantine_root / suffix
        shutil.copytree(source, quarantine_dir, symlinks=True)
        records = self._inventory(quarantine_dir)
        integrity = self._aggregate_integrity(records)
        return quarantine_dir, records, integrity

    @staticmethod
    def _aggregate_integrity(file_records: list[PackFileRecord], *, archive_sha256: str | None = None) -> dict[str, Any]:
        aggregate = hashlib.sha256()
        files_hashed = 0
        for record in sorted(file_records, key=lambda item: item.rel_path):
            if record.sha256:
                aggregate.update(f"{record.rel_path}:{record.sha256}\n".encode("utf-8"))
                files_hashed += 1
        integrity = {
            "sha256": aggregate.hexdigest(),
            "files_hashed": files_hashed,
            "signature_verified": False,
        }
        if archive_sha256:
            integrity["archive_sha256"] = str(archive_sha256)
        return integrity

    @staticmethod
    def _canonical_source_snapshot_path(rel_path: str) -> str:
        cleaned = str(rel_path or "").strip().lstrip("/")
        return str(Path("assets") / "source" / cleaned) if cleaned else "assets/source"

    @staticmethod
    def _canonical_text_source_path(rel_path: str) -> str:
        cleaned = str(rel_path or "").strip().lstrip("/")
        return str(Path("assets") / "source" / cleaned) if cleaned else "assets/source"

    @staticmethod
    def _canonical_asset_target_path(rel_path: str) -> str:
        cleaned = str(rel_path or "").strip().lstrip("/")
        if cleaned.lower().startswith("assets/"):
            cleaned = cleaned[7:]
        return str(Path("assets") / cleaned) if cleaned else "assets"

    def _canonical_source_docs(
        self,
        file_records: list[PackFileRecord],
    ) -> dict[str, Any]:
        main_sections: list[dict[str, str]] = []
        example_sections: list[dict[str, str]] = []
        kept_snapshot_paths: list[str] = []
        dropped_paths: list[str] = []
        blocked_categories: set[str] = set()
        text_sources: list[str] = []
        for record in sorted(file_records, key=lambda item: item.rel_path):
            rel_path = str(record.rel_path or "").strip()
            lower_path = rel_path.lower()
            suffix = Path(rel_path).suffix.lower()
            name_lower = Path(rel_path).name.lower()
            if _path_is_metadata(rel_path):
                continue
            if lower_path == "skill.md":
                if record.is_text:
                    full_text, _ = _load_text_file(Path(record.absolute_path), max_bytes=1024 * 1024)
                    parsed_meta, parsed_body = _parse_frontmatter(full_text or record.text_preview or "")
                    text_body = parsed_body.strip() or str(record.text_preview or "").strip()
                    if text_body:
                        text_sources.append(text_body)
                        main_sections.append(
                            {
                                "path": "SKILL.md source",
                                "text": _text_excerpt(text_body, max_chars=1800),
                            }
                        )
                        if isinstance(parsed_meta, dict) and parsed_meta:
                            meta_lines = [f"- {key}: {value}" for key, value in sorted(parsed_meta.items()) if value not in (None, "", [], {})]
                            if meta_lines:
                                main_sections.append(
                                    {
                                        "path": "SKILL.md frontmatter",
                                        "text": "\n".join(meta_lines),
                                    }
                                )
                continue
            if name_lower in EXECUTABLE_FILE_NAMES or suffix in EXECUTABLE_EXTENSIONS:
                dropped_paths.append(rel_path)
                blocked_categories.add("executable")
                continue
            if record.file_type == "symlink":
                dropped_paths.append(rel_path)
                blocked_categories.add("symlink")
                continue
            if not record.is_text and record.file_type == "file" and suffix not in STATIC_ASSET_EXTENSIONS:
                dropped_paths.append(rel_path)
                blocked_categories.add("binary_or_unknown")
                continue
            if not record.text_preview and not (suffix in STATIC_ASSET_EXTENSIONS or record.file_type == "file"):
                dropped_paths.append(rel_path)
                blocked_categories.add("unsupported")
                continue
            source_path = self._canonical_source_snapshot_path(rel_path)
            kept_snapshot_paths.append(source_path)
            if record.is_text:
                full_text, _ = _load_text_file(Path(record.absolute_path), max_bytes=1024 * 1024)
                parsed_meta, parsed_body = _parse_frontmatter(full_text or record.text_preview or "")
                text_body = parsed_body.strip() or str(record.text_preview or "").strip()
                if not text_body:
                    continue
                text_sources.append(text_body)
                destination = example_sections if _is_example_source(rel_path, text_body) else main_sections
                destination.append(
                    {
                        "path": rel_path,
                        "text": _text_excerpt(text_body, max_chars=1800),
                    }
                )
                if isinstance(parsed_meta, dict) and parsed_meta:
                    meta_lines = [f"- {key}: {value}" for key, value in sorted(parsed_meta.items()) if value not in (None, "", [], {})]
                    if meta_lines:
                        main_sections.append(
                            {
                                "path": f"{rel_path} frontmatter",
                                "text": "\n".join(meta_lines),
                            }
                        )
            else:
                kept_snapshot_paths.append(source_path)
        # collapse duplicates while preserving order
        def _dedupe(items: list[dict[str, str]]) -> list[dict[str, str]]:
            seen: set[str] = set()
            out: list[dict[str, str]] = []
            for item in items:
                key = _squash_blank_lines(f"{item.get('path')}\n{item.get('text')}")
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
            return out

        return {
            "main_sections": _dedupe(main_sections),
            "example_sections": _dedupe(example_sections),
            "kept_snapshot_paths": list(dict.fromkeys(kept_snapshot_paths)),
            "dropped_paths": list(dict.fromkeys(dropped_paths)),
            "blocked_categories": sorted(blocked_categories),
            "text_sources": text_sources,
        }

    def _build_canonical_skill_markdown(
        self,
        *,
        display_name: str,
        summary: str,
        skill_body: str,
        main_sections: list[dict[str, str]],
        example_sections: list[dict[str, str]],
        capabilities: dict[str, Any],
    ) -> tuple[str, bool]:
        normalized_skill_body = _squash_blank_lines(skill_body)
        if normalized_skill_body and _looks_like_strong_skill_markdown(normalized_skill_body):
            return normalized_skill_body + "\n", True

        main_text = "\n\n".join(
            f"### {section['path']}\n\n{section['text']}".strip()
            for section in main_sections[:8]
            if section.get("text")
        ).strip()
        example_text = "\n\n".join(
            f"### {section['path']}\n\n{section['text']}".strip()
            for section in example_sections[:6]
            if section.get("text")
        ).strip()
        declared_capabilities = ", ".join(
            str(item).strip()
            for item in (
                capabilities.get("declared") if isinstance(capabilities.get("declared"), list) else []
            )
            if str(item).strip()
        )
        inferred_capabilities = ", ".join(
            str(item).strip()
            for item in (
                capabilities.get("inferred") if isinstance(capabilities.get("inferred"), list) else []
            )
            if str(item).strip()
        )
        sections = [
            f"# {display_name}",
            _default_skill_section("Purpose", summary or f"{display_name} is a safe normalized pack."),
            _default_skill_section(
                "When to use",
                "Use this pack when the user asks for the task or guidance this pack describes, and stay within the safe text and asset content that was imported.",
            ),
            _default_skill_section(
                "Inputs",
                "User prompts, the safe instructions in this pack, and any preserved reference materials or assets.",
            ),
            _default_skill_section(
                "Behavior",
                main_text
                or "Use the preserved source instructions and the safe reference materials in this pack.",
            ),
            _default_skill_section(
                "Constraints",
                "Do not execute imported code or shell instructions. Use only safe text and assets. Ignore anything that was dropped during normalization.",
            ),
            _default_skill_section(
                "Response style",
                "Be concise, grounded, and explicit about uncertainty.",
            ),
            _default_skill_section(
                "Example prompts",
                example_text or "No explicit examples were found in the imported source.",
            ),
        ]
        if declared_capabilities or inferred_capabilities:
            sections.append(
                _default_skill_section(
                    "Capabilities",
                    f"Declared: {declared_capabilities or 'none'}\nInferred: {inferred_capabilities or 'none'}",
                )
            )
        return "\n\n".join(section.rstrip() for section in sections if section.strip()) + "\n", False

    def _build_prompts_markdown(
        self,
        *,
        display_name: str,
        summary: str,
        skill_body: str,
        main_sections: list[dict[str, str]],
        example_sections: list[dict[str, str]],
    ) -> tuple[str, str]:
        normalized_skill_body = _squash_blank_lines(skill_body)
        main_chunks: list[str] = []
        if normalized_skill_body:
            main_chunks.append(f"## Primary skill text\n\n{normalized_skill_body}")
        for section in main_sections[:12]:
            text = str(section.get("text") or "").strip()
            if not text:
                continue
            title = str(section.get("path") or "source").strip()
            main_chunks.append(f"## {title}\n\n{text}")
        if not main_chunks:
            main_chunks.append(summary or f"{display_name} has no extracted text material.")
        main_text = f"# {display_name} prompt material\n\n" + "\n\n".join(main_chunks).strip() + "\n"

        example_chunks: list[str] = []
        for section in example_sections[:10]:
            text = str(section.get("text") or "").strip()
            if not text:
                continue
            title = str(section.get("path") or "example").strip()
            example_chunks.append(f"## {title}\n\n{text}")
        if not example_chunks:
            example_chunks.append("No explicit example prompts were found in the imported source.")
        example_text = "# Example prompts\n\n" + "\n\n".join(example_chunks).strip() + "\n"
        return main_text, example_text

    def _build_pack_markdown(
        self,
        *,
        display_name: str,
        summary: str,
        canonical_id: str,
        version: str,
        classification: str,
        status: str,
        risk_report: PackRiskReport,
        kept_snapshot_paths: list[str],
        dropped_paths: list[str],
        source_origin: str,
        source_url: str | None,
        source_ref: str | None,
    ) -> str:
        state_label = _canonical_outcome_category(status).replace("_", " ")
        kept_preview = ", ".join(kept_snapshot_paths[:6]) if kept_snapshot_paths else "none"
        dropped_preview = ", ".join(dropped_paths[:6]) if dropped_paths else "none"
        lines = [
            f"# {display_name}",
            "",
            summary or f"{display_name} normalized as a safe pack.",
            "",
            "## Pack details",
            f"- Pack ID: `{canonical_id}`",
            f"- Version: `{version}`",
            f"- Classification: `{classification}`",
            f"- Outcome: `{state_label}`",
            f"- Risk level: `{risk_report.level}`",
            f"- Risk score: `{float(risk_report.score):.4f}`",
            "",
            "## Import notes",
            f"- Source origin: `{source_origin}`",
            f"- Source URL: `{source_url or 'n/a'}`",
            f"- Source ref: `{source_ref or 'n/a'}`",
            f"- Kept snapshot files: {kept_preview}",
            f"- Dropped files: {dropped_preview}",
            "",
            "## Next step",
        ]
        if status == STATUS_NORMALIZED:
            lines.append("Review the canonical `SKILL.md` and `prompts/` files before using the pack.")
        elif status == STATUS_PARTIAL_SAFE_IMPORT:
            lines.append("Use the safe text and assets only, and treat the dropped files as unavailable.")
        else:
            lines.append("Keep the audit snapshot only, or rebuild a safe text-only pack manually.")
        return "\n".join(lines).strip() + "\n"

    def _write_canonical_pack_tree(
        self,
        root: Path,
        *,
        display_name: str,
        summary: str,
        canonical_id: str,
        version: str,
        upstream_version: str | None,
        classification: str,
        pack_type: str,
        status: str,
        risk_report: PackRiskReport,
        source_origin: str,
        source_url: str | None,
        source_ref: str | None,
        source_url_resolved: str | None,
        source_commit: str | None,
        source_key: str,
        source_fingerprint: str,
        quarantine_path: str | None,
        source_history: list[dict[str, Any]],
        file_records: list[PackFileRecord],
        components: list[dict[str, Any]],
        assets: list[dict[str, Any]],
        capabilities: dict[str, Any],
        permissions_granted: list[str],
        main_sections: list[dict[str, str]],
        example_sections: list[dict[str, str]],
        kept_snapshot_paths: list[str],
        dropped_paths: list[str],
        blocked_categories: list[str],
        normalized_integrity: dict[str, Any],
        content_hash: str,
        raw_archive_path: str | None,
    ) -> dict[str, Any]:
        root.mkdir(parents=True, exist_ok=True)
        (root / "prompts").mkdir(parents=True, exist_ok=True)
        (root / "assets" / "source").mkdir(parents=True, exist_ok=True)
        (root / "metadata").mkdir(parents=True, exist_ok=True)
        source_snapshot_targets: list[str] = list(dict.fromkeys(kept_snapshot_paths))
        normalization_decisions: list[dict[str, Any]] = []
        for record in sorted(file_records, key=lambda item: item.rel_path):
            rel_path = str(record.rel_path or "").strip()
            if not rel_path:
                continue
            source_target_rel = self._canonical_source_snapshot_path(rel_path)
            source_target_path = root / source_target_rel
            source_target_path.parent.mkdir(parents=True, exist_ok=True)
            if record.file_type == "symlink":
                normalization_decisions.append(
                    {
                        "source_path": rel_path,
                        "action": "dropped",
                        "target_path": None,
                        "reason": "symlink",
                    }
                )
                continue
            if record.is_text or Path(rel_path).suffix.lower() in STATIC_ASSET_EXTENSIONS:
                try:
                    shutil.copy2(record.absolute_path, source_target_path)
                    source_snapshot_targets.append(source_target_rel)
                    if Path(rel_path).suffix.lower() in STATIC_ASSET_EXTENSIONS:
                        asset_target_rel = self._canonical_asset_target_path(rel_path)
                        asset_target_path = root / asset_target_rel
                        asset_target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(record.absolute_path, asset_target_path)
                    normalization_decisions.append(
                        {
                            "source_path": rel_path,
                            "action": "kept",
                            "target_path": source_target_rel,
                            "reason": "safe_text" if record.is_text else "static_asset",
                        }
                    )
                except OSError:
                    normalization_decisions.append(
                        {
                            "source_path": rel_path,
                            "action": "dropped",
                            "target_path": None,
                            "reason": "copy_failed",
                        }
                    )
                continue
            if rel_path.lower() not in {"skill.md"}:
                normalization_decisions.append(
                    {
                        "source_path": rel_path,
                        "action": "dropped",
                        "target_path": None,
                        "reason": "unsupported_or_executable",
                    }
                )
        skill_source_body = self._extract_primary_skill_body(file_records)
        skill_body, preserved_skill = self._build_canonical_skill_markdown(
            display_name=display_name,
            summary=summary,
            skill_body=skill_source_body,
            main_sections=main_sections,
            example_sections=example_sections,
            capabilities=capabilities,
        )
        main_markdown, examples_markdown = self._build_prompts_markdown(
            display_name=display_name,
            summary=summary,
            skill_body=skill_body,
            main_sections=main_sections,
            example_sections=example_sections,
        )
        canonical_skill_path = root / "SKILL.md"
        canonical_skill_path.write_text(skill_body, encoding="utf-8")
        (root / "prompts" / "main.md").write_text(main_markdown, encoding="utf-8")
        (root / "prompts" / "examples.md").write_text(examples_markdown, encoding="utf-8")
        pack_md = self._build_pack_markdown(
            display_name=display_name,
            summary=summary,
            canonical_id=canonical_id,
            version=version,
            classification=classification,
            status=status,
            risk_report=risk_report,
            kept_snapshot_paths=source_snapshot_targets,
            dropped_paths=list(dropped_paths),
            source_origin=source_origin,
            source_url=source_url,
            source_ref=source_ref,
        )
        (root / "PACK.md").write_text(pack_md, encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "pack_id": canonical_id,
            "version": version,
            "display_name": display_name,
            "summary": summary,
            "kind": pack_type,
            "classification": classification,
            "import_mode": status,
            "source": {
                "origin": source_origin,
                "url": source_url,
                "resolved_url": source_url_resolved,
                "ref": source_ref,
                "commit_hash": source_commit,
                "upstream_version": upstream_version,
                "source_key": source_key,
                "source_fingerprint": source_fingerprint,
                "archive_sha256": normalized_integrity.get("archive_sha256"),
                "raw_archive_path": raw_archive_path,
                "quarantine_path": quarantine_path,
                "normalized_path": str(root),
                "fetched_at": _now_iso(),
            },
            "trust": {
                "level": "review_required" if status != STATUS_BLOCKED else "blocked",
                "import_mode": status,
                "risk_level": risk_report.level,
                "risk_score": round(float(risk_report.score), 4),
                "flags": list(risk_report.flags),
                "blocked_reasons": list(risk_report.hard_block_reasons),
                "review_required": True,
            },
            "permissions_granted": list(permissions_granted),
            "contents": {
                "pack_md": "PACK.md",
                "skill_md": "SKILL.md",
                "prompts": ["prompts/main.md", "prompts/examples.md"],
                "source_snapshot": list(dict.fromkeys(source_snapshot_targets)),
                "kept_files": list(
                    dict.fromkeys(
                        [item["path"] for item in components if bool(item.get("included", False))]
                        + [item["path"] for item in assets if bool(item.get("included", False))]
                        + source_snapshot_targets
                    )
                ),
                "dropped_files": list(dict.fromkeys(dropped_paths)),
            },
            "capabilities": {
                "declared": list(capabilities.get("declared") if isinstance(capabilities.get("declared"), list) else []),
                "inferred": list(capabilities.get("inferred") if isinstance(capabilities.get("inferred"), list) else []),
                "summary": str(capabilities.get("summary") or "").strip() or summary,
            },
        }
        (root / "manifest.json").write_text(_safe_json(manifest) + "\n", encoding="utf-8")

        source_payload = {
            "schema_version": 1,
            "pack_id": canonical_id,
            "version": version,
            "source": {
                "origin": source_origin,
                "url": source_url,
                "resolved_url": source_url_resolved,
                "ref": source_ref,
                "commit_hash": source_commit,
                "upstream_version": upstream_version,
                "source_key": source_key,
                "source_fingerprint": source_fingerprint,
                "archive_sha256": normalized_integrity.get("archive_sha256"),
                "raw_archive_path": raw_archive_path,
                "quarantine_path": quarantine_path,
                "normalized_path": str(root),
                "fetched_at": _now_iso(),
            },
            "source_history": source_history,
        }
        normalization_payload = {
            "schema_version": 1,
            "pack_id": canonical_id,
            "version": version,
            "kind": pack_type,
            "files_seen": [record.rel_path for record in sorted(file_records, key=lambda item: item.rel_path)],
            "files_kept": list(
                dict.fromkeys(
                    [item["path"] for item in components if bool(item.get("included", False))]
                    + [item["path"] for item in assets if bool(item.get("included", False))]
                )
            ),
            "files_dropped": list(dict.fromkeys(dropped_paths)),
            "blocked_categories": list(blocked_categories),
            "normalization_decisions": normalization_decisions,
            "canonical_files": [
                "PACK.md",
                "manifest.json",
                "SKILL.md",
                "prompts/main.md",
                "prompts/examples.md",
                "metadata/source.json",
                "metadata/normalization.json",
                "metadata/review.json",
            ],
        }
        review_payload = {
            "schema_version": 1,
            "pack_id": canonical_id,
            "version": version,
            "kind": pack_type,
            "outcome_category": _canonical_outcome_category(status),
            "review_required": True,
            "warnings": list(risk_report.flags) + list(risk_report.hard_block_reasons),
            "next_step": (
                "Review the canonical `SKILL.md` and `prompts/` files before using the pack."
                if status == STATUS_NORMALIZED
                else "Use the safe text and assets only, and treat dropped files as unavailable."
                if status == STATUS_PARTIAL_SAFE_IMPORT
                else "Keep the audit snapshot only, or rebuild a safe text-only pack manually."
            ),
            "confidence": round(max(0.0, 1.0 - float(risk_report.score)), 4),
            "risk_level": risk_report.level,
            "risk_score": round(float(risk_report.score), 4),
            "summary": (
                "Normalized safe text import."
                if status == STATUS_NORMALIZED
                else "Partial safe import."
                if status == STATUS_PARTIAL_SAFE_IMPORT
                else "Blocked install."
            ),
        }
        (root / "metadata" / "source.json").write_text(_safe_json(source_payload) + "\n", encoding="utf-8")
        (root / "metadata" / "normalization.json").write_text(_safe_json(normalization_payload) + "\n", encoding="utf-8")
        (root / "metadata" / "review.json").write_text(_safe_json(review_payload) + "\n", encoding="utf-8")
        return {
            "manifest": manifest,
            "source": source_payload,
            "normalization": normalization_payload,
            "review": review_payload,
        }

    @staticmethod
    def _extract_primary_skill_body(file_records: list[PackFileRecord]) -> str:
        skill_record = next((record for record in file_records if record.rel_path.lower() == "skill.md"), None)
        if skill_record and skill_record.text_preview is not None:
            full_text, _ = _load_text_file(Path(skill_record.absolute_path), max_bytes=1024 * 1024)
            parsed_meta, parsed_body = _parse_frontmatter(full_text or skill_record.text_preview)
            if parsed_body.strip():
                return parsed_body.strip()
            if skill_record.text_preview:
                return skill_record.text_preview.strip()
        return ""

    def _inventory(self, root: Path) -> list[PackFileRecord]:
        out: list[PackFileRecord] = []
        for current_root, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            current = Path(current_root)
            dirnames[:] = sorted(dirnames)
            filenames = sorted(filenames)
            for name in filenames:
                path = current / name
                rel_path = str(path.relative_to(root))
                try:
                    stat_result = path.lstat()
                except OSError:
                    continue
                if path.is_symlink():
                    target = os.readlink(path)
                    sha256 = hashlib.sha256(f"symlink:{target}".encode("utf-8")).hexdigest()
                    out.append(
                        PackFileRecord(
                            rel_path=rel_path,
                            absolute_path=str(path),
                            file_type="symlink",
                            size=len(target),
                            sha256=sha256,
                            is_text=False,
                            text_preview=None,
                        )
                    )
                    continue
                if not path.is_file():
                    out.append(
                        PackFileRecord(
                            rel_path=rel_path,
                            absolute_path=str(path),
                            file_type="other",
                            size=int(stat_result.st_size),
                        )
                    )
                    continue
                sha256 = hashlib.sha256()
                try:
                    with open(path, "rb") as handle:
                        while True:
                            chunk = handle.read(1024 * 64)
                            if not chunk:
                                break
                            sha256.update(chunk)
                except OSError:
                    continue
                text_preview, _ = _load_text_file(path)
                out.append(
                    PackFileRecord(
                        rel_path=rel_path,
                        absolute_path=str(path),
                        file_type="file",
                        size=int(stat_result.st_size),
                        sha256=sha256.hexdigest(),
                        is_text=text_preview is not None,
                        text_preview=text_preview[:4000] if text_preview else None,
                    )
                )
        return out

    @staticmethod
    def _classify(file_records: list[PackFileRecord]) -> str:
        rel_paths = {record.rel_path.lower() for record in file_records}
        has_skill_md = "skill.md" in rel_paths
        has_package_manifest = any(Path(path).name in EXECUTABLE_FILE_NAMES for path in rel_paths)
        has_executable_code = any(Path(record.rel_path).suffix.lower() in EXECUTABLE_EXTENSIONS for record in file_records)
        has_binary = any((not record.is_text and record.file_type == "file" and Path(record.rel_path).suffix.lower() not in STATIC_ASSET_EXTENSIONS) for record in file_records)
        has_assets = any(part == "assets" or part.startswith("assets/") for part in rel_paths)
        has_experience_signals = has_assets or any(
            Path(path).name in {"character.json", "persona.json", "experience.json"}
            for path in rel_paths
        )
        if has_package_manifest or has_binary or (has_executable_code and not has_skill_md):
            return CLASS_NATIVE_CODE_PACK
        if has_skill_md or _has_prompt_like_text_layout(file_records):
            return CLASS_PORTABLE_TEXT_SKILL
        if has_experience_signals:
            return CLASS_EXPERIENCE_PACK
        return CLASS_UNKNOWN_PACK

    def _scan_risk(
        self,
        file_records: list[PackFileRecord],
        classification: str,
        *,
        remote_source: RemotePackSource | None = None,
    ) -> PackRiskReport:
        score = 0.15
        flags: set[str] = set()
        hard_block_reasons: set[str] = set()
        rel_paths = {record.rel_path.lower() for record in file_records}
        if remote_source is not None:
            flags.add("archive_fetch")
            score += 0.05
            if remote_source.ref and not str(remote_source.commit_hash_resolved or "").strip():
                flags.add("remote_unpinned_ref")
                score += 0.08
            if not str(remote_source.top_level_dir_name or "").strip():
                flags.add("unknown_archive_layout")
                score += 0.05
        if classification == CLASS_NATIVE_CODE_PACK:
            flags.add("contains_executable_runtime")
            hard_block_reasons.add("native_code_pack_requires_execution")
            score += 0.45
        elif classification == CLASS_UNKNOWN_PACK:
            flags.add("unknown_format")
            hard_block_reasons.add("unknown_pack_requires_manual_review")
            score += 0.35
        elif classification == CLASS_EXPERIENCE_PACK:
            flags.add("experience_pack_not_supported_yet")
            score += 0.2

        if "skill.md" in rel_paths and all(
            Path(record.rel_path).suffix.lower() in SAFE_TEXT_EXTENSIONS | STATIC_ASSET_EXTENSIONS
            or Path(record.rel_path).name in {"skill.md", "agents.md"}
            or record.rel_path.startswith("references/")
            or record.rel_path.startswith("assets/")
            for record in file_records
            if record.file_type == "file"
        ):
            score -= 0.15
            flags.add("pure_text_skill_layout")

        if not any(Path(record.rel_path).suffix.lower() in EXECUTABLE_EXTENSIONS for record in file_records):
            score -= 0.05
        for record in file_records:
            rel_lower = record.rel_path.lower()
            suffix = Path(record.rel_path).suffix.lower()
            name = Path(record.rel_path).name.lower()
            if suffix in EXECUTABLE_EXTENSIONS:
                flags.add("contains_executable_code")
                score += 0.2
                if suffix in {".sh", ".bash", ".zsh", ".fish", ".ps1"}:
                    flags.add("contains_shell_script")
                    score += 0.1
            if name in EXECUTABLE_FILE_NAMES:
                flags.add("contains_package_manifest")
                score += 0.15
            if any(part in HIGH_RISK_PATH_PARTS for part in Path(rel_lower).parts):
                flags.add("contains_high_risk_paths")
                score += 0.05
            if not record.is_text and record.file_type == "file" and suffix not in STATIC_ASSET_EXTENSIONS:
                flags.add("contains_binary_or_hidden_blob")
                score += 0.15
            if not record.text_preview:
                continue
            text = record.text_preview.lower()
            if any(re.search(pattern, text) for pattern in PROMPT_INJECTION_PATTERNS):
                flags.add("prompt_injection_text")
                score += 0.15
            if any(re.search(pattern, text) for pattern in INSTALL_PATTERNS):
                flags.add("dependency_install_instructions")
                hard_block_reasons.add("dependency_install_required")
                score += 0.2
            if any(re.search(pattern, text) for pattern in SHELL_REQUIREMENT_PATTERNS):
                flags.add("shell_execution_instructions")
                if name in {"skill.md", "readme.md", "agents.md"}:
                    hard_block_reasons.add("explicit_shell_execution_required")
                score += 0.12
            if any(re.search(pattern, text) for pattern in NETWORK_PATTERNS):
                flags.add("network_requirements")
                score += 0.08
            if any(re.search(pattern, text) for pattern in DOWNLOAD_EXECUTE_PATTERNS):
                flags.add("runtime_download_execute_pattern")
                hard_block_reasons.add("runtime_download_execute")
                score += 0.25
            if any(re.search(pattern, text) for pattern in PRIVILEGE_PATTERNS):
                flags.add("privilege_or_system_path_intent")
                hard_block_reasons.add("privilege_escalation_or_system_mutation_intent")
                score += 0.25
            if any(re.search(pattern, text) for pattern in ENCODED_EXECUTION_PATTERNS):
                flags.add("encoded_or_obfuscated_payload")
                if "base64" in text or "eval(" in text:
                    hard_block_reasons.add("encoded_payload_with_execution_intent")
                score += 0.1

        score = min(1.0, max(0.0, score))
        if score >= 0.85:
            level = "critical"
        elif score >= 0.6:
            level = "high"
        elif score >= 0.3:
            level = "medium"
        else:
            level = "low"
        return PackRiskReport(
            score=score,
            level=level,
            flags=tuple(sorted(flags)),
            hard_block_reasons=tuple(sorted(hard_block_reasons)),
        )

    def _normalize(
        self,
        *,
        quarantined_path: Path,
        file_records: list[PackFileRecord],
        classification: str,
        risk_report: PackRiskReport,
        integrity: dict[str, Any],
        source_origin: str,
        source_url: str | None,
        commit_hash: str | None,
        created_by: str,
        remote_source: RemotePackSource | None = None,
        raw_archive_path: str | None = None,
    ) -> PackNormalizationResult:
        skill_record = next((record for record in file_records if record.rel_path.lower() == "skill.md"), None)
        skill_meta: dict[str, Any] = {}
        skill_body = ""
        if skill_record and skill_record.text_preview is not None:
            full_text, _ = _load_text_file(Path(skill_record.absolute_path), max_bytes=1024 * 1024)
            parsed_meta, parsed_body = _parse_frontmatter(full_text or skill_record.text_preview)
            skill_meta = parsed_meta
            skill_body = parsed_body

        metadata = self._read_metadata_files(file_records)
        combined_meta = {**metadata, **skill_meta}
        name = (
            str(combined_meta.get("name") or combined_meta.get("title") or "").strip()
            or _infer_first_heading(skill_body)
            or quarantined_path.name
        )
        version = str(combined_meta.get("version") or "0.1.0").strip() or "0.1.0"
        description = (
            str(combined_meta.get("description") or "").strip()
            or self._first_paragraph(skill_body)
            or f"Imported external pack from {quarantined_path.name}."
        )
        pack_id = _slugify(str(combined_meta.get("id") or name or quarantined_path.name))
        pack_type = {
            CLASS_PORTABLE_TEXT_SKILL: PACK_TYPE_SKILL,
            CLASS_EXPERIENCE_PACK: PACK_TYPE_EXPERIENCE,
            CLASS_NATIVE_CODE_PACK: PACK_TYPE_NATIVE,
        }.get(classification, PACK_TYPE_NATIVE)

        components: list[dict[str, Any]] = []
        assets: list[dict[str, Any]] = []
        stripped_components: list[str] = []
        granted_permissions: list[str] = []
        requested_permissions = self._normalize_str_list(
            combined_meta.get("permissions") or combined_meta.get("requested_permissions")
        )
        declared_capabilities = self._normalize_str_list(combined_meta.get("capabilities") or combined_meta.get("tags"))
        inferred_capabilities = ["text_instruction"] if skill_record else []
        normalized_dir = self.normalized_root / f"{pack_id}-{int(time.time() * 1000)}"
        if normalized_dir.exists():
            shutil.rmtree(normalized_dir)
        normalized_dir.mkdir(parents=True, exist_ok=True)

        if skill_record and skill_body:
            normalized_skill_path = normalized_dir / "SKILL.md"
            normalized_skill_path.write_text(skill_body.strip() + "\n", encoding="utf-8")
            components.append(
                PackComponent(
                    path="SKILL.md",
                    component_type="instruction",
                    included=True,
                    executable=False,
                    sha256=skill_record.sha256 if skill_record else None,
                    notes="normalized_skill_instructions",
                ).to_dict()
            )
        for record in sorted(file_records, key=lambda item: item.rel_path):
            rel_path = record.rel_path
            lower_path = rel_path.lower()
            suffix = Path(rel_path).suffix.lower()
            name_lower = Path(rel_path).name.lower()
            if lower_path == "skill.md":
                continue
            if lower_path == "agents.md":
                components.append(
                    PackComponent(
                        path=rel_path,
                        component_type="secondary_reference",
                        included=False,
                        executable=False,
                        sha256=record.sha256,
                        notes="ignored_secondary_agents_document",
                    ).to_dict()
                )
                continue
            if name_lower in EXECUTABLE_FILE_NAMES or suffix in EXECUTABLE_EXTENSIONS:
                components.append(
                    PackComponent(
                        path=rel_path,
                        component_type="disallowed",
                        included=False,
                        executable=True,
                        sha256=record.sha256,
                        notes="executable_components_are_not_imported",
                    ).to_dict()
                )
                stripped_components.append(rel_path)
                continue
            target_path = normalized_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if lower_path.startswith("references/") and record.is_text:
                shutil.copy2(record.absolute_path, target_path)
                components.append(
                    PackComponent(
                        path=rel_path,
                        component_type="reference",
                        included=True,
                        executable=False,
                        sha256=record.sha256,
                        notes="reference_material",
                    ).to_dict()
                )
                if "reference_material" not in inferred_capabilities:
                    inferred_capabilities.append("reference_material")
                continue
            if lower_path.startswith("assets/") and suffix not in EXECUTABLE_EXTENSIONS and name_lower not in EXECUTABLE_FILE_NAMES:
                shutil.copy2(record.absolute_path, target_path)
                assets.append(
                    {
                        "path": rel_path,
                        "asset_type": suffix.lstrip(".") or "file",
                        "included": True,
                        "executable": False,
                        "sha256": record.sha256,
                    }
                )
                if "static_assets" not in inferred_capabilities:
                    inferred_capabilities.append("static_assets")
                continue
            if record.is_text and suffix in SAFE_TEXT_EXTENSIONS:
                shutil.copy2(record.absolute_path, target_path)
                components.append(
                    PackComponent(
                        path=rel_path,
                        component_type="reference",
                        included=True,
                        executable=False,
                        sha256=record.sha256,
                        notes="safe_text_component",
                    ).to_dict()
                )
                if "reference_material" not in inferred_capabilities:
                    inferred_capabilities.append("reference_material")
                continue
            components.append(
                PackComponent(
                    path=rel_path,
                    component_type="disallowed",
                    included=False,
                    executable=False,
                    sha256=record.sha256,
                    notes="unsupported_or_non_text_component",
                ).to_dict()
            )
            stripped_components.append(rel_path)

        if classification == CLASS_PORTABLE_TEXT_SKILL and not risk_report.hard_block_reasons:
            if stripped_components:
                status = STATUS_PARTIAL_SAFE_IMPORT
                adaptation_strategy = "stripped_and_reconstructed"
                adaptation_notes = "Unsafe executable or unsupported components were stripped. Text and static assets only were preserved."
            else:
                status = STATUS_NORMALIZED
                adaptation_strategy = "imported_as_text_skill"
                adaptation_notes = "Pure portable text skill imported without executable content."
        else:
            status = STATUS_BLOCKED
            adaptation_strategy = "blocked_by_policy"
            adaptation_notes = "This pack was preserved for review but not imported as a runnable skill."

        source_ref = str(remote_source.ref or "").strip() or None if remote_source else None
        source_path_value = str(quarantined_path)
        content_hash = str(integrity.get("sha256") or "").strip()
        source_fingerprint = _normalize_source_fingerprint(
            source_url=source_url,
            resolved_commit=commit_hash,
            archive_sha256=str(integrity.get("archive_sha256") or "").strip() or None,
            source_path=source_path_value,
        )
        source_key = _normalize_source_key(
            source_origin=source_origin,
            source_url=source_url,
            source_path=source_path_value if source_origin == "local_path" else None,
        )
        normalized_integrity = self._aggregate_integrity(file_records, archive_sha256=str(integrity.get("archive_sha256") or "").strip() or None)
        canonical_id = str(normalized_integrity.get("sha256") or "").strip() or _sha256_text(
            _safe_json(
                _component_identity_payload(
                    components=[item for item in components if bool(item.get("included", False))],
                    assets=[item for item in assets if bool(item.get("included", False))],
                    classification=classification,
                )
            )
        )
        final_normalized_dir = self.normalized_root / canonical_id
        existing_source_history: list[dict[str, Any]] = []
        existing_source_json = final_normalized_dir / "metadata" / "source.json"
        if existing_source_json.exists():
            try:
                existing_payload = json.loads(existing_source_json.read_text(encoding="utf-8"))
                existing_source_history = [
                    dict(item)
                    for item in (
                        existing_payload.get("source_history") if isinstance(existing_payload, dict) and isinstance(existing_payload.get("source_history"), list) else []
                    )
                    if isinstance(item, dict)
                ]
            except (OSError, json.JSONDecodeError):
                existing_source_history = []
        if final_normalized_dir.exists():
            shutil.rmtree(final_normalized_dir)
        source_history_entry = {
            "source_key": source_key,
            "source_fingerprint": source_fingerprint,
            "origin": source_origin,
            "url": source_url,
            "ref": source_ref,
            "commit_hash": commit_hash,
            "archive_sha256": str(integrity.get("archive_sha256") or "").strip() or None,
            "fetched_at": _now_iso(),
            "source_path": source_path_value,
        }
        version_entry = {
            "canonical_id": canonical_id,
            "content_hash": content_hash,
            "status": status,
            "seen_at": _now_iso(),
        }

        canonical_pack = CanonicalPack(
            id=canonical_id,
            name=name,
            version=content_hash or version,
            type=pack_type,
            pack_identity={
                "canonical_id": canonical_id,
                "content_hash": content_hash,
                "source_fingerprint": source_fingerprint,
                "source_key": source_key,
            },
            source={
                "origin": source_origin,
                "url": source_url,
                "ref": (str(remote_source.ref or "").strip() or None) if remote_source else None,
                "commit_hash": commit_hash,
                "fetched_at": _now_iso(),
                "source_path": str(quarantined_path),
                "resolved_url": (str(remote_source.resolved_url or "").strip() or None) if remote_source else None,
                "transport": (str(remote_source.transport or "").strip() or None) if remote_source else None,
                "archive_sha256": (
                    (str(remote_source.archive_sha256 or "").strip() or None) if remote_source else integrity.get("archive_sha256")
                ),
                "top_level_dir_name": (str(remote_source.top_level_dir_name or "").strip() or None) if remote_source else None,
                "provenance_notes": list(remote_source.provenance_notes) if remote_source else [],
                "raw_archive_path": str(raw_archive_path or "").strip() or None,
            },
            integrity=integrity,
            trust={
                "level": "review_required" if status != STATUS_BLOCKED else "blocked",
                "risk_score": round(float(risk_report.score), 4),
                "flags": list(risk_report.flags),
            },
            trust_anchor={
                "first_seen_at": _now_iso(),
                "first_seen_source": source_history_entry,
                "local_review_status": "unreviewed",
                "user_approved_hashes": [],
            },
            capabilities={
                "declared": list(declared_capabilities),
                "inferred": list(dict.fromkeys(inferred_capabilities)),
                "summary": description,
            },
            permissions={
                "requested": list(requested_permissions),
                "granted": granted_permissions,
            },
            components=tuple(components),
            assets=tuple(assets),
            source_history=(source_history_entry,),
            versions=(version_entry,),
            runtime={
                "isolation": "quarantined_text_only",
                "requires_process": False,
                "requires_gpu": False,
            },
            adaptation={
                "strategy": adaptation_strategy,
                "notes": adaptation_notes,
            },
            audit={
                "created_by": created_by,
                "created_at": _now_iso(),
                "declared_id": pack_id,
                "review_log": [],
            },
        )

        source_docs = self._canonical_source_docs(file_records)
        self._write_canonical_pack_tree(
            final_normalized_dir,
            display_name=name,
            summary=description,
            canonical_id=canonical_id,
            version=content_hash or version,
            upstream_version=version,
            classification=classification,
            pack_type=pack_type,
            status=status,
            risk_report=risk_report,
            source_origin=source_origin,
            source_url=source_url,
            source_ref=source_ref,
            source_url_resolved=(str(remote_source.resolved_url or "").strip() or None) if remote_source else None,
            source_commit=commit_hash,
            source_key=source_key,
            source_fingerprint=source_fingerprint,
            quarantine_path=str(quarantined_path),
            source_history=existing_source_history + [source_history_entry],
            file_records=file_records,
            components=components,
            assets=assets,
            capabilities={
                "declared": list(declared_capabilities),
                "inferred": list(dict.fromkeys(inferred_capabilities)),
                "summary": description,
            },
            permissions_granted=granted_permissions,
            main_sections=list(source_docs["main_sections"]),
            example_sections=list(source_docs["example_sections"]),
            kept_snapshot_paths=list(source_docs["kept_snapshot_paths"]),
            dropped_paths=list(source_docs["dropped_paths"]),
            blocked_categories=list(source_docs["blocked_categories"]),
            normalized_integrity=normalized_integrity,
            content_hash=content_hash,
            raw_archive_path=raw_archive_path,
        )
        if normalized_dir.exists():
            shutil.rmtree(normalized_dir)

        blocked_reasons = list(risk_report.hard_block_reasons)
        if classification == CLASS_EXPERIENCE_PACK:
            blocked_reasons.append("experience_packs_are_not_supported_for_runtime_import")
        elif classification == CLASS_UNKNOWN_PACK:
            blocked_reasons.append("unsupported_pack_layout")
        return PackNormalizationResult(
            classification=classification,
            status=status,
            pack=canonical_pack,
            risk_report=risk_report,
            blocked_reasons=tuple(sorted(set(blocked_reasons))),
            stripped_components=tuple(sorted(set(stripped_components))),
            normalized_path=str(final_normalized_dir),
        )

    def _build_fetch_blocked_result(
        self,
        error: RemoteFetchError,
        *,
        created_by: str,
    ) -> PackNormalizationResult:
        source = error.source
        source_name = Path(urllib.parse.urlparse(source.url).path).name if str(source.url or "").strip() else "remote-pack"
        pack_name = source_name or "remote-pack"
        source_key = _normalize_source_key(
            source_origin=source.kind,
            source_url=source.url,
            source_path=error.quarantine_path,
        )
        content_hash = str(error.archive_sha256 or "").strip() or _sha256_text(_safe_json({"url": source.url, "kind": source.kind}))
        source_fingerprint = _normalize_source_fingerprint(
            source_url=source.url,
            resolved_commit=source.commit_hash_resolved,
            archive_sha256=error.archive_sha256 or source.archive_sha256,
            source_path=error.quarantine_path,
        )
        pack_id = _sha256_text(
            _safe_json(
                {
                    "classification": CLASS_UNKNOWN_PACK,
                    "components": [],
                    "assets": [],
                    "content_hash": content_hash,
                    "source_kind": source.kind,
                }
            )
        )
        flags = tuple(sorted(set(error.flags + ("archive_fetch",))))
        hard_blocks = tuple(sorted(set(error.hard_block_reasons or ("could_not_safely_fetch",))))
        first_seen_source = {
            "source_key": source_key,
            "source_fingerprint": source_fingerprint,
            "origin": source.kind,
            "url": source.url,
            "ref": source.ref,
            "commit_hash": source.commit_hash_resolved,
            "archive_sha256": error.archive_sha256 or source.archive_sha256,
            "fetched_at": source.fetched_at or _now_iso(),
            "source_path": error.quarantine_path,
        }
        risk_report = PackRiskReport(
            score=0.95,
            level="critical",
            flags=flags,
            hard_block_reasons=hard_blocks,
        )
        canonical_pack = CanonicalPack(
            id=pack_id,
            name=pack_name,
            version="0.0.0",
            type=PACK_TYPE_NATIVE,
            pack_identity={
                "canonical_id": pack_id,
                "content_hash": content_hash,
                "source_fingerprint": source_fingerprint,
                "source_key": source_key,
            },
            source={
                "origin": source.kind,
                "url": source.url,
                "ref": source.ref,
                "commit_hash": source.commit_hash_resolved,
                "fetched_at": source.fetched_at or _now_iso(),
                "source_path": error.quarantine_path,
                "resolved_url": source.resolved_url,
                "transport": source.transport,
                "archive_sha256": error.archive_sha256 or source.archive_sha256,
                "top_level_dir_name": source.top_level_dir_name,
                "provenance_notes": list(source.provenance_notes),
                "raw_archive_path": None,
            },
            integrity={
                "sha256": "",
                "files_hashed": 0,
                "signature_verified": False,
                "archive_sha256": error.archive_sha256 or source.archive_sha256,
            },
            trust={
                "level": "blocked",
                "risk_score": 0.95,
                "flags": list(flags),
            },
            trust_anchor={
                "first_seen_at": _now_iso(),
                "first_seen_source": first_seen_source,
                "local_review_status": "unreviewed",
                "user_approved_hashes": [],
            },
            capabilities={
                "declared": [],
                "inferred": [],
                "summary": error.message,
            },
            permissions={
                "requested": [],
                "granted": [],
            },
            components=(),
            assets=(),
            source_history=(first_seen_source,),
            versions=(
                {
                    "canonical_id": pack_id,
                    "content_hash": content_hash,
                    "status": STATUS_BLOCKED,
                    "seen_at": _now_iso(),
                },
            ),
            runtime={
                "isolation": "quarantined_text_only",
                "requires_process": False,
                "requires_gpu": False,
            },
            adaptation={
                "strategy": "blocked_by_fetch_policy",
                "notes": error.message,
            },
            audit={
                "created_by": created_by,
                "created_at": _now_iso(),
                "review_log": [],
            },
        )
        return PackNormalizationResult(
            classification=CLASS_UNKNOWN_PACK,
            status=STATUS_BLOCKED,
            pack=canonical_pack,
            risk_report=risk_report,
            blocked_reasons=hard_blocks,
            stripped_components=(),
            quarantine_path=error.quarantine_path,
            normalized_path=None,
        )

    def _read_metadata_files(self, file_records: list[PackFileRecord]) -> dict[str, Any]:
        for candidate in ("metadata.json", "metadata.yaml", "metadata.yml"):
            record = next((row for row in file_records if row.rel_path.lower() == candidate), None)
            if record is None or not record.text_preview:
                continue
            if candidate.endswith(".json"):
                try:
                    parsed = json.loads(record.text_preview)
                except json.JSONDecodeError:
                    continue
                return parsed if isinstance(parsed, dict) else {}
            return _parse_simple_metadata_text(record.text_preview)
        return {}

    @staticmethod
    def _normalize_str_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return sorted({str(item).strip() for item in value if str(item).strip()})
        if isinstance(value, str):
            return sorted({part.strip() for part in value.split(",") if part.strip()})
        return []

    @staticmethod
    def _first_paragraph(text: str) -> str | None:
        for block in re.split(r"\n\s*\n", text):
            stripped = block.strip()
            if stripped:
                return stripped[:280]
        return None

    @staticmethod
    def _build_user_envelope(result: PackNormalizationResult) -> PackUserDecisionEnvelope:
        pack = result.pack
        source = pack.source if isinstance(pack.source, dict) else {}
        found_inside: list[str] = []
        instruction_count = sum(1 for item in pack.components if item.get("included") and item.get("component_type") == "instruction")
        reference_count = sum(1 for item in pack.components if item.get("included") and item.get("component_type") == "reference")
        asset_count = len(pack.assets)
        stripped_count = len(result.stripped_components)
        if instruction_count:
            found_inside.append(f"{instruction_count} instruction file")
        if reference_count:
            found_inside.append(f"{reference_count} reference file")
        if asset_count:
            found_inside.append(f"{asset_count} static asset")
        if stripped_count:
            found_inside.append(f"{stripped_count} stripped unsafe or unsupported file")
        origin = str(source.get("origin") or "").strip()
        if origin and origin != "local_path":
            found_inside.append(f"remote source: {origin}")
        if not found_inside:
            found_inside.append("metadata only")

        why_risk = list(result.risk_report.flags)
        if result.blocked_reasons:
            why_risk.extend(result.blocked_reasons)
        provenance_notes = source.get("provenance_notes")
        if isinstance(provenance_notes, list):
            why_risk.extend(str(item).strip() for item in provenance_notes if str(item).strip())
        if not why_risk:
            why_risk.append("portable_text_skill_layout")

        if result.status == STATUS_NORMALIZED:
            safe_options = (
                "Import instructions and reference material only.",
                "Recreate a stricter local version later if needed.",
            )
        elif result.status == STATUS_PARTIAL_SAFE_IMPORT:
            safe_options = (
                "Use the imported text and assets only.",
                "Recreate a safe local version without the stripped executable parts.",
                "Request advanced review before trusting anything else in the pack.",
            )
        else:
            safe_options = (
                "Block and keep only the audit snapshot.",
                "Recreate a safe local version from the text instructions manually.",
                "Request advanced review if you truly need the blocked runtime behavior.",
            )
        appears_to_do = str(pack.capabilities.get("summary") or f"{pack.name} imported from an external source.").strip()
        summary_prefix = ""
        source_url = str(source.get("url") or "").strip()
        source_ref = str(source.get("ref") or "").strip()
        resolved_commit = str(source.get("commit_hash") or "").strip()
        adaptation = pack.adaptation if isinstance(pack.adaptation, dict) else {}
        if origin and origin != "local_path":
            if str(adaptation.get("strategy") or "").strip() == "blocked_by_fetch_policy":
                summary_prefix = "I could not safely fetch this remote source. "
            else:
                summary_prefix = "I fetched a snapshot from a remote source. "
            if source_url:
                if str(adaptation.get("strategy") or "").strip() == "blocked_by_fetch_policy":
                    summary_prefix = f"I could not safely fetch {source_url}. "
                else:
                    summary_prefix = f"I fetched a snapshot from {source_url}. "
            if source_ref and not resolved_commit:
                summary_prefix += "This source was not pinned to a stable commit, so I treated it as higher risk. "
            elif resolved_commit:
                summary_prefix += f"It was pinned to {resolved_commit}. "
        artifact_label = result.classification.replace("_", " ")
        if result.status == STATUS_NORMALIZED:
            summary = summary_prefix + compose_actionable_message(
                what_happened=f"{pack.name} looks like a {artifact_label}",
                why=f"I imported only safe text/reference content and assigned {result.risk_report.level} risk.",
                next_action="Review the imported pack details before relying on it.",
            )
        elif result.status == STATUS_PARTIAL_SAFE_IMPORT:
            summary = summary_prefix + compose_actionable_message(
                what_happened=f"{pack.name} included some safe content I could keep",
                why=(
                    f"I stripped executable or unsafe files, kept only the safe text/static parts, "
                    f"and assigned {result.risk_report.level} risk."
                ),
                next_action="Use the imported safe content, or request advanced review if you need anything else from the pack.",
            )
        else:
            summary = summary_prefix + compose_actionable_message(
                what_happened=f"{pack.name} is not compatible with the current safe import policy",
                why=f"I blocked it after review and assigned {result.risk_report.level} risk.",
                next_action="Keep the audit snapshot only, or recreate a safe local text-only version manually.",
            )
        return PackUserDecisionEnvelope(
            pack_name=pack.name,
            appears_to_do=appears_to_do,
            found_inside=tuple(found_inside),
            risk_level=result.risk_report.level,
            why_risk=tuple(sorted(dict.fromkeys(why_risk))),
            safe_options=tuple(safe_options),
            review_required=True,
            summary=summary,
        )

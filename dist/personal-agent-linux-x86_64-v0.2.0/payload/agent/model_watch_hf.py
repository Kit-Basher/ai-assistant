from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable


_HF_STATE_SCHEMA_VERSION = 1
_GGUF_RE = re.compile(r"\.gguf$", re.IGNORECASE)
_MODEFILE_NAME = "modelfile"
_SAFE_MODEL_NAME_RE = re.compile(r"[^a-z0-9._-]+")
_QUANT_PREFERENCE = (
    "q4_k_m",
    "q4km",
    "q4_k_s",
    "q5_k_m",
    "q5km",
    "q8_0",
    "f16",
)


@dataclass(frozen=True)
class HFScanDelta:
    scanned_repos: int
    updates: tuple[dict[str, Any], ...]
    last_run_ts: int
    state_path: str


def default_hf_watch_state_document() -> dict[str, Any]:
    return {
        "schema_version": _HF_STATE_SCHEMA_VERSION,
        "last_run_ts": None,
        "last_error": None,
        "discovered_count": 0,
        "repos": {},
    }


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _normalize_hf_state(raw: dict[str, Any]) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    repos_raw = payload.get("repos") if isinstance(payload.get("repos"), dict) else {}
    repos: dict[str, dict[str, Any]] = {}
    for repo_id, repo_row in sorted(repos_raw.items()):
        normalized_repo = str(repo_id or "").strip()
        if not normalized_repo:
            continue
        row = repo_row if isinstance(repo_row, dict) else {}
        first_seen_ts = _safe_int(row.get("first_seen_ts"))
        last_seen_ts = _safe_int(row.get("last_seen_ts"))
        revision = str(row.get("revision") or "").strip() or None
        meta_hash = str(row.get("meta_hash") or "").strip().lower() or None
        if first_seen_ts is None or last_seen_ts is None:
            continue
        interesting_raw = row.get("interesting_files") if isinstance(row.get("interesting_files"), list) else []
        interesting_files: list[dict[str, Any]] = []
        for file_row in interesting_raw:
            if not isinstance(file_row, dict):
                continue
            rel_path = str(file_row.get("path") or "").strip()
            if not rel_path:
                continue
            size = _safe_int(file_row.get("size"))
            kind = str(file_row.get("kind") or "").strip().lower()
            if kind not in {"gguf", "modelfile", "ollama_hint"}:
                kind = "ollama_hint"
            interesting_files.append(
                {
                    "path": rel_path,
                    "size": size,
                    "kind": kind,
                }
            )
        selected_gguf = str(row.get("selected_gguf") or "").strip() or None
        installability = str(row.get("installability") or "").strip().lower()
        if installability not in {"installable_ollama", "download_only"}:
            installability = "download_only"
        total_size_bytes = _safe_int(row.get("total_size_bytes"))
        repos[normalized_repo] = {
            "first_seen_ts": int(first_seen_ts),
            "last_seen_ts": int(last_seen_ts),
            "revision": revision,
            "meta_hash": meta_hash,
            "interesting_files": sorted(
                interesting_files,
                key=lambda item: (str(item.get("kind") or ""), str(item.get("path") or "")),
            ),
            "selected_gguf": selected_gguf,
            "installability": installability,
            "total_size_bytes": total_size_bytes,
        }
    last_run_ts = _safe_int(payload.get("last_run_ts"))
    discovered_count = _safe_int(payload.get("discovered_count"))
    return {
        "schema_version": _HF_STATE_SCHEMA_VERSION,
        "last_run_ts": int(last_run_ts) if last_run_ts is not None else None,
        "last_error": str(payload.get("last_error") or "").strip() or None,
        "discovered_count": int(discovered_count) if discovered_count is not None else 0,
        "repos": repos,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
    return payload


def hf_watch_state_path_for_runtime(runtime: Any) -> Path:
    runtime_path = str(getattr(runtime, "_model_watch_hf_state_path", "") or "").strip()
    if runtime_path:
        return Path(runtime_path).expanduser().resolve()
    explicit = str(getattr(getattr(runtime, "config", None), "model_watch_hf_state_path", "") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_path = os.getenv("AGENT_MODEL_WATCH_HF_STATE_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path.home() / ".local" / "share" / "personal-agent" / "model_watch_hf_state.json").resolve()


def hf_download_base_path_for_runtime(runtime: Any) -> Path:
    explicit = str(getattr(getattr(runtime, "config", None), "model_watch_hf_download_base_path", "") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_path = os.getenv("AGENT_MODEL_WATCH_HF_DOWNLOAD_BASE_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path.home() / ".local" / "share" / "personal-agent" / "hf_models").resolve()


def load_hf_watch_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return default_hf_watch_state_document()
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default_hf_watch_state_document()
    if not isinstance(parsed, dict):
        return default_hf_watch_state_document()
    return _normalize_hf_state(parsed)


def save_hf_watch_state(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_hf_state(state)
    return _write_json_atomic(path, normalized)


def _load_hf_api() -> Any:
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("huggingface_hub_missing") from exc
    return HfApi()


def _load_snapshot_download() -> Any:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("huggingface_hub_missing") from exc
    return snapshot_download


def _extract_repo_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("id") or item.get("modelId") or "").strip()
    for attr in ("id", "modelId", "repo_id"):
        value = str(getattr(item, attr, "") or "").strip()
        if value:
            return value
    return ""


def _extract_revision(info: Any) -> str:
    if isinstance(info, dict):
        return str(info.get("sha") or info.get("revision") or "").strip()
    for attr in ("sha", "revision"):
        value = str(getattr(info, attr, "") or "").strip()
        if value:
            return value
    return ""


def _extract_siblings(info: Any) -> list[dict[str, Any]]:
    rows = info.get("siblings") if isinstance(info, dict) else getattr(info, "siblings", [])
    output: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return output
    for row in rows:
        if isinstance(row, dict):
            path = str(row.get("rfilename") or row.get("path") or row.get("filename") or "").strip()
            size = _safe_int(row.get("size"))
        else:
            path = str(getattr(row, "rfilename", "") or getattr(row, "path", "") or "").strip()
            size = _safe_int(getattr(row, "size", None))
        if not path:
            continue
        output.append({"path": path, "size": size})
    output.sort(key=lambda item: str(item.get("path") or ""))
    return output


def _interesting_kind(path: str) -> str | None:
    text = str(path or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if _GGUF_RE.search(lowered):
        return "gguf"
    if Path(lowered).name == _MODEFILE_NAME:
        return "modelfile"
    if "ollama" in lowered:
        return "ollama_hint"
    return None


def _quant_rank(path: str) -> int:
    lowered = str(path or "").strip().lower().replace("-", "_")
    for idx, marker in enumerate(_QUANT_PREFERENCE):
        if marker in lowered:
            return idx
    return len(_QUANT_PREFERENCE) + 1


def _select_gguf(files: Iterable[dict[str, Any]]) -> str | None:
    gguf_rows = [
        row
        for row in files
        if isinstance(row, dict) and str(row.get("kind") or "") == "gguf"
    ]
    if not gguf_rows:
        return None
    ordered = sorted(
        gguf_rows,
        key=lambda row: (
            _quant_rank(str(row.get("path") or "")),
            int(_safe_int(row.get("size")) or 2**62),
            str(row.get("path") or ""),
        ),
    )
    return str(ordered[0].get("path") or "").strip() or None


def _meta_hash(*, revision: str, interesting_files: list[dict[str, Any]], selected_gguf: str | None) -> str:
    payload = {
        "revision": str(revision or "").strip(),
        "selected_gguf": str(selected_gguf or "").strip() or None,
        "interesting_files": [
            {
                "kind": str(row.get("kind") or ""),
                "path": str(row.get("path") or ""),
                "size": _safe_int(row.get("size")),
            }
            for row in sorted(
                [item for item in interesting_files if isinstance(item, dict)],
                key=lambda item: (str(item.get("kind") or ""), str(item.get("path") or "")),
            )
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _allowlisted_repo_ids(
    *,
    client: Any,
    allowlist_repos: tuple[str, ...],
    allowlist_orgs: tuple[str, ...],
) -> list[str]:
    repo_ids = {str(item).strip() for item in allowlist_repos if str(item).strip()}
    for org in allowlist_orgs:
        org_name = str(org or "").strip()
        if not org_name:
            continue
        rows = client.list_models(author=org_name)
        for item in rows:
            repo_id = _extract_repo_id(item)
            if repo_id:
                repo_ids.add(repo_id)
    return sorted(repo_ids)


def _format_bytes(value: int | None) -> str:
    if value is None or value < 0:
        return "unknown"
    gb = float(value) / float(1024**3)
    return f"{gb:.2f} GiB"


def deterministic_ollama_model_name(*, repo_id: str, selected_gguf: str | None, revision: str) -> str:
    base = str(repo_id or "").strip().lower().replace("/", "-")
    quant = "q"
    if selected_gguf:
        stem = Path(str(selected_gguf)).name.lower().replace("-", "_")
        for marker in _QUANT_PREFERENCE:
            if marker in stem:
                quant = marker
                break
    rev = str(revision or "")[:8] or "latest"
    name = f"hf-{base}-{quant}-{rev}"
    name = _SAFE_MODEL_NAME_RE.sub("-", name).strip("-")
    return name[:120]


def scan_hf_watch(
    runtime: Any,
    *,
    client: Any | None = None,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    path = hf_watch_state_path_for_runtime(runtime)
    state = load_hf_watch_state(path)
    now = int(now_epoch if now_epoch is not None else time.time())
    config = getattr(runtime, "config", None)
    enabled = bool(getattr(config, "model_watch_hf_enabled", False))
    allow_repos = tuple(
        sorted(
            {
                str(item).strip()
                for item in (getattr(config, "model_watch_hf_allowlist_repos", tuple()) or tuple())
                if str(item).strip()
            }
        )
    )
    allow_orgs = tuple(
        sorted(
            {
                str(item).strip()
                for item in (getattr(config, "model_watch_hf_allowlist_orgs", tuple()) or tuple())
                if str(item).strip()
            }
        )
    )
    require_gguf_for_install = bool(getattr(config, "model_watch_hf_require_gguf_for_install", True))
    max_total_bytes = int(max(0, int(getattr(config, "model_watch_hf_max_total_bytes", 40 * 1024**3) or 0)))

    if not enabled:
        state["last_run_ts"] = now
        state["last_error"] = None
        state["discovered_count"] = 0
        save_hf_watch_state(path, state)
        return {
            "ok": True,
            "enabled": False,
            "scanned_repos": 0,
            "discovered_count": 0,
            "updates": [],
            "allowlist_repos": list(allow_repos),
            "allowlist_orgs": list(allow_orgs),
            "last_run_ts": now,
            "state_path": str(path),
        }

    if not allow_repos and not allow_orgs:
        state["last_run_ts"] = now
        state["last_error"] = None
        state["discovered_count"] = 0
        save_hf_watch_state(path, state)
        return {
            "ok": True,
            "enabled": True,
            "scanned_repos": 0,
            "discovered_count": 0,
            "updates": [],
            "allowlist_repos": [],
            "allowlist_orgs": [],
            "last_run_ts": now,
            "state_path": str(path),
            "reason": "allowlist_empty",
        }

    try:
        hf_client = client if client is not None else _load_hf_api()
        repo_ids = _allowlisted_repo_ids(client=hf_client, allowlist_repos=allow_repos, allowlist_orgs=allow_orgs)

        repos_state = state.get("repos") if isinstance(state.get("repos"), dict) else {}
        updates: list[dict[str, Any]] = []
        next_repos: dict[str, dict[str, Any]] = dict(repos_state)

        for repo_id in repo_ids:
            info = hf_client.model_info(repo_id, files_metadata=True)
            revision = _extract_revision(info) or "unknown"
            siblings = _extract_siblings(info)
            interesting_files: list[dict[str, Any]] = []
            total_size_bytes = 0
            size_known = True
            for row in siblings:
                file_path = str(row.get("path") or "").strip()
                if not file_path:
                    continue
                size = _safe_int(row.get("size"))
                if size is None:
                    size_known = False
                else:
                    total_size_bytes += int(size)
                kind = _interesting_kind(file_path)
                if kind is None:
                    continue
                interesting_files.append(
                    {
                        "path": file_path,
                        "size": size,
                        "kind": kind,
                    }
                )
            interesting_files.sort(key=lambda item: (str(item.get("kind") or ""), str(item.get("path") or "")))
            selected_gguf = _select_gguf(interesting_files)
            installability = "installable_ollama" if selected_gguf else "download_only"
            if require_gguf_for_install and not selected_gguf:
                installability = "download_only"
            estimated_total_bytes = int(total_size_bytes) if size_known else None
            over_size_limit = bool(
                estimated_total_bytes is not None
                and max_total_bytes > 0
                and int(estimated_total_bytes) > int(max_total_bytes)
            )
            meta_hash = _meta_hash(
                revision=revision,
                interesting_files=interesting_files,
                selected_gguf=selected_gguf,
            )
            previous = repos_state.get(repo_id) if isinstance(repos_state.get(repo_id), dict) else None
            previous_revision = str((previous or {}).get("revision") or "").strip()
            previous_meta_hash = str((previous or {}).get("meta_hash") or "").strip().lower()
            changed = previous is None or previous_revision != revision or previous_meta_hash != meta_hash
            first_seen_ts = int(_safe_int((previous or {}).get("first_seen_ts")) or now)

            next_repos[repo_id] = {
                "first_seen_ts": first_seen_ts,
                "last_seen_ts": now,
                "revision": revision,
                "meta_hash": meta_hash,
                "interesting_files": interesting_files,
                "selected_gguf": selected_gguf,
                "installability": installability,
                "total_size_bytes": estimated_total_bytes,
            }

            if not changed:
                continue
            status = "new" if previous is None else "changed"
            updates.append(
                {
                    "status": status,
                    "repo_id": repo_id,
                    "revision": revision,
                    "interesting_files": interesting_files,
                    "interesting_files_count": len(interesting_files),
                    "selected_gguf": selected_gguf,
                    "installability": installability,
                    "estimated_total_bytes": estimated_total_bytes,
                    "estimated_total_human": _format_bytes(estimated_total_bytes),
                    "over_size_limit": over_size_limit,
                    "recommended_action": (
                        "download_install"
                        if installability == "installable_ollama"
                        else "download_snapshot"
                    ),
                    "meta_hash": meta_hash,
                }
            )

        state["repos"] = {key: next_repos[key] for key in sorted(next_repos.keys())}
        state["last_run_ts"] = now
        state["last_error"] = None
        state["discovered_count"] = len(updates)
        save_hf_watch_state(path, state)

        updates.sort(
            key=lambda item: (
                0 if str(item.get("installability") or "") == "installable_ollama" else 1,
                0 if not bool(item.get("over_size_limit")) else 1,
                str(item.get("repo_id") or ""),
                str(item.get("revision") or ""),
            )
        )
        return {
            "ok": True,
            "enabled": True,
            "scanned_repos": len(repo_ids),
            "discovered_count": len(updates),
            "updates": updates,
            "allowlist_repos": list(allow_repos),
            "allowlist_orgs": list(allow_orgs),
            "last_run_ts": now,
            "state_path": str(path),
        }
    except Exception as exc:
        state["last_run_ts"] = now
        state["last_error"] = str(exc.__class__.__name__)
        save_hf_watch_state(path, state)
        return {
            "ok": False,
            "enabled": True,
            "error": "hf_scan_failed",
            "detail": str(exc.__class__.__name__),
            "scanned_repos": 0,
            "discovered_count": 0,
            "updates": [],
            "allowlist_repos": list(allow_repos),
            "allowlist_orgs": list(allow_orgs),
            "last_run_ts": now,
            "state_path": str(path),
        }


def hf_status(runtime: Any) -> dict[str, Any]:
    path = hf_watch_state_path_for_runtime(runtime)
    state = load_hf_watch_state(path)
    last_run_ts = _safe_int(state.get("last_run_ts"))
    discovered_count = int(_safe_int(state.get("discovered_count")) or 0)
    repos = state.get("repos") if isinstance(state.get("repos"), dict) else {}
    return {
        "ok": True,
        "enabled": bool(getattr(getattr(runtime, "config", None), "model_watch_hf_enabled", False)),
        "last_run_ts": int(last_run_ts) if last_run_ts is not None else None,
        "last_run_ts_iso": (
            datetime.fromtimestamp(int(last_run_ts), tz=timezone.utc).isoformat()
            if last_run_ts is not None
            else None
        ),
        "last_error": str(state.get("last_error") or "").strip() or None,
        "discovered_count": discovered_count,
        "tracked_repos": len(repos),
        "state_path": str(path),
    }


def build_hf_local_download_proposal(runtime: Any, *, scan_payload: dict[str, Any]) -> dict[str, Any] | None:
    updates = scan_payload.get("updates") if isinstance(scan_payload.get("updates"), list) else []
    if not updates:
        return None
    sorted_updates = sorted(
        [row for row in updates if isinstance(row, dict)],
        key=lambda item: (
            0 if str(item.get("installability") or "") == "installable_ollama" else 1,
            0 if not bool(item.get("over_size_limit")) else 1,
            str(item.get("repo_id") or ""),
        ),
    )
    if not sorted_updates:
        return None
    selected = sorted_updates[0]
    repo_id = str(selected.get("repo_id") or "").strip()
    revision = str(selected.get("revision") or "").strip()
    installability = str(selected.get("installability") or "download_only").strip().lower()
    selected_gguf = str(selected.get("selected_gguf") or "").strip() or None
    estimated_total_bytes = _safe_int(selected.get("estimated_total_bytes"))
    over_size_limit = bool(selected.get("over_size_limit"))
    download_base = hf_download_base_path_for_runtime(runtime)
    target_dir = (download_base / repo_id / (revision or "latest")).resolve()
    allow_patterns: list[str] = []
    if selected_gguf:
        allow_patterns.append(selected_gguf)
    for row in selected.get("interesting_files") if isinstance(selected.get("interesting_files"), list) else []:
        if not isinstance(row, dict):
            continue
        path = str(row.get("path") or "").strip()
        kind = str(row.get("kind") or "").strip().lower()
        if kind == "modelfile" and path:
            allow_patterns.append(path)
    allow_patterns = sorted({item for item in allow_patterns if item})
    if not allow_patterns:
        allow_patterns = ["*"]

    plan_rows: list[dict[str, Any]] = [
        {
            "id": "01_hf.snapshot_download",
            "kind": "safe_action",
            "action": "hf.snapshot_download",
            "reason": "Download allowlisted HF snapshot.",
            "params": {
                "repo_id": repo_id,
                "revision": revision,
                "target_dir": str(target_dir),
                "allow_patterns": allow_patterns,
                "estimated_total_bytes": estimated_total_bytes,
            },
            "safe_to_execute": True,
        }
    ]

    if installability == "installable_ollama" and selected_gguf:
        model_name = deterministic_ollama_model_name(
            repo_id=repo_id,
            selected_gguf=selected_gguf,
            revision=revision,
        )
        modelfile_path = (target_dir / "Modelfile.personal-agent").resolve()
        plan_rows.extend(
            [
                {
                    "id": "02_hf.generate_modelfile",
                    "kind": "safe_action",
                    "action": "hf.generate_modelfile",
                    "reason": "Generate deterministic Modelfile for Ollama.",
                    "params": {
                        "selected_gguf": selected_gguf,
                        "target_dir": str(target_dir),
                        "modelfile_path": str(modelfile_path),
                        "ollama_model_name": model_name,
                    },
                    "safe_to_execute": True,
                },
                {
                    "id": "03_hf.ollama_create",
                    "kind": "safe_action",
                    "action": "hf.ollama_create",
                    "reason": "Create local Ollama model from downloaded GGUF.",
                    "params": {
                        "modelfile_path": str(modelfile_path),
                        "ollama_model_name": model_name,
                    },
                    "safe_to_execute": True,
                },
                {
                    "id": "04_hf.refresh_ollama_registry",
                    "kind": "safe_action",
                    "action": "hf.refresh_ollama_registry",
                    "reason": "Refresh Ollama model inventory and registry state.",
                    "params": {"provider": "ollama"},
                    "safe_to_execute": True,
                },
            ]
        )
    else:
        plan_rows.append(
            {
                "id": "02_hf.mark_download_only",
                "kind": "safe_action",
                "action": "hf.mark_download_only",
                "reason": "Mark model snapshot downloaded for offline inspection.",
                "params": {
                    "repo_id": repo_id,
                    "revision": revision,
                    "target_dir": str(target_dir),
                },
                "safe_to_execute": True,
            }
        )

    recommendation = "download/install" if installability == "installable_ollama" else "download"
    details_lines = [
        f"Repo: {repo_id}",
        f"Revision: {revision}",
        f"Installability: {installability}",
        f"Selected file: {selected_gguf or 'none'}",
        f"Estimated size: {_format_bytes(estimated_total_bytes)}",
    ]
    if over_size_limit:
        details_lines.append("This exceeds the configured max size limit.")
    message_lines = [
        "Model Watch found a Hugging Face local model opportunity.",
        f"Repo: {repo_id}",
        f"Recommended action: {recommendation}",
        "Reply 1 to Download/install, 2 to Snooze, or 3 for Details.",
    ]
    return {
        "issue_code": "model_watch.proposal",
        "proposal_type": "local_download",
        "repo_id": repo_id,
        "revision": revision,
        "installability": installability,
        "selected_gguf": selected_gguf,
        "estimated_total_bytes": estimated_total_bytes,
        "over_size_limit": over_size_limit,
        "message": "\n".join(message_lines),
        "details": "\n".join(details_lines),
        "plan_rows": plan_rows,
        "choices": [
            {"id": "download_install_local", "label": "Download/install", "recommended": True},
            {"id": "snooze_model_watch", "label": "Snooze", "recommended": False},
            {"id": "details", "label": "Show details", "recommended": False},
        ],
    }


def hf_snapshot_download(
    *,
    repo_id: str,
    revision: str,
    target_dir: str,
    allow_patterns: list[str] | None = None,
) -> str:
    snapshot_download = _load_snapshot_download()
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns or []),
    )
    return str(path)


__all__ = [
    "HFScanDelta",
    "build_hf_local_download_proposal",
    "default_hf_watch_state_document",
    "deterministic_ollama_model_name",
    "hf_download_base_path_for_runtime",
    "hf_snapshot_download",
    "hf_status",
    "hf_watch_state_path_for_runtime",
    "load_hf_watch_state",
    "save_hf_watch_state",
    "scan_hf_watch",
]


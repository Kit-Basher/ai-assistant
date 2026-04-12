from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_DEFAULT_LIST_LIMIT = 200
_DEFAULT_READ_MAX_BYTES = 8192
_HARD_READ_MAX_BYTES = 65536
_DEFAULT_SEARCH_RESULTS = 25
_DEFAULT_SEARCH_DEPTH = 4
_DEFAULT_SEARCH_MAX_FILES = 200
_DEFAULT_SEARCH_MAX_BYTES_PER_FILE = 8192
_HARD_SEARCH_RESULTS = 100
_HARD_SEARCH_DEPTH = 8
_HARD_SEARCH_MAX_FILES = 1000


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class FileSystemSkill:
    """Native read-only filesystem access with explicit path boundaries."""

    def __init__(
        self,
        *,
        allowed_roots: list[str] | tuple[str, ...],
        base_dir: str | Path | None = None,
        sensitive_roots: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir or Path.cwd()).expanduser().resolve(strict=False)
        self.allowed_roots = self._normalize_roots(allowed_roots or [str(Path.home())])
        self.sensitive_roots = self._normalize_roots(
            sensitive_roots
            or [
                "~/.ssh",
                "~/.gnupg",
                "~/.aws",
                "~/.azure",
                "~/.kube",
                "~/.config/gcloud",
                "~/.config/google-chrome",
                "~/.config/chromium",
                "~/.mozilla",
                "~/.local/share/keyrings",
                "~/.config/1Password",
                "~/.config/Bitwarden",
                "~/.netrc",
            ]
        )

    @staticmethod
    def _normalize_roots(roots: list[str] | tuple[str, ...]) -> list[Path]:
        normalized: list[Path] = []
        seen: set[str] = set()
        for raw_root in roots:
            root = Path(str(raw_root or "").strip()).expanduser()
            if not str(root).strip():
                continue
            resolved = root.resolve(strict=False)
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(resolved)
        return normalized

    def _error_result(
        self,
        action: str,
        path: str | None,
        *,
        resolved_path: str | None = None,
        error_kind: str,
        message: str,
        exists: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "action": action,
            "path": str(path or "").strip() or None,
            "resolved_path": str(resolved_path or "").strip() or None,
            "error_kind": error_kind,
            "message": message,
        }
        if exists is not None:
            payload["exists"] = bool(exists)
        return payload

    def _resolve_request_path(self, raw_path: str | None) -> tuple[Path | None, dict[str, Any] | None]:
        requested = str(raw_path or "").strip()
        if not requested:
            return None, self._error_result(
                "path_resolution",
                raw_path,
                error_kind="missing_path",
                message="I need an exact file or directory path for that request.",
            )
        candidate = Path(requested).expanduser()
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        resolved = candidate.resolve(strict=False)
        if not any(_is_relative_to(resolved, root) for root in self.allowed_roots):
            return None, self._error_result(
                "path_resolution",
                raw_path,
                resolved_path=str(resolved),
                error_kind="outside_allowed_roots",
                message="That path is outside the allowed local file roots.",
            )
        if any(_is_relative_to(resolved, root) for root in self.sensitive_roots):
            return None, self._error_result(
                "path_resolution",
                raw_path,
                resolved_path=str(resolved),
                error_kind="sensitive_path_blocked",
                message="That path is blocked by the local privacy policy.",
            )
        return resolved, None

    @staticmethod
    def _bounded_int(value: int | None, *, default: int, minimum: int, maximum: int) -> int:
        try:
            candidate = int(value if value is not None else default)
        except (TypeError, ValueError):
            candidate = default
        return max(minimum, min(candidate, maximum))

    @staticmethod
    def _path_type(path: Path) -> str:
        if path.is_file():
            return "file"
        if path.is_dir():
            return "dir"
        if path.is_symlink():
            return "symlink"
        return "other"

    def list_directory(self, path: str | None, *, max_entries: int = _DEFAULT_LIST_LIMIT) -> dict[str, Any]:
        resolved, error = self._resolve_request_path(path)
        if error is not None:
            return self._error_result(
                "list_directory",
                path,
                resolved_path=error.get("resolved_path"),
                error_kind=str(error.get("error_kind") or "path_resolution_failed"),
                message=str(error.get("message") or "Directory access failed."),
            )
        assert resolved is not None
        if not resolved.exists():
            return self._error_result(
                "list_directory",
                path,
                resolved_path=str(resolved),
                error_kind="not_found",
                message="That directory does not exist.",
                exists=False,
            )
        if not resolved.is_dir():
            return self._error_result(
                "list_directory",
                path,
                resolved_path=str(resolved),
                error_kind="not_directory",
                message="That path is not a directory.",
                exists=True,
            )
        try:
            scan_rows: list[os.DirEntry[str]] = []
            with os.scandir(resolved) as iterator:
                scan_rows.extend(iterator)
        except PermissionError:
            return self._error_result(
                "list_directory",
                path,
                resolved_path=str(resolved),
                error_kind="not_readable",
                message="That directory is not readable.",
                exists=True,
            )

        entries: list[dict[str, Any]] = []
        truncated = False
        for entry in sorted(scan_rows, key=lambda item: item.name.lower()):
            if len(entries) >= max(1, int(max_entries or _DEFAULT_LIST_LIMIT)):
                truncated = True
                break
            try:
                stat_result = entry.stat(follow_symlinks=False)
                size = int(stat_result.st_size)
            except OSError:
                size = None
            entry_type = "other"
            try:
                if entry.is_dir(follow_symlinks=False):
                    entry_type = "dir"
                elif entry.is_file(follow_symlinks=False):
                    entry_type = "file"
                elif entry.is_symlink():
                    entry_type = "symlink"
            except OSError:
                entry_type = "other"
            entries.append(
                {
                    "name": entry.name,
                    "type": entry_type,
                    "size": size,
                }
            )

        return {
            "ok": True,
            "action": "list_directory",
            "path": str(path or "").strip() or None,
            "resolved_path": str(resolved),
            "entries": entries,
            "entry_count": len(entries),
            "truncated": truncated,
        }

    def stat_path(self, path: str | None) -> dict[str, Any]:
        resolved, error = self._resolve_request_path(path)
        if error is not None:
            return self._error_result(
                "stat_path",
                path,
                resolved_path=error.get("resolved_path"),
                error_kind=str(error.get("error_kind") or "path_resolution_failed"),
                message=str(error.get("message") or "Path access failed."),
            )
        assert resolved is not None
        if not resolved.exists():
            return self._error_result(
                "stat_path",
                path,
                resolved_path=str(resolved),
                error_kind="not_found",
                message="That path does not exist.",
                exists=False,
            )
        try:
            stat_result = resolved.stat()
        except PermissionError:
            return self._error_result(
                "stat_path",
                path,
                resolved_path=str(resolved),
                error_kind="not_readable",
                message="That path is not readable.",
                exists=True,
            )
        return {
            "ok": True,
            "action": "stat_path",
            "path": str(path or "").strip() or None,
            "resolved_path": str(resolved),
            "exists": True,
            "type": self._path_type(resolved),
            "size": int(stat_result.st_size),
            "modified_time": float(stat_result.st_mtime),
            "readable": os.access(resolved, os.R_OK),
        }

    @staticmethod
    def _decode_text(data: bytes) -> tuple[str | None, str | None]:
        if b"\x00" in data:
            return None, None
        if data.startswith(b"\xef\xbb\xbf"):
            try:
                return data.decode("utf-8-sig"), "utf-8-sig"
            except UnicodeDecodeError:
                return None, None
        try:
            return data.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            return None, None

    @staticmethod
    def _search_snippet(text: str, match_index: int, query_len: int, *, radius: int = 60) -> tuple[str, int | None]:
        start = max(0, match_index - radius)
        end = min(len(text), match_index + query_len + radius)
        snippet = text[start:end].replace("\n", " ").strip()
        if start > 0:
            snippet = f"...{snippet}"
        if end < len(text):
            snippet = f"{snippet}..."
        line_number = text[:match_index].count("\n") + 1
        return snippet, line_number

    def _validated_search_root(self, root: str | None, *, action: str) -> tuple[Path | None, dict[str, Any] | None]:
        resolved, error = self._resolve_request_path(root or ".")
        if error is not None:
            return None, self._error_result(
                action,
                root or ".",
                resolved_path=error.get("resolved_path"),
                error_kind=str(error.get("error_kind") or "path_resolution_failed"),
                message=str(error.get("message") or "Search root access failed."),
            )
        assert resolved is not None
        if not resolved.exists():
            return None, self._error_result(
                action,
                root or ".",
                resolved_path=str(resolved),
                error_kind="not_found",
                message="That search root does not exist.",
                exists=False,
            )
        if not resolved.is_dir():
            return None, self._error_result(
                action,
                root or ".",
                resolved_path=str(resolved),
                error_kind="not_directory",
                message="That search root is not a directory.",
                exists=True,
            )
        return resolved, None

    def _walk_search_entries(self, root: Path, *, max_depth: int) -> list[tuple[Path, list[os.DirEntry[str]]]]:
        rows: list[tuple[Path, list[os.DirEntry[str]]]] = []
        pending: list[tuple[Path, int]] = [(root, 0)]
        while pending:
            current, depth = pending.pop(0)
            try:
                with os.scandir(current) as iterator:
                    entries = sorted(list(iterator), key=lambda item: item.name.lower())
            except (PermissionError, FileNotFoundError, NotADirectoryError, OSError):
                continue
            rows.append((current, entries))
            if depth >= max_depth:
                continue
            for entry in entries:
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    is_dir = False
                if not is_dir:
                    continue
                resolved_child, error = self._resolve_request_path(entry.path)
                if resolved_child is None or error is not None:
                    continue
                pending.append((resolved_child, depth + 1))
        return rows

    def search_filenames(
        self,
        root: str | None,
        query: str | None,
        *,
        max_results: int = _DEFAULT_SEARCH_RESULTS,
        max_depth: int = _DEFAULT_SEARCH_DEPTH,
    ) -> dict[str, Any]:
        resolved_root, error = self._validated_search_root(root, action="search_filenames")
        if error is not None:
            return error
        assert resolved_root is not None
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return self._error_result(
                "search_filenames",
                root or ".",
                resolved_path=str(resolved_root),
                error_kind="missing_query",
                message="I need a filename query for that search.",
                exists=True,
            )
        bounded_results = self._bounded_int(
            max_results,
            default=_DEFAULT_SEARCH_RESULTS,
            minimum=1,
            maximum=_HARD_SEARCH_RESULTS,
        )
        bounded_depth = self._bounded_int(
            max_depth,
            default=_DEFAULT_SEARCH_DEPTH,
            minimum=0,
            maximum=_HARD_SEARCH_DEPTH,
        )
        query_lower = normalized_query.lower()
        results: list[dict[str, Any]] = []
        truncated = False
        for _current, entries in self._walk_search_entries(resolved_root, max_depth=bounded_depth):
            for entry in entries:
                if len(results) >= bounded_results:
                    truncated = True
                    break
                if query_lower not in entry.name.lower():
                    continue
                resolved_entry, entry_error = self._resolve_request_path(entry.path)
                if resolved_entry is None or entry_error is not None:
                    continue
                entry_type = "other"
                try:
                    if entry.is_dir(follow_symlinks=False):
                        entry_type = "dir"
                    elif entry.is_file(follow_symlinks=False):
                        entry_type = "file"
                    elif entry.is_symlink():
                        entry_type = "symlink"
                except OSError:
                    entry_type = "other"
                results.append(
                    {
                        "path": str(resolved_entry),
                        "type": entry_type,
                    }
                )
            if truncated:
                break
        if not results:
            return {
                "ok": False,
                "action": "search_filenames",
                "root": str(root or ".").strip() or ".",
                "resolved_root": str(resolved_root),
                "query": normalized_query,
                "results": [],
                "truncated": False,
                "error_kind": "no_matches",
                "message": "I didn't find any matching file or directory names.",
            }
        return {
            "ok": True,
            "action": "search_filenames",
            "root": str(root or ".").strip() or ".",
            "resolved_root": str(resolved_root),
            "query": normalized_query,
            "results": results,
            "truncated": truncated,
            "stop_reason": "search_limit_reached" if truncated else None,
        }

    def search_text(
        self,
        root: str | None,
        query: str | None,
        *,
        max_results: int = _DEFAULT_SEARCH_RESULTS,
        max_files: int = _DEFAULT_SEARCH_MAX_FILES,
        max_bytes_per_file: int = _DEFAULT_SEARCH_MAX_BYTES_PER_FILE,
    ) -> dict[str, Any]:
        resolved_root, error = self._validated_search_root(root, action="search_text")
        if error is not None:
            return error
        assert resolved_root is not None
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return self._error_result(
                "search_text",
                root or ".",
                resolved_path=str(resolved_root),
                error_kind="missing_query",
                message="I need a text query for that search.",
                exists=True,
            )
        bounded_results = self._bounded_int(
            max_results,
            default=_DEFAULT_SEARCH_RESULTS,
            minimum=1,
            maximum=_HARD_SEARCH_RESULTS,
        )
        bounded_files = self._bounded_int(
            max_files,
            default=_DEFAULT_SEARCH_MAX_FILES,
            minimum=1,
            maximum=_HARD_SEARCH_MAX_FILES,
        )
        bounded_bytes = self._bounded_int(
            max_bytes_per_file,
            default=_DEFAULT_SEARCH_MAX_BYTES_PER_FILE,
            minimum=1,
            maximum=_HARD_READ_MAX_BYTES,
        )
        query_lower = normalized_query.lower()
        results: list[dict[str, Any]] = []
        files_examined = 0
        truncated = False
        for _current, entries in self._walk_search_entries(resolved_root, max_depth=_DEFAULT_SEARCH_DEPTH):
            for entry in entries:
                if len(results) >= bounded_results or files_examined >= bounded_files:
                    truncated = True
                    break
                try:
                    is_file = entry.is_file(follow_symlinks=False)
                except OSError:
                    is_file = False
                if not is_file:
                    continue
                resolved_entry, entry_error = self._resolve_request_path(entry.path)
                if resolved_entry is None or entry_error is not None:
                    continue
                files_examined += 1
                try:
                    with resolved_entry.open("rb") as handle:
                        raw = handle.read(bounded_bytes)
                except (PermissionError, OSError):
                    continue
                text, _encoding = self._decode_text(raw)
                if text is None:
                    continue
                match_index = text.lower().find(query_lower)
                if match_index < 0:
                    continue
                snippet, line_number = self._search_snippet(text, match_index, len(normalized_query))
                results.append(
                    {
                        "path": str(resolved_entry),
                        "snippet": snippet,
                        "line_number": line_number,
                    }
                )
            if truncated:
                break
        if not results:
            return {
                "ok": False,
                "action": "search_text",
                "root": str(root or ".").strip() or ".",
                "resolved_root": str(resolved_root),
                "query": normalized_query,
                "results": [],
                "files_examined": files_examined,
                "truncated": truncated,
                "error_kind": "no_matches",
                "message": "I didn't find any text matches for that query.",
                "stop_reason": "search_limit_reached" if truncated else None,
            }
        return {
            "ok": True,
            "action": "search_text",
            "root": str(root or ".").strip() or ".",
            "resolved_root": str(resolved_root),
            "query": normalized_query,
            "results": results,
            "files_examined": files_examined,
            "truncated": truncated,
            "stop_reason": "search_limit_reached" if truncated else None,
        }

    def read_text_file(
        self,
        path: str | None,
        *,
        max_bytes: int = _DEFAULT_READ_MAX_BYTES,
        offset: int = 0,
    ) -> dict[str, Any]:
        resolved, error = self._resolve_request_path(path)
        if error is not None:
            return self._error_result(
                "read_text_file",
                path,
                resolved_path=error.get("resolved_path"),
                error_kind=str(error.get("error_kind") or "path_resolution_failed"),
                message=str(error.get("message") or "File access failed."),
            )
        assert resolved is not None
        if not resolved.exists():
            return self._error_result(
                "read_text_file",
                path,
                resolved_path=str(resolved),
                error_kind="not_found",
                message="That file does not exist.",
                exists=False,
            )
        if not resolved.is_file():
            return self._error_result(
                "read_text_file",
                path,
                resolved_path=str(resolved),
                error_kind="not_file",
                message="That path is not a regular text file.",
                exists=True,
            )
        if not os.access(resolved, os.R_OK):
            return self._error_result(
                "read_text_file",
                path,
                resolved_path=str(resolved),
                error_kind="not_readable",
                message="That file is not readable.",
                exists=True,
            )
        bounded_max_bytes = max(1, min(int(max_bytes or _DEFAULT_READ_MAX_BYTES), _HARD_READ_MAX_BYTES))
        bounded_offset = max(0, int(offset or 0))
        try:
            stat_result = resolved.stat()
            with resolved.open("rb") as handle:
                handle.seek(bounded_offset)
                raw = handle.read(bounded_max_bytes)
        except PermissionError:
            return self._error_result(
                "read_text_file",
                path,
                resolved_path=str(resolved),
                error_kind="not_readable",
                message="That file is not readable.",
                exists=True,
            )
        text, encoding = self._decode_text(raw)
        if text is None or encoding is None:
            return self._error_result(
                "read_text_file",
                path,
                resolved_path=str(resolved),
                error_kind="binary_file_not_supported",
                message="That file looks binary, and this skill only supports text files.",
                exists=True,
            )
        bytes_read = len(raw)
        total_size = int(stat_result.st_size)
        truncated = bounded_offset + bytes_read < total_size
        return {
            "ok": True,
            "action": "read_text_file",
            "path": str(path or "").strip() or None,
            "resolved_path": str(resolved),
            "text": text,
            "encoding": encoding,
            "bytes_read": bytes_read,
            "offset": bounded_offset,
            "truncated": truncated,
            "total_size": total_size,
        }

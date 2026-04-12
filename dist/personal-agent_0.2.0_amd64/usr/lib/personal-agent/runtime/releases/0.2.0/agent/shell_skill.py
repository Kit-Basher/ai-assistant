from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any
import re

from agent.diagnostics import redact_secrets
from agent.filesystem_skill import FileSystemSkill


_DEFAULT_TIMEOUT_S = 2.0
_HARD_TIMEOUT_S = 10.0
_DEFAULT_OUTPUT_CHARS = 4000
_HARD_OUTPUT_CHARS = 16000

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+._-]{0,63}$")
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+._-]{0,127}$")
_APT_QUERY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+._-]{0,127}$")
_MODEL_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+._:/-]{0,127}$")

_DESTRUCTIVE_COMMANDS = {
    "rm",
    "sudo",
    "chmod",
    "chown",
    "systemctl",
    "curl",
    "wget",
}
_SUPPORTED_SAFE_COMMANDS = {
    "pwd",
    "uname",
    "which",
    "python_version",
    "pip_version",
    "apt_search",
    "apt_cache_policy",
    "ollama_list",
    "ollama_ps",
    "ollama_show",
}
_SUPPORTED_INSTALL_MANAGERS = {"apt", "pip"}
_SUPPORTED_PYTHON_PACKAGE_SCOPES = {"user", "venv"}


class ShellSkill(FileSystemSkill):
    """Bounded shell-backed skill with an explicit allowlist and no arbitrary shell."""

    def _result(
        self,
        action: str,
        *,
        ok: bool,
        mutated: bool,
        command_name: str | None = None,
        argv: list[str] | None = None,
        cwd: str | None = None,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        timed_out: bool = False,
        truncated: bool = False,
        blocked_reason: str | None = None,
        error_kind: str | None = None,
        message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": bool(ok),
            "action": action,
            "mutated": bool(mutated),
            "command_name": str(command_name or "").strip() or None,
            "argv": [str(item) for item in (argv or []) if str(item)],
            "cwd": str(cwd or "").strip() or None,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timed_out": bool(timed_out),
            "truncated": bool(truncated),
            "blocked_reason": str(blocked_reason or "").strip() or None,
            "error_kind": str(error_kind or "").strip() or None,
            "message": str(message or "").strip() or None,
        }
        if isinstance(extra, dict):
            payload.update(extra)
        return payload

    def _blocked_result(
        self,
        action: str,
        *,
        blocked_reason: str,
        message: str,
        mutated: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._result(
            action,
            ok=False,
            mutated=mutated,
            blocked_reason=blocked_reason,
            error_kind=blocked_reason,
            message=message,
            extra=extra,
        )

    @staticmethod
    def _validated_name(value: str | None, *, pattern: re.Pattern[str], field_name: str) -> str | None:
        candidate = str(value or "").strip()
        if not candidate:
            return None
        if pattern.fullmatch(candidate) is None:
            return None
        return candidate

    @staticmethod
    def _truncate_output(text: str, max_chars: int) -> tuple[str, bool]:
        if len(text) <= max_chars:
            return text, False
        clipped = text[:max_chars].rstrip()
        return clipped, True

    def _resolve_cwd(self, cwd: str | None) -> tuple[Path | None, dict[str, Any] | None]:
        resolved, error = self._resolve_request_path(cwd or ".")
        if error is not None:
            return None, self._blocked_result(
                "execute_safe_command",
                blocked_reason=str(error.get("error_kind") or "path_resolution_failed"),
                message=str(error.get("message") or "Command cwd access failed."),
                extra={"cwd": cwd, "resolved_path": error.get("resolved_path")},
            )
        assert resolved is not None
        if not resolved.exists():
            return None, self._blocked_result(
                "execute_safe_command",
                blocked_reason="not_found",
                message="That working directory does not exist.",
                extra={"cwd": cwd, "resolved_path": str(resolved)},
            )
        if not resolved.is_dir():
            return None, self._blocked_result(
                "execute_safe_command",
                blocked_reason="not_directory",
                message="That working directory is not a directory.",
                extra={"cwd": cwd, "resolved_path": str(resolved)},
            )
        return resolved, None

    def _build_safe_command(
        self,
        *,
        command_name: str | None,
        subject: str | None,
        query: str | None,
        cwd: str | None,
    ) -> tuple[str | None, list[str] | None, Path | None, dict[str, Any], dict[str, Any] | None]:
        normalized_name = str(command_name or "").strip().lower()
        if not normalized_name:
            return None, None, None, {}, self._blocked_result(
                "execute_safe_command",
                blocked_reason="unsupported_command",
                message="I need a supported command name for that shell request.",
            )
        if normalized_name in _DESTRUCTIVE_COMMANDS:
            return None, None, None, {}, self._blocked_result(
                "execute_safe_command",
                blocked_reason="destructive_operation_blocked",
                message="Destructive or privilege-changing shell operations are blocked in this skill.",
                extra={"command_name": normalized_name},
            )
        if normalized_name not in _SUPPORTED_SAFE_COMMANDS:
            return None, None, None, {}, self._blocked_result(
                "execute_safe_command",
                blocked_reason="unsupported_command",
                message="That shell request is outside the small allowlisted command set.",
                extra={"command_name": normalized_name},
            )

        resolved_cwd, cwd_error = self._resolve_cwd(cwd)
        if cwd_error is not None:
            return None, None, None, {}, {
                **cwd_error,
                "action": "execute_safe_command",
                "command_name": normalized_name,
            }
        assert resolved_cwd is not None

        argv: list[str]
        extra: dict[str, Any] = {}
        if normalized_name == "pwd":
            argv = ["pwd"]
        elif normalized_name == "uname":
            argv = ["uname", "-a"]
        elif normalized_name == "python_version":
            argv = ["python", "--version"]
        elif normalized_name == "pip_version":
            argv = ["pip", "--version"]
        elif normalized_name == "which":
            tool_name = self._validated_name(subject, pattern=_TOOL_NAME_RE, field_name="tool")
            if tool_name is None:
                return None, None, None, {}, self._blocked_result(
                    "execute_safe_command",
                    blocked_reason="invalid_argument",
                    message="I need a simple executable name for a which lookup.",
                    extra={"command_name": normalized_name, "subject": subject},
                )
            argv = ["which", tool_name]
            extra["subject"] = tool_name
        elif normalized_name == "apt_search":
            apt_query = self._validated_name(query, pattern=_APT_QUERY_RE, field_name="query")
            if apt_query is None:
                return None, None, None, {}, self._blocked_result(
                    "execute_safe_command",
                    blocked_reason="invalid_argument",
                    message="I need a simple apt search term for that request.",
                    extra={"command_name": normalized_name, "query": query},
                )
            argv = ["apt-cache", "search", "--names-only", apt_query]
            extra["query"] = apt_query
        elif normalized_name == "apt_cache_policy":
            package_name = self._validated_name(subject, pattern=_PACKAGE_NAME_RE, field_name="package")
            if package_name is None:
                return None, None, None, {}, self._blocked_result(
                    "execute_safe_command",
                    blocked_reason="invalid_argument",
                    message="I need a valid package name for that apt policy lookup.",
                    extra={"command_name": normalized_name, "subject": subject},
                )
            argv = ["apt-cache", "policy", package_name]
            extra["subject"] = package_name
        elif normalized_name == "ollama_list":
            argv = ["ollama", "list"]
        elif normalized_name == "ollama_ps":
            argv = ["ollama", "ps"]
        elif normalized_name == "ollama_show":
            model_ref = self._validated_name(subject, pattern=_MODEL_REF_RE, field_name="model")
            if model_ref is None:
                return None, None, None, {}, self._blocked_result(
                    "execute_safe_command",
                    blocked_reason="invalid_argument",
                    message="I need a valid Ollama model reference for that request.",
                    extra={"command_name": normalized_name, "subject": subject},
                )
            argv = ["ollama", "show", model_ref]
            extra["subject"] = model_ref
        else:
            return None, None, None, {}, self._blocked_result(
                "execute_safe_command",
                blocked_reason="unsupported_command",
                message="That shell request is outside the small allowlisted command set.",
                extra={"command_name": normalized_name},
            )
        return normalized_name, argv, resolved_cwd, extra, None

    def _build_install_command(
        self,
        *,
        manager: str | None,
        package: str | None,
        scope: str | None,
        dry_run: bool,
        cwd: str | None,
    ) -> tuple[str | None, str | None, str | None, bool, list[str] | None, Path | None, dict[str, Any] | None]:
        normalized_manager = str(manager or "").strip().lower()
        normalized_package = self._validated_name(package, pattern=_PACKAGE_NAME_RE, field_name="package")
        normalized_scope = str(scope or "").strip().lower() or None
        if normalized_manager not in _SUPPORTED_INSTALL_MANAGERS:
            return None, None, None, bool(dry_run), None, None, self._blocked_result(
                "install_package",
                blocked_reason="unsupported_manager",
                message="That install manager is not supported by this bounded shell skill.",
                extra={"manager": manager, "package": package},
            )
        if normalized_package is None:
            return None, None, None, bool(dry_run), None, None, self._blocked_result(
                "install_package",
                blocked_reason="invalid_package_name",
                message="I need a valid package name for that install request.",
                extra={"manager": normalized_manager, "package": package},
            )
        resolved_cwd, cwd_error = self._resolve_cwd(cwd)
        if cwd_error is not None:
            return None, None, None, bool(dry_run), None, None, {
                **cwd_error,
                "action": "install_package",
                "manager": normalized_manager,
                "package": normalized_package,
            }
        assert resolved_cwd is not None

        if normalized_manager == "apt":
            argv = ["apt-get"]
            if dry_run:
                argv.append("-s")
            argv.extend(["install", "-y", normalized_package])
        else:
            if normalized_scope is None:
                normalized_scope = "user"
            if normalized_scope not in _SUPPORTED_PYTHON_PACKAGE_SCOPES:
                return None, None, None, bool(dry_run), None, None, self._blocked_result(
                    "install_package",
                    blocked_reason="unsupported_scope",
                    message="Python package installs only support the user or venv scopes in this skill.",
                    extra={
                        "manager": normalized_manager,
                        "package": normalized_package,
                        "scope": normalized_scope,
                    },
                )
            argv = ["python", "-m", "pip", "install"]
            if dry_run:
                argv.append("--dry-run")
            if normalized_scope == "user":
                argv.append("--user")
            argv.append(normalized_package)
        return normalized_manager, normalized_package, normalized_scope, bool(dry_run), argv, resolved_cwd, None

    def preview_install_package(
        self,
        *,
        manager: str | None,
        package: str | None,
        scope: str | None = None,
        dry_run: bool = False,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        normalized_manager, normalized_package, normalized_scope, preview_dry_run, argv, resolved_cwd, error = (
            self._build_install_command(
                manager=manager,
                package=package,
                scope=scope,
                dry_run=dry_run,
                cwd=cwd,
            )
        )
        if error is not None:
            return error
        assert argv is not None
        return self._result(
            "install_package_preview",
            ok=True,
            mutated=not preview_dry_run,
            command_name=f"{normalized_manager}_install",
            argv=argv,
            cwd=str(resolved_cwd),
            message="Install preview ready.",
            extra={
                "manager": normalized_manager,
                "package": normalized_package,
                "scope": normalized_scope,
                "dry_run": preview_dry_run,
            },
        )

    def preview_create_directory(self, path: str | None) -> dict[str, Any]:
        resolved, error = self._resolve_request_path(path)
        if error is not None:
            return self._blocked_result(
                "create_directory_preview",
                blocked_reason=str(error.get("error_kind") or "path_resolution_failed"),
                message=str(error.get("message") or "Directory creation failed."),
                extra={
                    "path": str(path or "").strip() or None,
                    "resolved_path": error.get("resolved_path"),
                },
            )
        assert resolved is not None
        if resolved.exists():
            if resolved.is_dir():
                return self._result(
                    "create_directory_preview",
                    ok=True,
                    mutated=False,
                    message="That directory already exists.",
                    extra={
                        "path": str(path or "").strip() or None,
                        "resolved_path": str(resolved),
                        "created": False,
                    },
                )
            return self._blocked_result(
                "create_directory_preview",
                blocked_reason="path_exists",
                message="A non-directory path already exists there.",
                extra={
                    "path": str(path or "").strip() or None,
                    "resolved_path": str(resolved),
                },
            )
        parent = resolved.parent
        if not parent.exists() or not parent.is_dir():
            return self._blocked_result(
                "create_directory_preview",
                blocked_reason="parent_not_directory",
                message="The parent directory does not exist or is not a directory.",
                extra={
                    "path": str(path or "").strip() or None,
                    "resolved_path": str(resolved),
                },
            )
        if any(parent == sensitive or str(parent).startswith(f"{sensitive}{os.sep}") for sensitive in self.sensitive_roots):
            return self._blocked_result(
                "create_directory_preview",
                blocked_reason="sensitive_path_blocked",
                message="That path is blocked by the local privacy policy.",
                extra={
                    "path": str(path or "").strip() or None,
                    "resolved_path": str(resolved),
                },
            )
        return self._result(
            "create_directory_preview",
            ok=True,
            mutated=True,
            message="Directory creation preview ready.",
            extra={
                "path": str(path or "").strip() or None,
                "resolved_path": str(resolved),
                "created": False,
            },
        )

    def _run_command(
        self,
        *,
        action: str,
        command_name: str,
        argv: list[str],
        cwd: Path | None,
        timeout_s: float,
        max_output_chars: int,
        mutated: bool,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bounded_timeout = max(0.1, min(float(timeout_s or _DEFAULT_TIMEOUT_S), _HARD_TIMEOUT_S))
        bounded_output = max(128, min(int(max_output_chars or _DEFAULT_OUTPUT_CHARS), _HARD_OUTPUT_CHARS))
        try:
            completed = subprocess.run(
                list(argv),
                cwd=str(cwd) if cwd is not None else None,
                capture_output=True,
                text=True,
                timeout=bounded_timeout,
                check=False,
            )
            raw_stdout = completed.stdout or ""
            raw_stderr = completed.stderr or ""
            exit_code = completed.returncode
            timed_out = False
            error_kind = None if exit_code == 0 else "command_failed"
            message = "Command completed." if exit_code == 0 else "The command exited with an error."
        except FileNotFoundError:
            raw_stdout = ""
            raw_stderr = ""
            exit_code = None
            timed_out = False
            error_kind = "command_not_available"
            message = "That command is not available in this environment."
        except PermissionError:
            raw_stdout = ""
            raw_stderr = ""
            exit_code = None
            timed_out = False
            error_kind = "permission_denied"
            message = "That command needs more privileges than this skill can use."
        except subprocess.TimeoutExpired as exc:
            raw_stdout = exc.stdout or ""
            raw_stderr = exc.stderr or ""
            exit_code = None
            timed_out = True
            error_kind = "timeout"
            message = "The command timed out."

        safe_stdout = redact_secrets(raw_stdout)
        safe_stderr = redact_secrets(raw_stderr)
        stdout, stdout_truncated = self._truncate_output(safe_stdout, bounded_output)
        stderr, stderr_truncated = self._truncate_output(safe_stderr, bounded_output)
        truncated = stdout_truncated or stderr_truncated
        ok = error_kind is None
        return self._result(
            action,
            ok=ok,
            mutated=mutated,
            command_name=command_name,
            argv=argv,
            cwd=str(cwd) if cwd is not None else None,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            truncated=truncated,
            error_kind=error_kind,
            message=message,
            extra=extra,
        )

    def execute_safe_command(
        self,
        command_name: str | None,
        *,
        subject: str | None = None,
        query: str | None = None,
        cwd: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_output_chars: int = _DEFAULT_OUTPUT_CHARS,
    ) -> dict[str, Any]:
        normalized_name, argv, resolved_cwd, extra, error = self._build_safe_command(
            command_name=command_name,
            subject=subject,
            query=query,
            cwd=cwd,
        )
        if error is not None:
            return error
        assert normalized_name is not None and argv is not None

        return self._run_command(
            action="execute_safe_command",
            command_name=normalized_name,
            argv=argv,
            cwd=resolved_cwd,
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
            mutated=False,
            extra=extra,
        )

    def install_package(
        self,
        *,
        manager: str | None,
        package: str | None,
        scope: str | None = None,
        dry_run: bool = False,
        cwd: str | None = None,
        timeout_s: float = 10.0,
        max_output_chars: int = _DEFAULT_OUTPUT_CHARS,
    ) -> dict[str, Any]:
        normalized_manager, normalized_package, normalized_scope, preview_dry_run, argv, resolved_cwd, error = (
            self._build_install_command(
                manager=manager,
                package=package,
                scope=scope,
                dry_run=dry_run,
                cwd=cwd,
            )
        )
        if error is not None:
            return error
        assert normalized_manager is not None and normalized_package is not None and argv is not None and resolved_cwd is not None

        return self._run_command(
            action="install_package",
            command_name=f"{normalized_manager}_install",
            argv=argv,
            cwd=resolved_cwd,
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
            mutated=not preview_dry_run,
            extra={
                "manager": normalized_manager,
                "package": normalized_package,
                "scope": normalized_scope,
                "dry_run": preview_dry_run,
            },
        )

    def create_directory(self, path: str | None) -> dict[str, Any]:
        preview = self.preview_create_directory(path)
        if not bool(preview.get("ok", False)):
            return {
                **preview,
                "action": "create_directory",
            }
        resolved_path = str(preview.get("resolved_path") or "").strip() or None
        if not resolved_path:
            return self._blocked_result(
                "create_directory",
                blocked_reason="path_resolution_failed",
                message="Directory creation failed.",
                extra={"path": str(path or "").strip() or None},
            )
        resolved = Path(resolved_path)
        if not bool(preview.get("mutated", False)):
            return self._result(
                "create_directory",
                ok=True,
                mutated=False,
                message="That directory already exists.",
                extra={
                    "path": str(path or "").strip() or None,
                    "resolved_path": str(resolved),
                    "created": False,
                },
            )
        try:
            resolved.mkdir(parents=True, exist_ok=False)
        except PermissionError:
            return self._blocked_result(
                "create_directory",
                blocked_reason="not_writable",
                message="That directory could not be created because the parent path is not writable.",
                mutated=False,
                extra={
                    "path": str(path or "").strip() or None,
                    "resolved_path": str(resolved),
                },
            )
        return self._result(
            "create_directory",
            ok=True,
            mutated=True,
            message="Directory created.",
            extra={
                "path": str(path or "").strip() or None,
                "resolved_path": str(resolved),
                "created": True,
            },
        )

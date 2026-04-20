from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parent
_DEFAULT_TIMESTAMP = int(os.getenv("SOURCE_DATE_EPOCH", "1704067200"))


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _project_table() -> dict[str, object]:
    project = _load_pyproject().get("project")
    if not isinstance(project, dict):
        raise RuntimeError("pyproject.toml is missing [project]")
    return project


def _tool_table() -> dict[str, object]:
    tool = _load_pyproject().get("tool")
    if not isinstance(tool, dict):
        return {}
    config = tool.get("personal_agent_build")
    return config if isinstance(config, dict) else {}


def _version() -> str:
    text = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("VERSION is missing or empty")
    return text


def _distribution_name() -> str:
    name = str(_project_table().get("name") or "").strip()
    if not name:
        raise RuntimeError("pyproject.toml [project].name is required")
    return name


def _normalized_distribution_name() -> str:
    return _distribution_name().replace("-", "_")


def _wheel_name() -> str:
    return f"{_normalized_distribution_name()}-{_version()}-py3-none-any.whl"


def _sdist_name() -> str:
    return f"{_normalized_distribution_name()}-{_version()}.tar.gz"


def _dist_info_dir() -> str:
    return f"{_normalized_distribution_name()}-{_version()}.dist-info"


def _source_packages() -> list[str]:
    configured = _tool_table().get("source_packages")
    if isinstance(configured, list):
        values = [str(item).strip() for item in configured if str(item).strip()]
        if values:
            return values
    return ["agent", "telegram_adapter"]


def _sdist_includes() -> list[str]:
    configured = _tool_table().get("sdist_include")
    if isinstance(configured, list):
        values = [str(item).strip() for item in configured if str(item).strip()]
        if values:
            return values
    return [
        "agent",
        "telegram_adapter",
        "tests",
        "systemd",
        "scripts/build_dist.py",
        "README.md",
        "VERSION",
        "requirements.txt",
        "pyproject.toml",
        "build_backend.py",
    ]


def _wheel_data_files() -> list[tuple[str, str]]:
    configured = _tool_table().get("wheel_data_files")
    rows: list[tuple[str, str]] = []
    if isinstance(configured, list):
        for item in configured:
            if isinstance(item, list | tuple) and len(item) == 2:
                source = str(item[0]).strip()
                target = str(item[1]).strip()
                if source and target:
                    rows.append((source, target))
    if rows:
        return rows
    return [
        ("systemd/personal-agent-api.service", "share/personal-agent/systemd/personal-agent-api.service"),
    ]


def _generated_wheel_package_files() -> list[tuple[str, bytes]]:
    rows: list[tuple[str, bytes]] = []
    version_text = (ROOT / "VERSION").read_text(encoding="utf-8")
    rows.append(("agent/VERSION", version_text.encode("utf-8")))
    return rows


def _iter_source_files() -> list[Path]:
    rows: list[Path] = []
    root_files = [
        ROOT / "personal_agent_bootstrap.py",
        ROOT / "personal_agent_bootstrap.pth",
        ROOT / "sitecustomize.py",
    ]
    for path in root_files:
        if path.is_file():
            rows.append(path.relative_to(ROOT))
    for package_name in _source_packages():
        root = ROOT / package_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if "__pycache__" in rel.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if rel.name in {"agent.db", "llm_usage_stats.json"}:
                continue
            rows.append(rel)
    return rows


def _iter_sdist_files() -> list[Path]:
    rows: list[Path] = []
    root_files = [
        ROOT / "personal_agent_bootstrap.py",
        ROOT / "personal_agent_bootstrap.pth",
        ROOT / "sitecustomize.py",
    ]
    for path in root_files:
        if path.exists():
            rows.append(path.relative_to(ROOT))
    for item in _sdist_includes():
        target = ROOT / item
        if not target.exists():
            continue
        if target.is_file():
            rows.append(target.relative_to(ROOT))
            continue
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if "__pycache__" in rel.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            rows.append(rel)
    deduped: dict[str, Path] = {}
    for row in rows:
        deduped[row.as_posix()] = row
    return [deduped[key] for key in sorted(deduped)]


def _metadata_text() -> str:
    project = _project_table()
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {_distribution_name()}",
        f"Version: {_version()}",
    ]
    summary = str(project.get("description") or "").strip()
    if summary:
        lines.append(f"Summary: {summary}")
    readme = project.get("readme")
    if isinstance(readme, str) and readme.strip().endswith(".md"):
        lines.append("Description-Content-Type: text/markdown")
    requires_python = str(project.get("requires-python") or "").strip()
    if requires_python:
        lines.append(f"Requires-Python: {requires_python}")
    for dependency in project.get("dependencies", []) if isinstance(project.get("dependencies"), list) else []:
        dep = str(dependency).strip()
        if dep:
            lines.append(f"Requires-Dist: {dep}")
    urls = project.get("urls")
    if isinstance(urls, dict):
        for key in sorted(urls):
            value = str(urls.get(key) or "").strip()
            if value:
                lines.append(f"Project-URL: {key}, {value}")
    description = ""
    readme_value = project.get("readme")
    if isinstance(readme_value, str) and readme_value.strip():
        description = (ROOT / readme_value).read_text(encoding="utf-8")
    return "\n".join(lines) + "\n\n" + description


def _wheel_text() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: personal-agent build backend",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _entry_points_text() -> str:
    scripts = _project_table().get("scripts")
    if not isinstance(scripts, dict) or not scripts:
        return ""
    lines = ["[console_scripts]"]
    for name in sorted(scripts):
        target = str(scripts.get(name) or "").strip()
        if target:
            lines.append(f"{name} = {target}")
    lines.append("")
    return "\n".join(lines)


def _sha256_digest(payload: bytes) -> tuple[str, int]:
    digest = hashlib.sha256(payload).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"sha256={encoded}", len(payload)


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path)
    info.date_time = tuple(time.gmtime(_DEFAULT_TIMESTAMP)[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    return info


def _write_zip_entry(
    zf: zipfile.ZipFile,
    *,
    path: str,
    payload: bytes,
    records: list[tuple[str, str, int]],
) -> None:
    zf.writestr(_zip_info(path), payload)
    digest, size = _sha256_digest(payload)
    records.append((path, digest, size))


def _build_dist_info_entries() -> list[tuple[str, bytes]]:
    entries = [
        (f"{_dist_info_dir()}/METADATA", _metadata_text().encode("utf-8")),
        (f"{_dist_info_dir()}/WHEEL", _wheel_text().encode("utf-8")),
    ]
    entry_points = _entry_points_text()
    if entry_points:
        entries.append((f"{_dist_info_dir()}/entry_points.txt", entry_points.encode("utf-8")))
    return entries


def _build_standard_wheel(wheel_directory: str) -> str:
    outdir = Path(wheel_directory)
    outdir.mkdir(parents=True, exist_ok=True)
    wheel_path = outdir / _wheel_name()
    records: list[tuple[str, str, int]] = []
    with zipfile.ZipFile(wheel_path, "w") as zf:
        for rel in _iter_source_files():
            _write_zip_entry(
                zf,
                path=rel.as_posix(),
                payload=(ROOT / rel).read_bytes(),
                records=records,
            )
        for rel_path, payload in _generated_wheel_package_files():
            _write_zip_entry(zf, path=rel_path, payload=payload, records=records)
        for source, target in _wheel_data_files():
            source_path = ROOT / source
            if not source_path.is_file():
                continue
            target_path = f"{_normalized_distribution_name()}-{_version()}.data/data/{PurePosixPath(target).as_posix()}"
            _write_zip_entry(
                zf,
                path=target_path,
                payload=source_path.read_bytes(),
                records=records,
            )
        for rel_path, payload in _build_dist_info_entries():
            _write_zip_entry(zf, path=rel_path, payload=payload, records=records)
        record_path = f"{_dist_info_dir()}/RECORD"
        record_rows = records + [(record_path, "", 0)]
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        for row in record_rows:
            writer.writerow(row)
        zf.writestr(_zip_info(record_path), buffer.getvalue().encode("utf-8"))
    return wheel_path.name


def _build_editable_wheel(wheel_directory: str) -> str:
    outdir = Path(wheel_directory)
    outdir.mkdir(parents=True, exist_ok=True)
    wheel_path = outdir / _wheel_name()
    records: list[tuple[str, str, int]] = []
    with zipfile.ZipFile(wheel_path, "w") as zf:
        pth_name = f"{_normalized_distribution_name()}-editable.pth"
        pth_path = f"{_normalized_distribution_name()}-{_version()}.data/purelib/{pth_name}"
        _write_zip_entry(
            zf,
            path=pth_path,
            payload=(str(ROOT.resolve()) + "\n").encode("utf-8"),
            records=records,
        )
        for rel_path, payload in _build_dist_info_entries():
            _write_zip_entry(zf, path=rel_path, payload=payload, records=records)
        record_path = f"{_dist_info_dir()}/RECORD"
        record_rows = records + [(record_path, "", 0)]
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        for row in record_rows:
            writer.writerow(row)
        zf.writestr(_zip_info(record_path), buffer.getvalue().encode("utf-8"))
    return wheel_path.name


def _build_metadata_dir(metadata_directory: str) -> str:
    metadata_root = Path(metadata_directory)
    metadata_root.mkdir(parents=True, exist_ok=True)
    dist_info_name = _dist_info_dir()
    dist_info_root = metadata_root / dist_info_name
    dist_info_root.mkdir(parents=True, exist_ok=True)
    for rel_path, payload in _build_dist_info_entries():
        leaf = rel_path.split("/", 1)[1]
        (dist_info_root / leaf).write_bytes(payload)
    return dist_info_name


def _tar_info(path: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(path)
    info.size = size
    info.mode = 0o644
    info.mtime = _DEFAULT_TIMESTAMP
    return info


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ = (config_settings, metadata_directory)
    return _build_standard_wheel(wheel_directory)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ = (config_settings, metadata_directory)
    return _build_editable_wheel(wheel_directory)


def build_sdist(sdist_directory: str, config_settings: dict[str, object] | None = None) -> str:
    _ = config_settings
    outdir = Path(sdist_directory)
    outdir.mkdir(parents=True, exist_ok=True)
    sdist_path = outdir / _sdist_name()
    root_prefix = f"{_normalized_distribution_name()}-{_version()}"
    with sdist_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=_DEFAULT_TIMESTAMP) as gz_handle:
            with tarfile.open(fileobj=gz_handle, mode="w:") as tf:
                for rel in _iter_sdist_files():
                    payload = (ROOT / rel).read_bytes()
                    archive_path = f"{root_prefix}/{rel.as_posix()}"
                    info = _tar_info(archive_path, len(payload))
                    tf.addfile(info, io.BytesIO(payload))
    return sdist_path.name


def get_requires_for_build_wheel(config_settings: dict[str, object] | None = None) -> list[str]:
    _ = config_settings
    return []


def get_requires_for_build_editable(config_settings: dict[str, object] | None = None) -> list[str]:
    _ = config_settings
    return []


def get_requires_for_build_sdist(config_settings: dict[str, object] | None = None) -> list[str]:
    _ = config_settings
    return []


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, object] | None = None,
) -> str:
    _ = config_settings
    return _build_metadata_dir(metadata_directory)


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, object] | None = None,
) -> str:
    _ = config_settings
    return _build_metadata_dir(metadata_directory)

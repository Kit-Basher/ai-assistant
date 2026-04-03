from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from agent.packs.manifest import (
    PackManifestError,
    compute_permissions_hash,
    load_manifest,
    manifest_to_dict,
    normalize_permissions,
)
from agent.skill_governance import (
    SkillExecutionRequest,
    parse_skill_execution_request,
    scan_skill_source_for_persistence,
)


@dataclass
class SkillFunction:
    name: str
    args_schema: dict[str, Any]
    handler: Callable[..., Any]
    read_only: bool = False


@dataclass
class Skill:
    name: str
    description: str
    version: str
    permissions: list[str]
    functions: dict[str, SkillFunction]
    skill_type: str = "general"
    execution_request: SkillExecutionRequest | None = None
    governance_source_issues: tuple[str, ...] = ()
    source_path: str | None = None
    pack_id: str = ""
    pack_trust: str = "native"
    pack_permissions: dict[str, Any] | None = None
    pack_permissions_hash: str = ""
    pack_manifest: dict[str, Any] | None = None
    pack_manifest_path: str | None = None


class SkillLoader:
    def __init__(self, skills_path: str) -> None:
        self.skills_path = skills_path
        self.blocked_skills: list[dict[str, Any]] = []

    def load_all(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        self.blocked_skills = []
        if not os.path.isdir(self.skills_path):
            return skills

        for entry in sorted(os.listdir(self.skills_path)):
            skill_dir = os.path.join(self.skills_path, entry)
            manifest_path = os.path.join(skill_dir, "manifest.json")
            handler_path = os.path.join(skill_dir, "handler.py")
            if not os.path.isfile(manifest_path) or not os.path.isfile(handler_path):
                continue

            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            execution_request = parse_skill_execution_request(manifest, skill_id=str(manifest.get("name") or entry))
            with open(handler_path, "r", encoding="utf-8") as handle:
                handler_source = handle.read()
            governance_source_issues = scan_skill_source_for_persistence(handler_source)
            if governance_source_issues:
                self.blocked_skills.append(
                    {
                        "skill_id": execution_request.skill_id,
                        "skill_type": execution_request.skill_type,
                        "requested_execution_mode": execution_request.requested_execution_mode,
                        "requested_capabilities": list(execution_request.requested_capabilities),
                        "persistence_requested": bool(execution_request.persistence_requested),
                        "reason": "forbidden_persistence_pattern",
                        "source_issues": list(governance_source_issues),
                        "source_pack": str(manifest.get("name") or entry).strip() or entry,
                        "source_path": handler_path,
                    }
                )
                continue

            spec = importlib.util.spec_from_file_location(f"skills.{entry}.handler", handler_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            functions = {}
            for func in manifest.get("functions", []):
                func_name = func["name"]
                handler = getattr(module, func_name, None)
                if handler is None:
                    continue
                functions[func_name] = SkillFunction(
                    name=func_name,
                    args_schema=func.get("args_schema", {}),
                    handler=handler,
                    read_only=bool(func.get("read_only", False)),
                )

            pack_manifest_path = os.path.join(skill_dir, "pack.json")
            synthesized_permissions = normalize_permissions({"ifaces": sorted(functions.keys())})
            pack_manifest_dict: dict[str, Any] = {
                "pack_id": str(manifest.get("name") or "").strip() or entry,
                "version": str(manifest.get("version") or "0.1.0"),
                "title": str(manifest.get("name") or "").strip() or entry,
                "description": str(manifest.get("description") or "").strip(),
                "entrypoints": [f"skills.{entry}.handler"],
                "trust": "native",
                "permissions": synthesized_permissions,
            }
            if os.path.isfile(pack_manifest_path):
                try:
                    parsed_pack = load_manifest(pack_manifest_path)
                except PackManifestError:
                    continue
                pack_manifest_dict = manifest_to_dict(parsed_pack)
            pack_permissions = normalize_permissions(pack_manifest_dict.get("permissions"))
            pack_permissions_hash = compute_permissions_hash(pack_permissions)

            skill = Skill(
                name=manifest["name"],
                description=manifest.get("description", ""),
                version=manifest.get("version", "0.1.0"),
                permissions=manifest.get("permissions", []),
                functions=functions,
                skill_type=execution_request.skill_type,
                execution_request=execution_request,
                governance_source_issues=governance_source_issues,
                source_path=handler_path,
                pack_id=str(pack_manifest_dict.get("pack_id") or manifest["name"]),
                pack_trust=str(pack_manifest_dict.get("trust") or "native"),
                pack_permissions=pack_permissions,
                pack_permissions_hash=pack_permissions_hash,
                pack_manifest=pack_manifest_dict,
                pack_manifest_path=pack_manifest_path if os.path.isfile(pack_manifest_path) else None,
            )
            skills[skill.name] = skill

        return skills

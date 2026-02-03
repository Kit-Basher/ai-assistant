from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SkillFunction:
    name: str
    args_schema: dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class Skill:
    name: str
    description: str
    version: str
    permissions: list[str]
    functions: dict[str, SkillFunction]


class SkillLoader:
    def __init__(self, skills_path: str) -> None:
        self.skills_path = skills_path

    def load_all(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        if not os.path.isdir(self.skills_path):
            return skills

        for entry in os.listdir(self.skills_path):
            skill_dir = os.path.join(self.skills_path, entry)
            manifest_path = os.path.join(skill_dir, "manifest.json")
            handler_path = os.path.join(skill_dir, "handler.py")
            if not os.path.isfile(manifest_path) or not os.path.isfile(handler_path):
                continue

            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

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
                )

            skill = Skill(
                name=manifest["name"],
                description=manifest.get("description", ""),
                version=manifest.get("version", "0.1.0"),
                permissions=manifest.get("permissions", []),
                functions=functions,
            )
            skills[skill.name] = skill

        return skills

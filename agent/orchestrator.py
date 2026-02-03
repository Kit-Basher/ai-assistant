from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.intent_router import route_message
from agent.commands import parse_command, split_pipe_args
from agent.confirmations import ConfirmationStore, PendingAction
from agent.logging_utils import log_event
from agent.policy import evaluate_policy
from agent.skills_loader import SkillLoader
from memory.db import MemoryDB


@dataclass
class OrchestratorResponse:
    text: str
    data: dict[str, Any] | None = None


class Orchestrator:
    def __init__(self, db: MemoryDB, skills_path: str, log_path: str, timezone: str, llm_client: Any) -> None:
        self.db = db
        self.skills = SkillLoader(skills_path).load_all()
        self.log_path = log_path
        self.timezone = timezone
        self.llm_client = llm_client
        self.confirmations = ConfirmationStore()

    def _context(self) -> dict[str, Any]:
        return {"db": self.db, "timezone": self.timezone}

    def _call_skill(
        self,
        user_id: str,
        skill_name: str,
        function_name: str,
        args: dict[str, Any],
        requested_permissions: list[str],
        action_type: str | None = None,
    ) -> OrchestratorResponse:
        skill = self.skills.get(skill_name)
        if not skill:
            return OrchestratorResponse("Skill not found.")

        func = skill.functions.get(function_name)
        if not func:
            return OrchestratorResponse("Function not found.")

        action = {"action_type": action_type or ""}
        decision = evaluate_policy(skill.permissions, requested_permissions, action)
        if not decision.allowed:
            return OrchestratorResponse("Permission denied.")

        if decision.requires_confirmation:
            pending = PendingAction(
                user_id=user_id,
                action={
                    "skill": skill_name,
                    "function": function_name,
                    "args": args,
                    "requested_permissions": requested_permissions,
                    "action_type": action_type,
                },
                message="This will delete or overwrite data. Reply /confirm to proceed.",
            )
            self.confirmations.set(pending)
            return OrchestratorResponse(pending.message)

        result = func.handler(self._context(), **args)
        log_event(self.log_path, "skill_call", {"skill": skill_name, "function": function_name})
        return OrchestratorResponse("OK", result)

    def _format_projects(self, projects: list[dict[str, Any]]) -> str:
        if not projects:
            return "No projects yet. Use /project_new to add one."
        lines = ["Projects:"]
        for proj in projects:
            status = proj.get("status", "")
            pitch = proj.get("pitch") or ""
            suffix = f" - {pitch}" if pitch else ""
            lines.append(f"- {proj['name']} ({status}){suffix}")
        return "\n".join(lines)

    def handle_intent(self, decision: dict[str, Any], user_id: str) -> OrchestratorResponse:
        if decision.get("type") != "skill_call":
            return OrchestratorResponse(decision.get("text", ""))

        skill_name = decision.get("skill", "")
        function_name = decision.get("function", "")
        args = decision.get("args") or {}

        permissions_map: dict[str, tuple[list[str], str | None]] = {
            "remember_note": (["db:write"], "insert"),
            "list_projects": (["db:read"], None),
            "add_reminder": (["db:write"], "insert"),
            "set_reminder": (["db:write"], "insert"),
            "next_best_task": (["db:read"], None),
            "daily_plan": (["db:read"], None),
            "weekly_review": (["db:read"], None),
        }
        requested_permissions, action_type = permissions_map.get(function_name, ([], None))
        result = self._call_skill(
            user_id,
            skill_name,
            function_name,
            args,
            requested_permissions,
            action_type=action_type,
        )

        if function_name == "remember_note":
            return OrchestratorResponse("Saved.") if result.data else result
        if function_name in {"add_reminder", "set_reminder"}:
            return OrchestratorResponse("Reminder set.") if result.data else result
        if function_name == "list_projects":
            if result.data:
                return OrchestratorResponse(self._format_projects(result.data.get("projects", [])))
            return result
        if function_name == "next_best_task":
            return OrchestratorResponse("Next-best-task suggestions are coming soon.")
        if function_name == "daily_plan":
            return OrchestratorResponse("Planning is coming soon.")
        if function_name == "weekly_review":
            return OrchestratorResponse("Weekly review is coming soon.")

        return result

    def handle_message(self, text: str, user_id: str) -> OrchestratorResponse:
        cmd = parse_command(text)
        if cmd and cmd.name == "confirm":
            pending = self.confirmations.pop(user_id)
            if not pending:
                return OrchestratorResponse("No pending action to confirm.")
            action = pending.action
            return self._call_skill(
                user_id,
                action["skill"],
                action["function"],
                action["args"],
                action["requested_permissions"],
                action.get("action_type"),
            )

        if cmd:
            if cmd.name == "remember":
                result = self._call_skill(
                    user_id,
                    "core",
                    "remember_note",
                    {"text": cmd.args},
                    ["db:write"],
                    action_type="insert",
                )
                return OrchestratorResponse("Saved.") if result.data else result

            if cmd.name == "projects":
                result = self._call_skill(
                    user_id,
                    "core",
                    "list_projects",
                    {},
                    ["db:read"],
                )
                if result.data:
                    return OrchestratorResponse(self._format_projects(result.data.get("projects", [])))
                return result

            if cmd.name == "project_new":
                name, pitch = split_pipe_args(cmd.args, 2)
                result = self._call_skill(
                    user_id,
                    "core",
                    "add_project",
                    {"name": name, "pitch": pitch or None},
                    ["db:write"],
                    action_type="insert",
                )
                return OrchestratorResponse("Project created.") if result.data else result

            if cmd.name == "task_add":
                project, title, effort, impact = split_pipe_args(cmd.args, 4)
                effort_mins = int(effort) if effort else None
                impact_1to5 = int(impact) if impact else None
                result = self._call_skill(
                    user_id,
                    "core",
                    "add_task",
                    {
                        "project": project or None,
                        "title": title,
                        "effort_mins": effort_mins,
                        "impact_1to5": impact_1to5,
                    },
                    ["db:write"],
                    action_type="insert",
                )
                return OrchestratorResponse("Task added.") if result.data else result

            if cmd.name == "remind":
                when_local, text = split_pipe_args(cmd.args, 2)
                result = self._call_skill(
                    user_id,
                    "core",
                    "add_reminder",
                    {"when_local": when_local, "text": text},
                    ["db:write"],
                    action_type="insert",
                )
                return OrchestratorResponse("Reminder set.") if result.data else result

            if cmd.name == "next":
                return OrchestratorResponse("Next-best-task suggestions are coming soon.")

            if cmd.name == "plan":
                return OrchestratorResponse("Planning is coming soon.")

            if cmd.name == "weekly":
                return OrchestratorResponse("Weekly review is coming soon.")

            if cmd.name == "done":
                return OrchestratorResponse("Logged. Mapping to tasks coming soon.")

        if self.llm_client and getattr(self.llm_client, "enabled", lambda: False)():
            intent = self.llm_client.intent_from_text(text)
            if intent:
                return OrchestratorResponse("LLM intent parsing not wired yet.")

        # Rule-based intent routing (v0.1)
        decision = route_message(user_id, text, self._intent_context())

        if decision.get("type") == "skill_call":
            return self._call_skill(
                user_id,
                "core",
                decision["function"],
                decision.get("args", {}),
                decision.get("scopes", ["db:read"]),
                action_type="auto",
            )

        if decision.get("type") == "clarification_request":
            return OrchestratorResponse(decision["prompt"])

        return OrchestratorResponse(
            "I can help with /remember, /projects, /project_new, /task_add, /remind. Use slash commands for now."
        )

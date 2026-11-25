"""Tool definitions and the registry the agent dispatches through."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..memory import LongTermMemory


class Risk(str, Enum):
    READ = "read"        # observes the environment only
    WRITE = "write"      # changes files in the workspace
    EXEC = "exec"        # runs arbitrary commands or code
    NETWORK = "network"  # reaches out to the internet
    CONTROL = "control"  # agent bookkeeping (plan, memory, finish)


class ToolError(Exception):
    """A tool failed in a way the agent should see and recover from."""


@dataclass
class PlanStep:
    step: str
    status: str = "pending"  # pending | in_progress | done | blocked | skipped


@dataclass
class RunState:
    """Mutable state of one run that control tools read and write."""

    goal: str
    plan: list[PlanStep] = field(default_factory=list)
    finish_request: dict | None = None


@dataclass
class ToolContext:
    config: "AgentConfig"
    memory: "LongTermMemory"
    state: RunState
    ask_human: Callable[[str], str | None]

    @property
    def workspace(self) -> Path:
        return self.config.workspace


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict, ToolContext], str]
    risk: Risk

    def to_api(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def schema(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def api_definitions(self) -> list[dict]:
        # Sorted so the tool list is byte-identical every request (prompt cache).
        return [self._tools[name].to_api() for name in sorted(self._tools)]


def validate_input(tool: Tool, data: Any) -> None:
    """Light schema check before running a tool (required keys and basic types)."""
    if not isinstance(data, dict):
        raise ToolError("Tool input must be a JSON object.")
    props = tool.input_schema.get("properties", {})
    for key in tool.input_schema.get("required", []):
        if key not in data:
            raise ToolError(f"Missing required argument '{key}'.")
    unknown = set(data) - set(props)
    if unknown:
        raise ToolError(f"Unknown argument(s): {', '.join(sorted(unknown))}.")
    type_map = {"string": str, "integer": int, "boolean": bool, "array": list, "object": dict, "number": (int, float)}
    for key, value in data.items():
        expected = props[key].get("type")
        py = type_map.get(expected)
        if py is not None and not isinstance(value, py):
            raise ToolError(f"Argument '{key}' must be of type {expected}.")

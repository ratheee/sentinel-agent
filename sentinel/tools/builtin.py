"""The built-in tool set."""

from __future__ import annotations

from . import control, execution, files, web
from .registry import ToolRegistry


def default_registry() -> ToolRegistry:
    return ToolRegistry(files.TOOLS + execution.TOOLS + web.TOOLS + control.TOOLS)

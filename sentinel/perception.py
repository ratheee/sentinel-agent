"""Perception: turn the agent's environment into observations.

Two kinds of observation feed the reasoning loop:

* an initial **snapshot** - platform, time, workspace layout, git state
* **change detection** after every action - which files appeared, changed or
  vanished - so the agent sees side effects it didn't explicitly ask about
  (e.g. a test run that wrote a cache, a script that produced an output file)
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

IGNORED_DIRS = {".git", ".sentinel", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache",
                ".pytest_cache", ".idea", ".vscode", "dist", "build", ".tox"}


def iter_files(root: Path, limit: int = 5000):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info"))
        for name in sorted(filenames):
            yield Path(dirpath) / name
            count += 1
            if count >= limit:
                return


def tree(root: Path, max_depth: int = 3, max_entries: int = 150) -> str:
    lines: list[str] = []

    def walk(directory: Path, depth: int) -> None:
        if depth > max_depth or len(lines) >= max_entries:
            return
        try:
            children = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        for child in children:
            if child.name in IGNORED_DIRS:
                continue
            if len(lines) >= max_entries:
                lines.append("  " * depth + "... (truncated)")
                return
            if child.is_dir():
                lines.append("  " * depth + child.name + "/")
                walk(child, depth + 1)
            else:
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                lines.append("  " * depth + f"{child.name} ({size} B)")

    walk(root, 0)
    return "\n".join(lines) if lines else "(empty)"


def _git_summary(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "status", "--short", "--branch"], cwd=root, capture_output=True,
                             text=True, timeout=10, encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = out.stdout.strip().splitlines()
    return "\n".join(lines[:30]) if lines else None


def snapshot(workspace: Path) -> str:
    shell = "cmd.exe (Windows)" if os.name == "nt" else os.environ.get("SHELL", "/bin/sh")
    parts = [
        f"Time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Platform: {platform.system()} {platform.release()} ({platform.machine()})",
        f"Shell used by run_command: {shell}",
        f"Python: {sys.version.split()[0]} at {sys.executable}",
        f"Workspace: {workspace}",
        "Workspace contents:\n" + tree(workspace),
    ]
    git = _git_summary(workspace)
    if git:
        parts.append("Git status:\n" + git)
    return "\n".join(parts)


class WorkspaceWatcher:
    """Detects file-level changes in the workspace between two points in time."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self._last = self._scan()

    def _scan(self) -> dict[str, tuple[float, int]]:
        state = {}
        for path in iter_files(self.workspace):
            try:
                st = path.stat()
            except OSError:
                continue
            state[path.relative_to(self.workspace).as_posix()] = (st.st_mtime, st.st_size)
        return state

    def changes(self) -> str | None:
        current = self._scan()
        added = sorted(set(current) - set(self._last))
        deleted = sorted(set(self._last) - set(current))
        modified = sorted(p for p in set(current) & set(self._last) if current[p] != self._last[p])
        self._last = current
        if not (added or deleted or modified):
            return None

        def fmt(label: str, items: list[str]) -> str:
            shown = ", ".join(items[:15]) + (f" (+{len(items) - 15} more)" if len(items) > 15 else "")
            return f"{label}: {shown}"

        parts = [fmt(label, items) for label, items in
                 (("added", added), ("modified", modified), ("deleted", deleted)) if items]
        return "; ".join(parts)

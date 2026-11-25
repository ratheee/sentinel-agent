"""Tools that execute commands or code inside the workspace."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from ..guardrails import check_command, scrubbed_env
from .registry import Risk, Tool, ToolContext, ToolError, schema


def _run(argv_or_cmd, ctx: ToolContext, timeout: int, shell: bool) -> str:
    try:
        proc = subprocess.run(
            argv_or_cmd, shell=shell, cwd=ctx.workspace, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, env=scrubbed_env(),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"Timed out after {timeout}s. Partial output:\n{(exc.stdout or '')[-2000:]}") from exc
    out = []
    if proc.stdout:
        out.append(f"[stdout]\n{proc.stdout.rstrip()}")
    if proc.stderr:
        out.append(f"[stderr]\n{proc.stderr.rstrip()}")
    out.append(f"[exit code {proc.returncode}]")
    return "\n".join(out)


def _run_command(args: dict, ctx: ToolContext) -> str:
    command = args["command"].strip()
    if not command:
        raise ToolError("Empty command.")
    blocked = check_command(command)
    if blocked:
        raise ToolError(f"Blocked by guardrail ({blocked}). Find a safer way to reach the goal.")
    timeout = min(int(args.get("timeout", ctx.config.command_timeout)), 600)
    return _run(command, ctx, timeout, shell=True)


def _run_python(args: dict, ctx: ToolContext) -> str:
    code = args["code"]
    blocked = check_command(code)
    if blocked:
        raise ToolError(f"Blocked by guardrail ({blocked}).")
    timeout = min(int(args.get("timeout", ctx.config.command_timeout)), 600)
    fd, script = tempfile.mkstemp(suffix=".py", prefix="sentinel_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(code)
        return _run([sys.executable, script], ctx, timeout, shell=False)
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


TOOLS = [
    Tool(
        "run_command",
        "Run a shell command in the workspace directory and return stdout, stderr and the exit code. "
        "Use it for tests, builds, git, package managers and other CLIs. Non-interactive: stdin is closed. "
        "Secrets are removed from the environment.",
        schema({"command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds, default 120, max 600"}}, ["command"]),
        _run_command, Risk.EXEC,
    ),
    Tool(
        "run_python",
        "Execute a Python script (with the same interpreter that runs the agent) in the workspace directory. "
        "Good for data analysis, calculations, parsing files and quick experiments. Print what you need to see.",
        schema({"code": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds, default 120, max 600"}}, ["code"]),
        _run_python, Risk.EXEC,
    ),
]

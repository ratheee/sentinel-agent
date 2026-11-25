"""Safety guardrails: what the agent may touch, what needs a human, and when to stop.

These are limits on *actions*, not on reasoning. The LLM decides what to do;
the guardrails decide whether a particular action is allowed to run.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .tools.registry import Risk, ToolError

# Commands that are never run, whatever the approval mode.
_BLOCKED_COMMANDS: list[tuple[str, str]] = [
    (r"\brm\s+-[a-z]*(rf|fr)[a-z]*\s+(/|~|\*|\.\.?)(\s|/?$)", "recursive delete of a root, home or whole directory"),
    (r"\bmkfs(\.\w+)?\b|\bdiskpart\b|\bformat\s+[a-z]:", "disk formatting"),
    (r"\bdd\s+.*\bof=/dev/", "raw write to a device"),
    (r"\b(shutdown|reboot|halt|poweroff)\b|Stop-Computer|Restart-Computer", "power state change"),
    (r":\(\)\s*\{.*\};\s*:", "fork bomb"),
    (r"(curl|wget|iwr|Invoke-WebRequest|irm|Invoke-RestMethod)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|iex|Invoke-Expression|python\d?|powershell|pwsh)\b",
     "piping a download straight into an interpreter"),
    (r"\bsudo\b|\brunas\b", "privilege escalation"),
    (r"\bgit\s+push\b.*(--force\b|\s-f\b)", "force push"),
    (r"\bdel\s+(/[a-z]\s+)*[a-z]:\\\s*$|\brd\s+/s\s+/q\s+[a-z]:\\", "deleting a drive root"),
    (r"Remove-Item\b.*-Recurse.*\s([A-Za-z]:\\?|~|\$env:\w+)\s*$", "recursive delete outside the workspace"),
    (r"\breg\s+(add|delete)\b|Set-ItemProperty\s+.*HK(LM|CU):", "registry modification"),
    (r"\bchmod\s+-R\s+777\s+/", "opening permissions on the filesystem root"),
    (r"\b(crontab|schtasks|launchctl|systemctl\s+enable)\b", "installing persistent scheduled jobs"),
    (r"[\\/]\.ssh[\\/]|\bid_(rsa|ed25519)\b|\.aws[\\/]credentials", "reading private keys or cloud credentials"),
]
_BLOCKED_RE = [(re.compile(p, re.IGNORECASE), reason) for p, reason in _BLOCKED_COMMANDS]

_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE)


def check_command(command: str) -> str | None:
    """Return the reason a shell command is blocked, or None if it may run."""
    for pattern, reason in _BLOCKED_RE:
        if pattern.search(command):
            return reason
    return None


def scrubbed_env() -> dict[str, str]:
    """Environment for child processes, minus anything that looks like a secret."""
    return {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}


def resolve_path(workspace: Path, raw: str, *, for_write: bool = False) -> Path:
    """Resolve ``raw`` inside the workspace, refusing anything that escapes it."""
    candidate = Path(raw).expanduser()
    target = (candidate if candidate.is_absolute() else workspace / candidate).resolve()
    if target != workspace and workspace not in target.parents:
        raise ToolError(f"Path '{raw}' is outside the workspace ({workspace}). Work inside the workspace.")
    if for_write:
        state_dir = workspace / ".sentinel"
        if target == state_dir or state_dir in target.parents:
            raise ToolError("The .sentinel directory is the agent's own state; write elsewhere.")
    return target


class ApprovalPolicy:
    """Decides which actions need a human's OK before they run.

    auto  - nothing needs approval (fully autonomous; hard blocks still apply)
    risky - shell/code execution needs approval
    all   - every action with a side effect (write, exec, network) needs approval
    """

    def __init__(self, mode: str):
        self.mode = mode

    def needs_approval(self, risk: Risk) -> bool:
        if self.mode == "auto":
            return False
        if self.mode == "risky":
            return risk is Risk.EXEC
        return risk in (Risk.WRITE, Risk.EXEC, Risk.NETWORK)


@dataclass
class Budget:
    max_steps: int
    max_cost_usd: float
    steps: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    warned: bool = False

    def record(self, usage, cost: float) -> None:
        self.cost_usd += cost
        if usage is not None:
            self.input_tokens += (getattr(usage, "input_tokens", 0) or 0) + (
                getattr(usage, "cache_read_input_tokens", 0) or 0
            ) + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0

    def exhausted(self) -> str | None:
        if self.steps >= self.max_steps:
            return f"step limit reached ({self.max_steps})"
        if self.max_cost_usd > 0 and self.cost_usd >= self.max_cost_usd:
            return f"cost budget reached (${self.cost_usd:.2f} of ${self.max_cost_usd:.2f})"
        return None

    def remaining_note(self) -> str:
        return f"{self.max_steps - self.steps} steps and ${max(self.max_cost_usd - self.cost_usd, 0):.2f} of budget left"

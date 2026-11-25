"""Runtime configuration for a Sentinel run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"

# USD per million tokens: (input, output, cache_read). Cache writes bill at 1.25x input.
PRICING: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-fable-5-2": (8.00, 40.00, 0.20),
}


@dataclass
class AgentConfig:
    workspace: Path = field(default_factory=Path.cwd)
    model: str = DEFAULT_MODEL
    effort: str = "high"  # low | medium | high | xhigh | max
    max_tokens: int = 16000
    max_steps: int = 40
    max_cost_usd: float = 3.00
    approval: str = "risky"  # auto | risky | all
    web_search: bool = True
    refusal_fallback: bool = True
    command_timeout: int = 120
    max_tool_output_chars: int = 12000
    max_verifications: int = 3

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).expanduser().resolve()
        if self.approval not in ("auto", "risky", "all"):
            raise ValueError(f"approval must be auto, risky or all (got {self.approval!r})")

    @property
    def state_dir(self) -> Path:
        """Where Sentinel keeps memory and run traces (inside the workspace)."""
        return self.workspace / ".sentinel"

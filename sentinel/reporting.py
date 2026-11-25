"""How a run is shown to the human: a silent reporter and a rich console one."""

from __future__ import annotations

import json
import sys
from typing import Any

from .tools.registry import PlanStep, Risk


class Reporter:
    """No-op reporter. Subclass and override what you want to display."""

    def run_started(self, goal: str, config: Any) -> None: ...
    def perceived(self, environment: str, memories: list[str]) -> None: ...
    def step(self, number: int, budget: Any) -> None: ...
    def thinking(self, text: str) -> None: ...
    def message(self, text: str) -> None: ...
    def tool_call(self, name: str, args: dict, risk: Risk) -> None: ...
    def tool_result(self, name: str, output: str, is_error: bool) -> None: ...
    def server_tool(self, name: str, args: Any) -> None: ...
    def plan(self, steps: list[PlanStep]) -> None: ...
    def observation(self, text: str) -> None: ...
    def note(self, text: str) -> None: ...
    def verification(self, verdict: dict) -> None: ...
    def reflection(self, lessons: list[str]) -> None: ...
    def run_finished(self, result: Any) -> None: ...

    def confirm(self, name: str, args: dict, risk: Risk) -> bool:
        return False  # nobody to ask: deny

    def ask(self, question: str) -> str | None:
        return None


_UNICODE = {"perceive": "◉", "reason": "✻", "act": "▶", "ok": "◂", "err": "✗", "plan": "☰", "note": "⚑",
            "pass": "✔", "retry": "↺", "reflect": "✎", "bullet": "•", "nl": " ⏎ ", "more": "…", "dot": "·"}
_ASCII = {"perceive": "(o)", "reason": "~", "act": ">", "ok": "<", "err": "x", "plan": "=", "note": "!",
          "pass": "+", "retry": "<>", "reflect": "*", "bullet": "-", "nl": " | ", "more": "...", "dot": "|"}


def _glyphs() -> dict[str, str]:
    """Unicode symbols on UTF-8 terminals, ASCII on legacy code pages (e.g. Windows cp1252)."""
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "")
    return _UNICODE if encoding in ("utf8", "utf8sig") else _ASCII


class ConsoleReporter(Reporter):
    def __init__(self, verbose: bool = True):
        from rich.console import Console

        self.g = _glyphs()
        self.console = Console(highlight=False)
        self.verbose = verbose
        self.interactive = sys.stdin.isatty()

    def _short(self, value: Any, limit: int = 160) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        text = text.replace("\n", self.g["nl"])
        return text if len(text) <= limit else text[: limit - 1] + self.g["more"]

    def _p(self, text: str, **kw: Any) -> None:
        self.console.print(text, **kw)

    def run_started(self, goal, config):
        from rich.markup import escape
        from rich.panel import Panel

        d = self.g["dot"]
        self._p(Panel(
            f"[bold]{escape(goal)}[/]\n\n[dim]workspace[/] {config.workspace}\n[dim]model[/] {config.model} "
            f"{d} effort {config.effort} {d} approval {config.approval} {d} max {config.max_steps} steps"
            + (f" / ${config.max_cost_usd:.2f}" if config.max_cost_usd > 0 else " (no cost cap)"),
            title=f"Sentinel {d} goal", border_style="cyan"))

    def perceived(self, environment, memories):
        extra = f", {len(memories)} relevant memories recalled" if memories else ""
        self._p(f"[bold cyan]{self.g['perceive']} perceive[/] environment snapshot taken{extra}")
        if self.verbose:
            for m in memories:
                self.console.print("   memory: " + self._short(m, 140), style="dim", markup=False)

    def step(self, number, budget):
        self.console.rule(f"[dim]step {number} {self.g['dot']} ${budget.cost_usd:.3f} spent[/]", style="dim")

    def thinking(self, text):
        if self.verbose and text.strip():
            self.console.print(f"{self.g['reason']} reason  " + self._short(text, 600),
                               style="italic magenta", markup=False)

    def message(self, text):
        if text.strip():
            self.console.print(text.strip(), markup=False)

    def tool_call(self, name, args, risk):
        if name == "update_plan":
            return
        colour = {"read": "green", "write": "yellow", "exec": "red", "network": "blue"}.get(risk.value, "cyan")
        self._p(f"[bold {colour}]{self.g['act']} act[/] [bold]{name}[/] ", end="")
        self.console.print(self._short(args), style="dim", markup=False)

    def tool_result(self, name, output, is_error):
        if name in ("update_plan", "finish"):
            return
        if self.verbose or is_error:
            prefix = self.g["err"] if is_error else self.g["ok"]
            self.console.print(f"  {prefix} " + self._short(output, 300), style="red" if is_error else "dim",
                               markup=False)

    def server_tool(self, name, args):
        self._p(f"[bold blue]{self.g['act']} act[/] [bold]{name}[/] [dim](server)[/] ", end="")
        self.console.print(self._short(args), style="dim", markup=False)

    def plan(self, steps):
        from .tools.control import render_plan

        self._p(f"[bold cyan]{self.g['plan']} plan[/]")
        for line in render_plan(steps).splitlines():
            self.console.print(f"   {line}", markup=False)

    def observation(self, text):
        self._p(f"[cyan]{self.g['perceive']} observe[/] ", end="")
        self.console.print(self._short(text, 300), style="dim", markup=False)

    def note(self, text):
        self.console.print(f"{self.g['note']} {text}", style="yellow", markup=False)

    def verification(self, verdict):
        if verdict.get("accept"):
            self._p(f"[bold green]{self.g['pass']} verify[/] reviewer accepted: ", end="")
            self.console.print(self._short(verdict.get("feedback", ""), 300), markup=False)
        else:
            detail = "; ".join(verdict.get("gaps", [])) or verdict.get("feedback", "")
            self._p(f"[bold yellow]{self.g['retry']} verify[/] reviewer sent the agent back: ", end="")
            self.console.print(self._short(detail, 400), markup=False)

    def reflection(self, lessons):
        if lessons:
            self._p(f"[bold cyan]{self.g['reflect']} reflect[/] lessons saved to memory:")
            for lesson in lessons:
                self.console.print(f"   {self.g['bullet']} {lesson}", markup=False)

    def run_finished(self, result):
        from rich.markup import escape
        from rich.panel import Panel

        colour = {"success": "green", "failed": "red", "partial": "yellow"}.get(result.status, "yellow")
        d = self.g["dot"]
        body = escape(result.summary)
        if result.evidence:
            body += f"\n\n[dim]evidence:[/] {escape(result.evidence)}"
        body += f"\n\n[dim]{result.steps} steps {d} ${result.cost_usd:.3f} {d} trace {result.trace_path}[/]"
        self._p(Panel(body, title=f"result {d} {result.status}", border_style=colour))

    def confirm(self, name, args, risk):
        if not self.interactive:
            return False
        from rich.prompt import Confirm

        self._p(f"[bold yellow]? approval needed[/] {name} ({risk.value})")
        self.console.print("  " + json.dumps(args, indent=2, ensure_ascii=False)[:1500], markup=False)
        return Confirm.ask("  allow this action?", default=True, console=self.console)

    def ask(self, question):
        if not self.interactive:
            return None
        from rich.prompt import Prompt

        self._p("[bold yellow]? the agent asks:[/] ", end="")
        self.console.print(question, markup=False)
        answer = Prompt.ask("  your answer (blank = let the agent decide)", default="", console=self.console)
        return answer or None

"""The agent loop: perceive -> reason -> act -> observe, with verification and reflection.

    perceive   snapshot of the workspace + relevant long-term memories
       |
    reason     the LLM reads the conversation and picks the next action(s)
       |
    act        tool calls go through validation, guardrails and approval, then run
       |
    observe    tool results + detected workspace changes go back to the LLM
       |       (repeated failures trigger a re-plan prompt)
       v
    finish     the agent claims completion -> an independent verifier checks the
               evidence; if it isn't convincing the agent is sent back to work
       |
    reflect    lessons from the run are written to long-term memory
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

from . import prompts
from .config import AgentConfig
from .guardrails import ApprovalPolicy, Budget
from .llm import LLM, LLMAuthError, LLMError, LLMRefusal
from .memory import LongTermMemory
from .perception import WorkspaceWatcher, snapshot
from .reporting import Reporter
from .tools.builtin import default_registry
from .tools.control import render_plan
from .tools.registry import RunState, ToolContext, ToolError, ToolRegistry, validate_input
from .tools.web import WEB_SEARCH_SERVER_TOOL

_NONZERO_EXIT = re.compile(r"\[exit code (?!0\])-?\d+\]")


@dataclass
class ActionRecord:
    step: int
    tool: str
    args: str
    result: str
    failed: bool


@dataclass
class RunResult:
    status: str  # success | failed | partial | stopped
    summary: str
    evidence: str = ""
    steps: int = 0
    cost_usd: float = 0.0
    run_id: str = ""
    trace_path: Path | None = None
    actions: list[ActionRecord] = field(default_factory=list)


class Trace:
    """Append-only JSONL log of everything that happened in a run."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **data: Any) -> None:
        record = {"t": round(time.time(), 3), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_jsonable, ensure_ascii=False) + "\n")


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return str(obj)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} characters omitted] ...\n{text[-tail:]}"


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        *,
        llm: LLM | None = None,
        registry: ToolRegistry | None = None,
        reporter: Reporter | None = None,
        memory: LongTermMemory | None = None,
    ):
        self.config = config
        self.llm = llm or LLM(config)
        self.registry = registry or default_registry()
        self.reporter = reporter or Reporter()
        self.memory = memory or LongTermMemory(config.state_dir / "memory.jsonl")
        self.policy = ApprovalPolicy(config.approval)

    # ------------------------------------------------------------------ run

    def run(self, goal: str) -> RunResult:
        cfg = self.config
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        trace = Trace(cfg.state_dir / "runs" / f"{run_id}.jsonl")
        state = RunState(goal=goal)
        ctx = ToolContext(cfg, self.memory, state, ask_human=self.reporter.ask)
        budget = Budget(cfg.max_steps, cfg.max_cost_usd)
        actions: list[ActionRecord] = []

        self.reporter.run_started(goal, cfg)
        trace.write("run_started", goal=goal, model=cfg.model, workspace=str(cfg.workspace),
                    approval=cfg.approval)

        # PERCEIVE: initial view of the world and of what we already know about it.
        environment = snapshot(cfg.workspace)
        memories = [m.render() for m in self.memory.search(goal, k=8)]
        watcher = WorkspaceWatcher(cfg.workspace)
        self.reporter.perceived(environment, memories)
        trace.write("perceived", environment=environment, memories=memories)

        messages: list[dict] = [
            {"role": "user", "content": prompts.kickoff_message(goal, environment, memories)}
        ]
        tools = self.registry.api_definitions()
        if cfg.web_search:
            tools = tools + [WEB_SEARCH_SERVER_TOOL]

        consecutive_failures = 0
        idle_turns = 0
        verifications = 0
        result: RunResult | None = None

        while result is None:
            stop = budget.exhausted()
            if stop:
                result = RunResult("stopped", f"Stopped before the goal was confirmed: {stop}.\n\n"
                                   f"Plan at stop:\n{render_plan(state.plan)}")
                break
            budget.steps += 1
            self.reporter.step(budget.steps, budget)

            # REASON
            try:
                response = self.llm.respond(prompts.ACTOR_SYSTEM, messages, tools)
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                    LLMAuthError):
                raise  # configuration problems: let the caller report them
            except (anthropic.APIStatusError, anthropic.APIConnectionError, LLMError) as exc:
                trace.write("api_error", error=str(exc))
                result = RunResult("stopped", f"Stopped: the model API failed ({type(exc).__name__}: {exc}).")
                break

            cost = self.llm.cost_of(response.usage)
            budget.record(response.usage, cost)
            trace.write("assistant", step=budget.steps, stop_reason=response.stop_reason,
                        content=response.content, usage=response.usage, cost_usd=round(cost, 5))
            messages.append({"role": "assistant", "content": response.content})
            self._display(response.content)

            if response.stop_reason == "refusal":
                result = RunResult("stopped", "Stopped: the model declined to continue with this goal.")
                break
            if response.stop_reason == "pause_turn":
                continue  # a server-side tool (web search) paused mid-turn; re-send to resume

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                idle_turns += 1
                if idle_turns > 2:
                    last_text = "\n".join(b.text for b in response.content if b.type == "text")
                    result = RunResult("partial", last_text or "The agent stopped without calling finish.")
                    break
                messages.append({"role": "user", "content": prompts.IDLE_NUDGE})
                continue
            idle_turns = 0

            # ACT
            truncated = response.stop_reason == "max_tokens"
            results: list[dict] = []
            step_failed = []
            # Run finish last so the verifier sees the effect of everything else in this turn.
            ordered = sorted(tool_uses, key=lambda b: b.name == "finish")
            for block in ordered:
                if truncated:
                    output, is_error = ("This tool call was cut off by the output limit and did not run. "
                                        "Re-issue it, splitting large content across smaller calls."), True
                else:
                    output, is_error = self._execute(block, ctx)
                failed = is_error or bool(_NONZERO_EXIT.search(output))
                if block.name not in ("update_plan", "finish"):
                    step_failed.append(failed)
                    actions.append(ActionRecord(budget.steps, block.name, _truncate(json.dumps(block.input,
                                                ensure_ascii=False), 400), _truncate(output, 600), failed))

                if block.name == "update_plan" and not is_error:
                    self.reporter.plan(state.plan)

                if block.name == "finish" and state.finish_request and not truncated:
                    verifications += 1
                    output, result = self._review_finish(goal, state, actions, verifications, budget)
                    trace.write("verification", attempt=verifications, feedback=output,
                                accepted=result is not None)
                    if result is None:
                        state.finish_request = None

                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output,
                                **({"is_error": True} if is_error else {})})
                trace.write("tool", step=budget.steps, name=block.name, input=block.input,
                            output=output, is_error=is_error)

            if result is not None:
                break

            # OBSERVE: side effects in the environment, then decide if a re-plan is needed.
            notes: list[str] = []
            changes = watcher.changes()
            if changes:
                notes.append(f"[observation] Workspace changes since your last step: {changes}")
                self.reporter.observation(changes)
            if step_failed and all(step_failed):
                consecutive_failures += 1
            elif step_failed:
                consecutive_failures = 0
            if consecutive_failures >= 3:
                notes.append(prompts.REPLAN_NUDGE)
                self.reporter.note("repeated failures: asking the agent to step back and re-plan")
                consecutive_failures = 0
            steps_left = cfg.max_steps - budget.steps
            near_cost_cap = cfg.max_cost_usd > 0 and budget.cost_usd > 0.85 * cfg.max_cost_usd
            if not budget.warned and (steps_left <= 3 or near_cost_cap):
                budget.warned = True
                notes.append(prompts.budget_warning(budget.remaining_note()))

            messages.append({"role": "user",
                             "content": results + [{"type": "text", "text": n} for n in notes]})

        result.steps = budget.steps
        result.run_id = run_id
        result.trace_path = trace.path
        result.actions = actions

        # REFLECT
        self._reflect(goal, result, actions, budget, trace)
        result.cost_usd = budget.cost_usd
        trace.write("run_finished", status=result.status, summary=result.summary, cost_usd=budget.cost_usd,
                    steps=budget.steps)
        self.reporter.run_finished(result)
        return result

    # ------------------------------------------------------------- helpers

    def _display(self, content: list) -> None:
        for block in content:
            kind = block.type
            if kind == "thinking":
                self.reporter.thinking(getattr(block, "thinking", "") or "")
            elif kind == "text":
                self.reporter.message(block.text)
            elif kind == "server_tool_use":
                self.reporter.server_tool(block.name, block.input)
            elif kind == "fallback":
                self.reporter.note("the primary model declined this turn; a fallback model continued it")

    def _execute(self, block: Any, ctx: ToolContext) -> tuple[str, bool]:
        tool = self.registry.get(block.name)
        if tool is None:
            return f"Unknown tool '{block.name}'.", True
        args = block.input if isinstance(block.input, dict) else {}
        try:
            validate_input(tool, block.input)
        except ToolError as exc:
            return f"Invalid input: {exc}", True

        if self.policy.needs_approval(tool.risk):
            if not self.reporter.confirm(tool.name, args, tool.risk):
                self.reporter.note(f"{tool.name} was not approved")
                return ("The human supervisor did not approve this action, so it did not run. "
                        "Pick a different approach, or ask_human what would be acceptable."), True

        self.reporter.tool_call(tool.name, args, tool.risk)
        try:
            output, is_error = tool.handler(args, ctx), False
        except ToolError as exc:
            output, is_error = f"Error: {exc}", True
        except Exception as exc:  # a tool bug must not kill the run; the agent can route around it
            output, is_error = f"Unexpected error in {tool.name}: {type(exc).__name__}: {exc}", True
        output = _truncate(output or "(no output)", self.config.max_tool_output_chars)
        self.reporter.tool_result(tool.name, output, is_error)
        return output, is_error

    def _review_finish(self, goal: str, state: RunState, actions: list[ActionRecord], attempt: int,
                       budget: Budget) -> tuple[str, RunResult | None]:
        """Run the independent verifier. Returns (tool_result text, RunResult if the run may end)."""
        req = state.finish_request or {}
        log = "\n".join(
            f"[step {a.step}] {a.tool} {a.args}\n    -> {'FAILED ' if a.failed else ''}{a.result}"
            for a in actions[-60:]
        ) or "(no actions)"
        prompt = (
            f"<goal>\n{goal}\n</goal>\n\n"
            f"<agent_claim success=\"{str(req.get('success')).lower()}\">\n"
            f"Summary: {req.get('summary', '')}\n\nEvidence: {req.get('evidence', '')}\n</agent_claim>\n\n"
            f"<plan>\n{render_plan(state.plan)}\n</plan>\n\n"
            f"<action_log>\n{log}\n</action_log>"
        )
        try:
            verdict, usage = self.llm.structured(prompts.VERIFIER_SYSTEM, prompt, prompts.VERIFIER_SCHEMA)
            budget.record(usage, self.llm.cost_of(usage))
        except (anthropic.APIError, LLMError, LLMRefusal, json.JSONDecodeError, ValueError) as exc:
            # Verifier unavailable: fall back to the agent's own claim rather than loop forever.
            verdict = {"accept": True, "goal_met": bool(req.get("success")), "gaps": [],
                       "feedback": f"(verifier unavailable: {exc}; accepted the agent's own assessment)"}
        self.reporter.verification(verdict)

        summary, evidence = req.get("summary", ""), req.get("evidence", "")
        if verdict.get("accept"):
            status = "success" if verdict.get("goal_met") and req.get("success") else "failed"
            return "Reviewer accepted. The run is complete.", RunResult(status, summary, evidence)
        if attempt >= self.config.max_verifications:
            self.reporter.note("verification attempts exhausted; ending with a partial result")
            gaps = "; ".join(verdict.get("gaps", []))
            return "Reviewer still not satisfied; ending.", RunResult(
                "partial", f"{summary}\n\nReviewer's remaining concerns: {gaps or verdict.get('feedback', '')}",
                evidence)
        gaps = "\n".join(f"- {g}" for g in verdict.get("gaps", []))
        return (f"The reviewer did not accept this finish. Keep working.\n"
                f"Feedback: {verdict.get('feedback', '')}\nGaps:\n{gaps or '- (none listed)'}"), None

    def _reflect(self, goal: str, result: RunResult, actions: list[ActionRecord], budget: Budget,
                 trace: Trace) -> None:
        self.memory.add(f"Goal: {goal[:200]} -> {result.status}. {result.summary[:300]}",
                        kind="episode", tags=[result.status], goal=goal)
        if len(actions) < 2:
            return
        log = "\n".join(f"{a.tool} {a.args[:200]} -> {'FAILED ' if a.failed else ''}{a.result[:300]}"
                        for a in actions[-40:])
        prompt = (f"<goal>{goal}</goal>\n<outcome status=\"{result.status}\">{result.summary}</outcome>\n"
                  f"<action_log>\n{log}\n</action_log>")
        try:
            data, usage = self.llm.structured(prompts.REFLECTOR_SYSTEM, prompt, prompts.REFLECTOR_SCHEMA,
                                              effort="low")
            budget.record(usage, self.llm.cost_of(usage))
        except (anthropic.APIError, LLMError, LLMRefusal, json.JSONDecodeError, ValueError) as exc:
            trace.write("reflection_error", error=str(exc))
            return
        lessons = [lesson.strip() for lesson in data.get("lessons", []) if lesson.strip()][:3]
        for lesson in lessons:
            self.memory.add(lesson, kind="lesson", tags=["reflection"], goal=goal)
        trace.write("reflection", lessons=lessons)
        self.reporter.reflection(lessons)

"""Tools the agent uses to manage itself: its plan, its memory, humans, and finishing."""

from __future__ import annotations

from .registry import PlanStep, Risk, Tool, ToolContext, ToolError, schema

PLAN_STATUSES = ("pending", "in_progress", "done", "blocked", "skipped")
_MARK = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]", "blocked": "[!]", "skipped": "[-]"}


def render_plan(plan: list[PlanStep]) -> str:
    if not plan:
        return "(no plan yet)"
    return "\n".join(f"{i}. {_MARK.get(s.status, '[?]')} {s.step}" for i, s in enumerate(plan, 1))


def _update_plan(args: dict, ctx: ToolContext) -> str:
    steps = []
    for item in args["steps"]:
        if not isinstance(item, dict) or not str(item.get("step", "")).strip():
            raise ToolError("Each step needs a non-empty 'step' string.")
        status = item.get("status", "pending")
        if status not in PLAN_STATUSES:
            raise ToolError(f"status must be one of {', '.join(PLAN_STATUSES)}.")
        steps.append(PlanStep(str(item["step"]).strip(), status))
    ctx.state.plan = steps
    return "Plan recorded:\n" + render_plan(steps)


def _remember(args: dict, ctx: ToolContext) -> str:
    kind = args.get("kind", "fact")
    if kind not in ("fact", "lesson"):
        raise ToolError("kind must be 'fact' or 'lesson'.")
    entry = ctx.memory.add(args["text"], kind=kind, tags=args.get("tags", []), goal=ctx.state.goal)
    return f"Saved to long-term memory (id {entry.id})."


def _recall(args: dict, ctx: ToolContext) -> str:
    hits = ctx.memory.search(args["query"], k=int(args.get("limit", 6)))
    return "\n".join(f"- {h.render()}" for h in hits) if hits else "Nothing relevant in long-term memory."


def _ask_human(args: dict, ctx: ToolContext) -> str:
    answer = ctx.ask_human(args["question"])
    if answer is None:
        return ("No human is available. Make the most reasonable assumption, state it explicitly in your "
                "final summary, and continue.")
    return f"Human answered: {answer}"


def _finish(args: dict, ctx: ToolContext) -> str:
    ctx.state.finish_request = {
        "success": bool(args["success"]),
        "summary": args["summary"],
        "evidence": args.get("evidence", ""),
    }
    return "Finish requested."  # the agent loop replaces this with the verifier's verdict


TOOLS = [
    Tool(
        "update_plan",
        "Record or revise your plan for the goal as an ordered list of steps with statuses. "
        "Call it at the start, and again whenever you complete a step or new information changes the plan. "
        "The full list replaces the previous plan.",
        schema({"steps": {"type": "array", "items": {
            "type": "object",
            "properties": {"step": {"type": "string"}, "status": {"type": "string", "enum": list(PLAN_STATUSES)}},
            "required": ["step", "status"], "additionalProperties": False}}}, ["steps"]),
        _update_plan, Risk.CONTROL,
    ),
    Tool(
        "remember",
        "Save a durable fact or lesson to long-term memory so future runs can reuse it "
        "(project conventions, how to run things here, approaches that failed and why). Not for transient details.",
        schema({"text": {"type": "string"},
                "kind": {"type": "string", "enum": ["fact", "lesson"]},
                "tags": {"type": "array", "items": {"type": "string"}}}, ["text"]),
        _remember, Risk.CONTROL,
    ),
    Tool(
        "recall",
        "Search long-term memory from previous runs.",
        schema({"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]),
        _recall, Risk.CONTROL,
    ),
    Tool(
        "ask_human",
        "Ask the human supervisor a question. Use only when blocked on something only they can decide or provide "
        "(an ambiguous requirement that changes the outcome, missing credentials). They may not answer.",
        schema({"question": {"type": "string"}}, ["question"]),
        _ask_human, Risk.CONTROL,
    ),
    Tool(
        "finish",
        "End the run. Call when the goal is achieved, or when it cannot be achieved and you have exhausted "
        "reasonable approaches. An independent reviewer checks your summary and evidence against the goal "
        "and may send you back to keep working.",
        schema({"success": {"type": "boolean", "description": "true if the goal is achieved"},
                "summary": {"type": "string", "description": "What was done and the outcome, for the user"},
                "evidence": {"type": "string", "description": "Concrete proof: test output, files created, "
                                                              "values observed, sources consulted"}},
               ["success", "summary", "evidence"]),
        _finish, Risk.CONTROL,
    ),
]

"""Test helpers: a scripted stand-in for the LLM so the agent loop runs offline."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from sentinel.config import AgentConfig
from sentinel.llm import LLM

_ids = itertools.count(1)


def text(t: str):
    return SimpleNamespace(type="text", text=t)


def thinking(t: str):
    return SimpleNamespace(type="thinking", thinking=t, signature="sig")


def call(name: str, **args):
    return SimpleNamespace(type="tool_use", id=f"toolu_{next(_ids)}", name=name, input=args)


def response(*blocks, stop_reason: str | None = None):
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    usage = SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0)
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=usage)


class ScriptedLLM(LLM):
    """Replays a fixed list of actor responses; verifier/reflector answers are queued separately."""

    def __init__(self, config, turns, verdicts=None, lessons=None):
        super().__init__(config, client=SimpleNamespace())
        self.turns = list(turns)
        self.verdicts = list(verdicts or [])
        self.lessons = lessons or []
        self.seen_messages: list[list] = []
        self.structured_prompts: list[str] = []

    def respond(self, system, messages, tools, effort=None):
        self.seen_messages.append(list(messages))
        if not self.turns:
            return response(call("list_files"))  # keep busy: lets budget limits be tested
        turn = self.turns.pop(0)
        return turn(messages) if callable(turn) else turn

    def structured(self, system, prompt, schema, effort="medium"):
        self.structured_prompts.append(prompt)
        usage = SimpleNamespace(input_tokens=10, output_tokens=10)
        if "accept" in schema["properties"]:
            verdict = self.verdicts.pop(0) if self.verdicts else {"accept": True, "goal_met": True,
                                                                  "gaps": [], "feedback": "ok"}
            return verdict, usage
        return {"lessons": self.lessons}, usage


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


@pytest.fixture
def config(workspace):
    return AgentConfig(workspace=workspace, approval="auto", max_steps=12, max_cost_usd=0, web_search=False)

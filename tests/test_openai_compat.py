"""The free-provider backend, driven end to end through a fake OpenAI-compatible endpoint."""

import json

import pytest

from sentinel.agent import Agent
from sentinel.llm import LLMAuthError
from sentinel.openai_compat import OpenAICompatLLM, pick_gemini_model


class FakeEndpoint:
    """Mimics Gemini's OpenAI-compatible API: model listing, tool calls with thought signatures, JSON mode."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, method, path, body=None):
        self.requests.append((method, path, body))
        if path == "/models":
            return {"data": [{"id": f"models/{m}"} for m in (
                "gemini-2.0-flash", "gemini-3.6-flash", "gemini-3.6-flash-lite", "gemini-3.6-pro",
                "gemini-3.6-flash-image", "text-embedding-004")]}
        if body.get("response_format") or "JSON schema" in body["messages"][0]["content"]:
            schema_text = body["messages"][0]["content"]
            answer = {"accept": True, "goal_met": True, "gaps": [], "feedback": "verified"} \
                if '"accept"' in schema_text else {"lessons": ["Notes go in notes.txt"]}
            return {"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}]}
        return self.replies.pop(0)


def tool_reply(name, args, signature=None, call_id=None):
    call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    if call_id is None:
        del call["id"]  # some providers omit ids; the backend must invent one
    if signature:
        call["extra_content"] = {"google": {"thought_signature": signature}}
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]},
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 40}}


def test_full_run_on_a_free_provider(config, workspace, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "free-key")
    endpoint = FakeEndpoint([
        tool_reply("write_file", {"path": "notes.txt", "content": "hello"}, signature="sig-1", call_id="c1"),
        tool_reply("finish", {"success": True, "summary": "wrote notes.txt", "evidence": "created"}),
    ])
    llm = OpenAICompatLLM(config, "gemini", transport=endpoint)
    assert llm.model == "gemini-3.6-flash" and config.model == "gemini-3.6-flash"

    result = Agent(config, llm=llm).run("write notes.txt")

    assert result.status == "success" and (workspace / "notes.txt").read_text() == "hello"
    assert result.cost_usd == 0.0
    chat_calls = [b for m, p, b in endpoint.requests if p == "/chat/completions" and "tools" in b]
    first, second = chat_calls
    # tools are sent in OpenAI function format, without keywords Gemini rejects
    names = {t["function"]["name"] for t in first["tools"]}
    assert "write_file" in names and "web_search" not in names
    assert "additionalProperties" not in json.dumps(first["tools"])
    # the assistant turn is replayed exactly (thought signature kept), followed by the tool result
    replayed = [m for m in second["messages"] if m["role"] == "assistant"][0]
    assert replayed["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "sig-1"
    tool_msg = [m for m in second["messages"] if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "c1" and "Created notes.txt" in tool_msg["content"]


def test_missing_key_gives_signup_hint(config, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMAuthError, match="aistudio.google.com"):
        OpenAICompatLLM(config, "gemini", transport=FakeEndpoint([]))


def test_explicit_model_is_respected(config, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    config.model = "llama-3.3-70b-versatile"
    assert OpenAICompatLLM(config, "groq", transport=FakeEndpoint([])).model == "llama-3.3-70b-versatile"


def test_gemini_model_picker_prefers_newest_stable_flash():
    ids = ["gemini-2.5-flash", "gemini-3.6-flash-preview-09", "gemini-3.6-flash", "gemini-3.6-flash-lite",
           "gemini-3.6-pro", "gemini-3.6-flash-tts"]
    assert pick_gemini_model(ids) == "gemini-3.6-flash"
    assert pick_gemini_model(["gemini-3.6-pro"]) is None


def test_truncated_and_bad_json_arguments_are_handled():
    resp = OpenAICompatLLM.from_openai_response({"choices": [{"message": {"tool_calls": [
        {"id": "x", "function": {"name": "read_file", "arguments": "{\"path\": \"a"}}]}, "finish_reason": "length"}]})
    call = next(b for b in resp.content if b.type == "tool_use")
    assert resp.stop_reason == "max_tokens" and "__invalid_json__" in call.input


def test_overloaded_model_falls_back_to_next(config, monkeypatch):
    from sentinel.openai_compat import ModelUnavailable

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    seen = []

    def transport(method, path, body=None):
        if path == "/models":
            return {"data": [{"id": "models/gemini-3.8-flash"}, {"id": "models/gemini-3.5-flash"}]}
        seen.append(body["model"])
        if body["model"] == "gemini-3.8-flash":
            raise ModelUnavailable("HTTP 503: busy")
        return {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}

    llm = OpenAICompatLLM(config, "gemini", transport=transport)
    reply = llm.respond("s", [{"role": "user", "content": "hi"}], [])
    assert seen == ["gemini-3.8-flash", "gemini-3.5-flash"]
    assert llm.model == "gemini-3.5-flash" and reply.content[0].text == "hi"

"""Backend for free / OpenAI-compatible providers: Google Gemini, Groq, Ollama, or any custom endpoint.

The agent loop speaks one message format (Anthropic-style content blocks). This
module translates that history to the OpenAI chat-completions format on the way
out, and the reply back into blocks on the way in, so ``agent.py`` doesn't need
to know which provider it is talking to.

HTTP is done with the standard library so free use needs no extra packages.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .config import AgentConfig
from .llm import LLMAuthError, LLMError


@dataclass(frozen=True)
class Preset:
    base_url: str
    key_envs: tuple[str, ...]
    preferred_models: tuple[str, ...] = ()
    signup: str = ""


PRESETS: dict[str, Preset] = {
    "gemini": Preset(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        signup="https://aistudio.google.com/apikey",
    ),
    "groq": Preset(
        "https://api.groq.com/openai/v1",
        ("GROQ_API_KEY",),
        preferred_models=("openai/gpt-oss-120b", "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct",
                          "qwen/qwen3-32b", "openai/gpt-oss-20b"),
        signup="https://console.groq.com/keys",
    ),
    "ollama": Preset(os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/v1", ()),
    "openai-compatible": Preset(os.environ.get("SENTINEL_BASE_URL", ""), ("SENTINEL_API_KEY", "OPENAI_API_KEY")),
}

_STOP_REASONS = {"tool_calls": "tool_use", "function_call": "tool_use", "length": "max_tokens",
                 "content_filter": "refusal", "stop": "end_turn"}


def _strip_unsupported(schema: Any) -> Any:
    """Some providers (Gemini) reject JSON-schema keywords such as additionalProperties."""
    if isinstance(schema, dict):
        return {k: _strip_unsupported(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_strip_unsupported(v) for v in schema]
    return schema


def _version_key(model_id: str) -> tuple:
    match = re.search(r"gemini-(\d+(?:\.\d+)*)", model_id)
    nums = tuple(int(n) for n in match.group(1).split(".")) if match else ()
    unstable = any(tag in model_id for tag in ("preview", "exp"))
    return (nums, not unstable)


def rank_gemini_models(ids: list[str]) -> list[str]:
    """Usable Gemini models, best first: newest stable Flash, then previews, aliases, and Flash-Lite last."""
    skip = ("image", "tts", "audio", "live", "embed", "vision", "thinking", "learnlm", "8b", "omni")
    flash = [m for m in ids if "gemini" in m and "flash" in m and not any(s in m for s in skip)]
    full = sorted((m for m in flash if "lite" not in m and "latest" not in m), key=_version_key, reverse=True)
    stable = [m for m in full if _version_key(m)[1]]
    previews = [m for m in full if not _version_key(m)[1]]
    aliases = [m for m in flash if "latest" in m and "lite" not in m]
    lite = sorted((m for m in flash if "lite" in m and "latest" not in m), key=_version_key, reverse=True)
    return stable + previews + aliases + lite


def pick_gemini_model(ids: list[str]) -> str | None:
    ranked = rank_gemini_models(ids)
    return ranked[0] if ranked else None


class ModelUnavailable(LLMError):
    """This model is overloaded, out of free quota, or not offered: try another one."""


def _notice(text: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {text}", file=sys.stderr, flush=True)


class OpenAICompatLLM:
    """Drop-in replacement for ``LLM`` that talks to an OpenAI-compatible endpoint."""

    def __init__(self, config: AgentConfig, provider: str, base_url: str | None = None,
                 api_key: str | None = None, transport=None):
        if provider not in PRESETS:
            raise ValueError(f"unknown provider {provider!r}")
        self.config = config
        self.provider = provider
        self.preset = PRESETS[provider]
        self.base_url = (base_url or self.preset.base_url).rstrip("/")
        if not self.base_url:
            raise LLMAuthError("Set --base-url (or SENTINEL_BASE_URL) for the openai-compatible provider.")
        self.api_key = api_key or next((os.environ[k] for k in self.preset.key_envs if os.environ.get(k)), "")
        if self.preset.key_envs and not self.api_key:
            hint = f" Get a free key at {self.preset.signup}." if self.preset.signup else ""
            raise LLMAuthError(f"No API key for {provider}: set {self.preset.key_envs[0]} "
                               f"(environment or .env file).{hint}")
        self._transport = transport or self._http
        self.candidates = self._candidate_models(config.model)
        self.model = self.candidates[0]
        config.model = self.model

    # -- HTTP ----------------------------------------------------------------

    def _http(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": "sentinel-agent/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        last_error = ""
        for attempt in range(4):
            req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:600]
                if exc.code in (401, 403):
                    raise LLMAuthError(f"{self.provider} rejected the API key (HTTP {exc.code}): {detail}") from exc
                if exc.code == 404 and path.startswith("/chat"):
                    raise ModelUnavailable("model not available (HTTP 404)") from exc
                if exc.code in (429, 503) and attempt >= 1:
                    # Still overloaded / out of quota after one wait: let the caller try another model.
                    raise ModelUnavailable(f"HTTP {exc.code}: busy or out of free quota") from exc
                if exc.code == 429 or exc.code >= 500:
                    retry_after = exc.headers.get("retry-after")
                    wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() \
                        else 8 * (attempt + 1)
                    wait = min(wait, 60)
                    reason = "rate limit reached" if exc.code == 429 else f"server busy (HTTP {exc.code})"
                    _notice(f"{self.provider}: {reason}, waiting {wait:.0f}s and retrying...")
                    last_error = f"HTTP {exc.code}: {detail}"
                    time.sleep(wait)
                    continue
                raise LLMError(f"{self.provider} HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = str(exc)
                if self.provider == "ollama":
                    raise LLMError(f"Cannot reach Ollama at {self.base_url} - is `ollama serve` running?") from exc
                _notice(f"{self.provider}: connection problem ({exc}), retrying...")
                time.sleep(5 * (attempt + 1))
        raise LLMError(f"{self.provider} kept failing after retries: {last_error}")

    def _candidate_models(self, requested: str) -> list[str]:
        if requested and not requested.startswith("claude-"):
            return [requested]  # user chose explicitly
        try:
            listing = self._transport("GET", "/models")
            ids = [m["id"].removeprefix("models/") for m in listing.get("data", [])]
        except LLMAuthError:
            raise
        except LLMError:
            ids = []
        if self.provider == "gemini":
            ranked = rank_gemini_models(ids)
        else:
            ranked = [m for m in self.preset.preferred_models if m in ids] or ids[:1]
        if not ranked:
            raise LLMError(f"Could not find a usable model on {self.provider}; pass one with --model.")
        return ranked

    def _chat(self, body: dict) -> dict:
        """POST a chat request, moving to the next candidate model if the current one is unavailable."""
        tried: list[str] = []
        while True:
            body["model"] = self.model
            try:
                return self._transport("POST", "/chat/completions", body)
            except ModelUnavailable as exc:
                tried.append(self.model)
                nxt = next((m for m in self.candidates if m not in tried), None)
                if nxt is None:
                    raise LLMError(f"No {self.provider} model is available right now ({exc}). "
                                   "Try again in a few minutes.") from exc
                _notice(f"{self.model} unavailable ({exc}); switching to {nxt}")
                self.model = nxt
                self.config.model = nxt

    # -- format translation -------------------------------------------------

    @staticmethod
    def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for msg in messages:
            content = msg["content"]
            if msg["role"] == "user":
                if isinstance(content, str):
                    out.append({"role": "user", "content": content})
                    continue
                texts = []
                for block in content:
                    if block.get("type") == "tool_result":
                        out.append({"role": "tool", "tool_call_id": block["tool_use_id"],
                                    "content": str(block.get("content", ""))})
                    elif block.get("type") == "text":
                        texts.append(block["text"])
                if texts:
                    out.append({"role": "user", "content": "\n\n".join(texts)})
            elif msg["role"] == "assistant":
                raw = next((b.raw for b in content if getattr(b, "type", None) == "provider_raw"), None)
                if raw is not None:
                    out.append(raw)  # replay exactly (keeps provider extras such as thought signatures)
                    continue
                text = "".join(b.text for b in content if getattr(b, "type", None) == "text")
                calls = [{"id": b.id, "type": "function",
                          "function": {"name": b.name, "arguments": json.dumps(b.input)}}
                         for b in content if getattr(b, "type", None) == "tool_use"]
                entry: dict = {"role": "assistant", "content": text or None}
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
        return out

    @staticmethod
    def to_openai_tools(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": _strip_unsupported(t["input_schema"])}}
                for t in tools if "input_schema" in t]  # server-side Anthropic tools are skipped

    @staticmethod
    def from_openai_response(data: dict) -> SimpleNamespace:
        choice = (data.get("choices") or [{}])[0]
        message = dict(choice.get("message") or {})
        blocks: list[Any] = []
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            blocks.append(SimpleNamespace(type="thinking", thinking=reasoning))
        if message.get("content"):
            blocks.append(SimpleNamespace(type="text", text=message["content"]))

        raw_calls = []
        for call in message.get("tool_calls") or []:
            call = dict(call)
            call.setdefault("id", f"call_{uuid.uuid4().hex[:12]}")
            call.setdefault("type", "function")
            raw_calls.append(call)
            fn = call.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": fn.get("arguments")}
            blocks.append(SimpleNamespace(type="tool_use", id=call["id"], name=fn.get("name", ""), input=args))

        raw = {"role": "assistant", "content": message.get("content") or ""}
        if raw_calls:
            raw["tool_calls"] = raw_calls
            if not message.get("content"):
                raw["content"] = None
        blocks.append(SimpleNamespace(type="provider_raw", raw=raw))

        stop = _STOP_REASONS.get(choice.get("finish_reason") or "stop", "end_turn")
        if raw_calls:
            stop = "tool_use" if stop != "max_tokens" else stop
        usage = data.get("usage") or {}
        return SimpleNamespace(
            content=blocks, stop_reason=stop,
            usage=SimpleNamespace(input_tokens=usage.get("prompt_tokens", 0),
                                  output_tokens=usage.get("completion_tokens", 0)),
        )

    # -- the LLM interface used by the agent ----------------------------------

    def respond(self, system: str, messages: list, tools: list, effort: str | None = None):
        body = {
            "messages": self.to_openai_messages(system, messages),
            "tools": self.to_openai_tools(tools),
            "max_tokens": min(self.config.max_tokens, 8192),
        }
        return self.from_openai_response(self._chat(body))

    def structured(self, system: str, prompt: str, schema: dict, effort: str = "medium") -> tuple[dict, Any]:
        instructions = (f"{system}\n\nReply with a single JSON object, no other text, matching this JSON schema:\n"
                        f"{json.dumps(schema)}")
        body = {"max_tokens": 4096, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": prompt}]}
        try:
            data = self._chat(body)
        except ModelUnavailable:
            raise
        except LLMError:
            body.pop("response_format")  # not every model supports JSON mode
            data = self._chat(body)
        reply = self.from_openai_response(data)
        text = "".join(b.text for b in reply.content if b.type == "text")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"model did not return JSON: {text[:200]!r}")
        return json.loads(match.group(0)), reply.usage

    def cost_of(self, usage: Any) -> float:
        return 0.0  # free tiers / local models

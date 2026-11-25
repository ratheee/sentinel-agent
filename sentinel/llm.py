"""Thin wrapper around the Anthropic Messages API.

The agent loop lives in ``agent.py``; this module only knows how to make the
two kinds of call the agent needs:

* ``respond`` - a tool-using turn of the main reasoning loop
* ``structured`` - a one-shot call whose answer must match a JSON schema
  (used by the verifier and the post-run reflection)
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from .config import PRICING, AgentConfig

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusal(RuntimeError):
    """The model declined the request (stop_reason == "refusal")."""


class LLMError(RuntimeError):
    """A provider call failed (used by the non-Anthropic backends)."""


class LLMAuthError(LLMError):
    """Missing or rejected credentials: a setup problem the user must fix."""


class LLM:
    def __init__(self, config: AgentConfig, client: anthropic.Anthropic | None = None):
        self.config = config
        self.client = client or anthropic.Anthropic(max_retries=4)

    # -- calls ---------------------------------------------------------------

    def _create(self, **kwargs: Any):
        kwargs.setdefault("model", self.config.model)
        kwargs.setdefault("max_tokens", self.config.max_tokens)
        if self.config.refusal_fallback:
            # On a safety-classifier decline, the API re-runs the turn on a
            # fallback model inside the same call instead of just stopping.
            return self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs
            )
        return self.client.messages.create(**kwargs)

    def respond(self, system: str, messages: list, tools: list, effort: str | None = None):
        return self._create(
            system=system,
            messages=messages,
            tools=tools,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": effort or self.config.effort},
            # Auto-cache the growing conversation prefix: each step re-sends it.
            cache_control={"type": "ephemeral"},
        )

    def structured(self, system: str, prompt: str, schema: dict, effort: str = "medium") -> tuple[dict, Any]:
        response = self._create(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        if response.stop_reason == "refusal":
            raise LLMRefusal("model declined the structured request")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return json.loads(text), response.usage

    # -- accounting ----------------------------------------------------------

    def cost_of(self, usage: Any) -> float:
        prices = PRICING.get(self.config.model)
        if prices is None or usage is None:
            return 0.0
        p_in, p_out, p_read = prices
        tokens_in = getattr(usage, "input_tokens", 0) or 0
        tokens_out = getattr(usage, "output_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        return (
            tokens_in * p_in
            + tokens_out * p_out
            + cache_write * p_in * 1.25
            + cache_read * p_read
        ) / 1_000_000

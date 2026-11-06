"""
Thin wrapper over an Azure AI Foundry model deployment's OpenAI-compatible
chat-completions endpoint, with a BOUNDED client-side tool-calling loop.

Deliberately NOT using Foundry's stateful Agent Service (threads / server-
executed tools): every tool call here must be validated by
backend.agent.guardrails BEFORE it touches PaperBroker, which is simplest to
guarantee by running the tool-calling loop ourselves. This also makes the
whole thing trivial to mock in tests (one method: chat_with_tools).

Secrets pattern copied from backend/alerts/regime_alert.py::_env() — empty or
an un-expanded Azure Pipelines placeholder ("$(...)") is treated as unset.
Unlike the SMTP alert (optional, no-ops gracefully), a missing LLM config
means the agent CANNOT function — config_from_env() returning None must make
the caller fail closed (refuse to start), not silently skip a cycle.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_API_VERSION = "2024-10-21"
# gpt-4o-mini is marked Deprecated in the Microsoft Foundry model catalog
# (retires 2027-04-14, no official replacement listed) — gpt-5-mini is the
# current GA "mini" tier (retires 2027-02-09, ~$0.25/$2.00 per 1M in/out
# tokens on Global Standard, cheaper than gpt-5.4-mini's $0.75/$4.50).
# Re-check the model catalog's lifecycle column before changing this, since
# Azure's "mini" line keeps moving (checked 2026-10-28).
DEFAULT_DEPLOYMENT = "gpt-5-mini"


def _env(name: str) -> Optional[str]:
    """Env var, treating empty or un-expanded Azure placeholders ($(VAR)) as unset."""
    v = os.getenv(name)
    if not v or v.startswith("$("):
        return None
    return v


@dataclass
class AgentLLMConfig:
    endpoint: str
    api_key: str
    deployment: str = DEFAULT_DEPLOYMENT
    api_version: str = DEFAULT_API_VERSION


def config_from_env() -> Optional[AgentLLMConfig]:
    """None if Azure AI Foundry isn't configured — caller must fail closed."""
    endpoint, api_key = _env("AZURE_AI_FOUNDRY_ENDPOINT"), _env("AZURE_AI_FOUNDRY_API_KEY")
    if not (endpoint and api_key):
        return None
    return AgentLLMConfig(
        endpoint=endpoint, api_key=api_key,
        deployment=_env("AZURE_AI_FOUNDRY_DEPLOYMENT") or DEFAULT_DEPLOYMENT,
        api_version=_env("AZURE_AI_FOUNDRY_API_VERSION") or DEFAULT_API_VERSION,
    )


@dataclass
class ChatResult:
    text: str = ""
    tool_trace: List[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    error: Optional[str] = None


class AgentLLMClient:
    """Call via asyncio.to_thread from the engine — this does blocking network I/O."""

    def __init__(self, cfg: AgentLLMConfig):
        self.cfg = cfg
        self._client = None  # lazy, so importing this module never requires the SDK/network

    def _get_client(self):
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=self.cfg.endpoint,
                api_key=self.cfg.api_key,
                api_version=self.cfg.api_version,
            )
        return self._client

    def chat_with_tools(self, messages: List[dict], tools: List[dict],
                        tool_dispatch: Dict[str, Callable[[dict], dict]],
                        max_tool_turns: int = 4) -> ChatResult:
        """
        Bounded ReAct-style loop: call the model, execute any requested tool
        via `tool_dispatch`, feed the result back, repeat until the model
        returns a plain (non-tool) response or `max_tool_turns` is hit.

        Never raises on Azure/network errors — returns a ChatResult with
        `.error` set so the engine can fail closed to a `hold` for the cycle.
        Tool EXECUTION exceptions are caught per-call and fed back to the
        model as a tool error message (so one bad tool call doesn't blow up
        the whole cycle), and are also recorded in `tool_trace`.
        """
        start = time.monotonic()
        result = ChatResult()
        convo = list(messages)
        try:
            client = self._get_client()
        except Exception as exc:
            result.error = f"client init failed: {exc}"
            return result

        try:
            for _ in range(max_tool_turns):
                response = client.chat.completions.create(
                    model=self.cfg.deployment, messages=convo, tools=tools,
                )
                usage = getattr(response, "usage", None)
                if usage:
                    result.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    result.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

                choice = response.choices[0].message
                tool_calls = getattr(choice, "tool_calls", None)
                if not tool_calls:
                    result.text = choice.content or ""
                    break

                convo.append({"role": "assistant", "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ]})
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        import json
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception as exc:
                        args = {}
                        tool_output = {"error": f"malformed tool arguments: {exc}"}
                    else:
                        handler = tool_dispatch.get(name)
                        if handler is None:
                            tool_output = {"error": f"unknown tool '{name}'"}
                        else:
                            try:
                                tool_output = handler(args)
                            except Exception as exc:
                                tool_output = {"error": f"tool '{name}' raised: {exc}"}

                    result.tool_trace.append({"name": name, "args": args, "output": tool_output})
                    import json
                    convo.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(tool_output, default=str)})
            else:
                result.text = result.text or "(max tool turns reached without a final answer)"
        except Exception as exc:
            result.error = str(exc)

        result.latency_ms = int((time.monotonic() - start) * 1000)
        return result

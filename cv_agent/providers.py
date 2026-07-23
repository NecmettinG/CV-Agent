"""LLM providers behind a single structured-output (tool-use) interface.

The extraction step (:mod:`cv_agent.extract`) needs one thing from a model:
"call this tool with arguments shaped like the CV schema". Different vendors
spell that differently, so each provider here adapts its own SDK to a tiny
uniform surface the extractor drives:

    provider.call(messages, ...)   -> ToolCall(arguments=..., call_id=..., ...)
    provider.user_message(text)    -> a message to start the conversation
    provider.assistant_message(c)  -> replay the model's (invalid) tool call
    provider.error_message(c, err) -> hand the validation error back for a retry

That last pair is what lets the extractor's repair loop stay provider-agnostic.

Two adapters cover a lot of ground:

* :class:`AnthropicProvider` - native ``anthropic`` SDK. Most reliable tool use;
  the paid default.
* :class:`OpenAICompatProvider` - the ``openai`` SDK pointed at any
  OpenAI-compatible ``base_url``. That single adapter reaches OpenAI **and** the
  free / low-cost options a no-budget user needs: Google Gemini (Flash), Kimi
  (Moonshot), OpenRouter (one key, many models incl. ``:free`` ones), Groq, ...

Named :data:`PRESETS` wire up base_url + a sensible default model + which env var
holds the key, so callers just pass ``provider="gemini"`` (say) and an API key.

NOTE: model IDs move fast and free tiers change. The presets' ``default_model``
values are only a starting point - always let the user override with ``model=``.
API keys are never hardcoded: they are passed in as a parameter (or read from the
documented env var), never written to disk.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# --------------------------------------------------------------------------- #
# The uniform result of asking a model to call our tool.
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """One model turn, normalized across providers.

    ``arguments`` is the tool-call payload (a dict to validate) or ``None`` if the
    model answered without calling the tool. ``text`` carries any plain-text the
    model produced instead (useful for error messages).
    """

    tool_name: str
    arguments: Optional[Dict[str, Any]]
    call_id: Optional[str] = None
    text: Optional[str] = None
    raw: Any = None


# --------------------------------------------------------------------------- #
# Provider base + the two concrete adapters.
# --------------------------------------------------------------------------- #
class Provider:
    """Base class: an SDK client + a chosen model, adapted to a tool-call surface."""

    #: human-readable id, e.g. "anthropic" or "openrouter (openai-compatible)".
    label: str = "provider"

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def call(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: str,
        tool_name: str,
        tool_description: str,
        parameters: Dict[str, Any],
        max_tokens: int,
        tool_choice: Any = None,
    ) -> ToolCall:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def user_message(text: str) -> Dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_message(self, call: ToolCall) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def error_message(self, call: ToolCall, error: str) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


class AnthropicProvider(Provider):
    label = "anthropic"

    def call(
        self,
        messages,
        *,
        system,
        tool_name,
        tool_description,
        parameters,
        max_tokens,
        tool_choice=None,
    ) -> ToolCall:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=[{"name": tool_name, "description": tool_description, "input_schema": parameters}],
            tool_choice=tool_choice or {"type": "tool", "name": tool_name},
            messages=messages,
        )
        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                return ToolCall(tool_name, block.input, call_id=block.id, raw=response.content)
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        return ToolCall(tool_name, None, text="".join(text_parts) or None, raw=response.content)

    def assistant_message(self, call: ToolCall) -> Dict[str, Any]:
        return {"role": "assistant", "content": call.raw}

    def error_message(self, call: ToolCall, error: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.call_id,
                    "is_error": True,
                    "content": error,
                }
            ],
        }


#: OpenAI's reasoning models (o-series, gpt-5) reject ``max_tokens`` and require
#: ``max_completion_tokens``; every other OpenAI-compatible host takes ``max_tokens``.
_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _token_limit_param(model: str) -> str:
    """The right output-cap kwarg for ``model``: ``max_completion_tokens`` only for
    native OpenAI reasoning ids, ``max_tokens`` everywhere else.

    Routing hubs use ``vendor/model`` ids (e.g. OpenRouter's ``openai/gpt-5``) and
    normalize ``max_tokens`` themselves, so a slash in the id means "not native
    OpenAI" - leave those on ``max_tokens``.
    """
    if "/" in model:
        return "max_tokens"
    return "max_completion_tokens" if model.lower().startswith(_OPENAI_REASONING_PREFIXES) else "max_tokens"


class OpenAICompatProvider(Provider):
    """Any OpenAI-compatible chat-completions endpoint (OpenAI, Gemini, Kimi, ...)."""

    def __init__(self, client: Any, model: str, label: str = "openai-compatible") -> None:
        super().__init__(client, model)
        self.label = label

    def call(
        self,
        messages,
        *,
        system,
        tool_name,
        tool_description,
        parameters,
        max_tokens,
        tool_choice=None,
    ) -> ToolCall:
        full = [{"role": "system", "content": system}, *messages]
        # Compatibility shims for the fussier / free OpenAI-compatible backends:
        #  * sanitize the schema (Gemini rejects additionalProperties / anyOf-null / ...),
        #  * force *a* tool call with the portable "required" rather than naming the
        #    function (some endpoints reject the specific-function form),
        #  * spell the token cap the way this model expects it.
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": full,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": sanitize_schema(parameters),
                    },
                }
            ],
            "tool_choice": tool_choice or "required",
            _token_limit_param(self.model): max_tokens,
        }
        response = self.client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ToolCall(
                tool_name, None, raw=response,
                text="provider returned no choices (blocked by a content filter or upstream error).",
            )
        message = choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            tc = tool_calls[0]
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                return ToolCall(
                    tool_name, None, call_id=tc.id, raw=message,
                    text=f"(model returned invalid JSON arguments: {exc}) {tc.function.arguments!r}",
                )
            return ToolCall(tool_name, args, call_id=tc.id, raw=message)
        return ToolCall(tool_name, None, text=getattr(message, "content", None), raw=message)

    def assistant_message(self, call: ToolCall) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.tool_name,
                        "arguments": json.dumps(call.arguments or {}),
                    },
                }
            ],
        }

    def error_message(self, call: ToolCall, error: str) -> Dict[str, Any]:
        return {"role": "tool", "tool_call_id": call.call_id, "content": error}


# --------------------------------------------------------------------------- #
# Presets: name -> how to reach it + a starting model.
# --------------------------------------------------------------------------- #
@dataclass
class Preset:
    kind: str  # "anthropic" | "openai"
    default_model: str
    env_var: str
    base_url: Optional[str] = None
    cost: str = ""  # short human note for menus

    @property
    def free(self) -> bool:
        return "free" in self.cost.lower()


PRESETS: Dict[str, Preset] = {
    "anthropic": Preset(
        "anthropic", "claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY",
        cost="Paid, cheap + fast (default). Reliable tool use; Opus/Sonnet are "
             "stronger for hard CVs - pass --model / pick it in Settings.",
    ),
    "openai": Preset(
        "openai", "gpt-4o-mini", "OPENAI_API_KEY",
        cost="Paid.",
    ),
    "gemini": Preset(
        "openai", "gemini-2.5-flash", "GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        cost="FREE tier via Google AI Studio (aistudio.google.com). Good no-budget pick.",
    ),
    "moonshot": Preset(
        "openai", "kimi-k2-0711-preview", "MOONSHOT_API_KEY",
        base_url="https://api.moonshot.ai/v1",
        cost="Low cost (Kimi). Strong tool use.",
    ),
    "openrouter": Preset(
        "openai", "deepseek/deepseek-chat-v3-0324:free", "OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        cost="One key, many models incl. ':free' ones (Kimi, Gemini, DeepSeek). Best no-budget hub.",
    ),
    "groq": Preset(
        "openai", "llama-3.3-70b-versatile", "GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        cost="FREE tier, very fast.",
    ),
}

#: Convenient aliases.
_ALIASES = {"kimi": "moonshot", "google": "gemini"}


def resolve_provider_name(name: str) -> str:
    key = name.strip().lower()
    return _ALIASES.get(key, key)


def _anthropic_client(api_key: Optional[str]) -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise ImportError("`pip install anthropic` to use provider='anthropic'.") from exc
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _openai_client(api_key: Optional[str], base_url: Optional[str], env_var: str) -> Any:
    try:
        import openai
    except ImportError as exc:  # pragma: no cover
        raise ImportError("`pip install openai` to use OpenAI-compatible providers.") from exc
    key = api_key or os.environ.get(env_var)
    if not key:
        raise ValueError(f"No API key: pass api_key=... or set {env_var}.")
    return openai.OpenAI(api_key=key, base_url=base_url)


def build_provider(
    provider: Union[str, Provider],
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Any = None,
) -> Provider:
    """Resolve ``provider`` (a name or a ready :class:`Provider`) into a Provider.

    Args:
        provider: a preset name (see :data:`PRESETS`, e.g. ``"anthropic"``,
            ``"gemini"``, ``"openrouter"``) / alias (``"kimi"``, ``"google"``),
            or an already-built :class:`Provider` (returned as-is).
        api_key: the key, passed straight to the SDK (never stored). Falls back to
            the preset's env var for OpenAI-compatible providers.
        model: model id; defaults to the preset's ``default_model``.
        base_url: override the preset's endpoint (e.g. a different OpenAI-compat host).
        client: inject a pre-built SDK client (mainly for testing/reuse).
    """
    if isinstance(provider, Provider):
        return provider

    name = resolve_provider_name(provider)
    preset = PRESETS.get(name)
    if preset is None:
        raise ValueError(
            f"Unknown provider {provider!r}. Available: {', '.join(sorted(PRESETS))} "
            f"(aliases: {', '.join(sorted(_ALIASES))})."
        )

    chosen_model = model or preset.default_model
    if preset.kind == "anthropic":
        return AnthropicProvider(client or _anthropic_client(api_key), chosen_model)
    resolved_url = base_url or preset.base_url
    return OpenAICompatProvider(
        client or _openai_client(api_key, resolved_url, preset.env_var),
        chosen_model,
        label=f"{name} (openai-compatible)",
    )


def dereference_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline every ``$ref`` and drop ``$defs`` -> a self-contained JSON schema.

    Anthropic/OpenAI accept ``$ref``, but Gemini and various small/free models are
    fussier. Inlining maximizes compatibility. Our schema has no recursive models;
    a guard still breaks any future cycle into a bare ``{"type": "object"}``.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, seen: frozenset) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = node["$ref"].split("/")[-1]
                if target in seen:
                    return {"type": "object"}
                return resolve(copy.deepcopy(defs.get(target, {})), seen | {target})
            return {k: resolve(v, seen) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v, seen) for v in node]
        return node

    return resolve({k: v for k, v in schema.items() if k != "$defs"}, frozenset())


#: Schema keys the fussiest OpenAI-compatible tool-calling backends reject
#: (Gemini's especially). Dropping them is loss-free: Pydantic re-imposes every
#: real constraint when we validate the model's output in :mod:`cv_agent.schema`.
_SCHEMA_DROP_KEYS = frozenset({"title", "default", "additionalProperties", "$schema"})


def sanitize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a JSON schema to the conservative subset the fussiest OpenAI-compatible
    tool-calling backends accept - Gemini's chief among them.

    Two transforms, both harmless for us:

    * drop ``title`` / ``default`` / ``additionalProperties`` / ``$schema`` keys;
    * collapse Pydantic's ``Optional[T]`` (an ``anyOf`` with a ``{"type": "null"}``
      branch) down to plain ``T`` - these backends don't model a null branch.

    Anthropic accepts the full schema, so only :class:`OpenAICompatProvider` runs
    this. It only shapes what we *advertise* to the model; the authoritative
    constraints still live in :mod:`cv_agent.schema`. Assumes ``$ref`` are already
    inlined (see :func:`dereference_schema`).
    """
    if isinstance(schema, list):
        return [sanitize_schema(v) for v in schema]
    if not isinstance(schema, dict):
        return schema

    options = schema.get("anyOf")
    if isinstance(options, list):
        non_null = [o for o in options if not (isinstance(o, dict) and o.get("type") == "null")]
        if len(non_null) == 1:
            merged = {k: v for k, v in schema.items() if k != "anyOf"}
            merged.update(non_null[0])  # the surviving branch's type/keys win
            return sanitize_schema(merged)

    return {k: sanitize_schema(v) for k, v in schema.items() if k not in _SCHEMA_DROP_KEYS}

"""LLM extraction: raw CV text -> a validated :class:`~cv_agent.schema.CV`.

This is the bridge between the input side (``cv_agent.parsers.*`` turn a PDF /
DOCX / paste into text) and the output side (``cv_agent.render`` turns a CV into
a PDF). It asks an LLM to read the text and fill in our schema.

Crucially it does **not** say "please reply in JSON" and then hope. It uses
*structured output via tool use*:

    1. Turn the Pydantic schema into a tool's parameter schema
       (``CV.model_json_schema()``, then dereferenced for wide model support).
    2. Force the model to call that one tool, so the response is tool-call
       arguments already shaped like the schema - not free text.
    3. Validate those arguments into a real ``CV``. On failure, hand the error
       back to the model and let it repair its own call, up to
       ``max_repair_attempts`` times.

The model never sees a Python object and we never trust its raw output: the
Pydantic layer in :mod:`cv_agent.schema` is the gate everything must pass.

**Multi-provider / model selection.** The provider-specific bits live in
:mod:`cv_agent.providers`; this module only holds the prompt, the tool schema,
and the provider-agnostic repair loop. Pick a provider by name - ``"anthropic"``
(paid default), or free/low-cost options for no-budget users: ``"gemini"``
(Google AI Studio free tier), ``"openrouter"`` (many ``:free`` models, incl.
Kimi/Gemini/DeepSeek), ``"groq"``, ``"moonshot"`` (Kimi), ``"openai"`` - and the
model with ``model=``. API keys are passed in as a parameter (or read from the
provider's documented env var); never hardcoded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from cv_agent.providers import Provider, build_provider, dereference_schema
from cv_agent.schema import CV

DEFAULT_MAX_TOKENS = 16000  # roomy: a dense 2-page CV's JSON must not truncate
TOOL_NAME = "record_cv"
TOOL_DESCRIPTION = (
    "Record the structured contents of the CV. Call this exactly once, filling "
    "every field you can support from the CV text."
)

SYSTEM_PROMPT = (
    "You are a meticulous CV parser. You receive the raw text of ONE person's CV "
    "(extracted from a PDF, DOCX, or pasted in) and must transcribe it into a "
    f"structured record by calling the `{TOOL_NAME}` tool exactly once.\n\n"
    "Rules:\n"
    "- COMPLETENESS IS CRITICAL. Transcribe the ENTIRE CV: every section and every "
    "line of detail. Do NOT summarize, shorten, condense, or skip anything. A CV may "
    "run two or more pages - capture all of it. Dropping content is a failure.\n"
    "- The record has three parts: the header (name, headline, summary, contact); the "
    "typed `experience` and `education` lists; and `sections` - a list holding EVERY "
    "OTHER section of the CV, in the CV's own order (Skills, Projects, Languages, "
    "Certificates, Community, Volunteering, Interests, References, Personal Details, ...).\n"
    "- Each section KEEPS ITS OWN HEADING: set `title` to the section's heading exactly "
    "as written in the CV, in its language (e.g. 'Community', 'Projeler', 'Referanslar'). "
    "Do NOT rename a section to a different concept - if the CV says 'Community', the "
    "title is 'Community', NOT 'Interests'. Only if a block truly has no heading, give it "
    "a short faithful one.\n"
    "- For each section pick a `kind` and fill the matching field:\n"
    "    * 'list' - a simple bullet list; one string per bullet in `bullets` (language "
    "lists, interests, community lines, personal details).\n"
    "    * 'skills' - a skills list (rendered in two columns); use `bullets`. Keep each "
    "skill line as written, e.g. 'Java: Spring Boot, JPA'.\n"
    "    * 'entries' - titled entries in `entries`; use for projects, references, "
    "certificates, competitions, awards. Each entry: `title` (bold lead - a project/"
    "certificate/award name, or a reference's person name), optional `detail` (a "
    "description, or a reference's 'Title, Company'), optional `date_range`; for a "
    "reference put phone in `phone` and email in `email` (verbatim); use `url` to link "
    "the title (e.g. a project repo) and `links` for any extra links.\n"
    "    * 'text' - a single paragraph; put it in `text`.\n"
    "- Experience/education entries: record every descriptive line. Put each separate "
    "line/bullet as its own string in `highlights`; use `description` only for one flowing "
    "paragraph; put a role's explicit technologies line into `tech_stack`.\n"
    "- Experience shape: a normal role sets `title` and leaves `sub_roles` empty. An "
    "umbrella employer (an agency hosting several client roles) sets `sub_roles` and "
    "leaves its own top-level `title` empty.\n"
    "- `headline`: fill it ONLY if the CV literally shows a short title/role line under "
    "the name. If there is no such line, leave `headline` empty - do NOT infer or invent "
    "one from the person's degree or job titles.\n"
    "- `summary`: the profile / objective / cover-letter paragraph, transcribed faithfully "
    "(do not shorten it).\n"
    "- Set `language` to the ISO 639-1 code of the CV's primary language ('tr', 'en', "
    "...); it sets the EXPERIENCE / EDUCATION heading language.\n"
    "- Do NOT fabricate URLs. Fill a `url` or a link only when the text contains a REAL "
    "one. Hyperlinks appear inline as `anchor text <https://the-real-url>`; use that exact "
    "URL, with the anchor as the label. If the CV shows 'LinkedIn'/'GitHub' (or any label) "
    "as plain text with NO url, do NOT invent one - omit the link entirely.\n"
    "- Use ONLY information actually present. Never invent, guess, or embellish names, "
    "employers, dates, metrics, or descriptions. Do not reformat phone numbers.\n"
    "- Transcribe wording faithfully. You may fix obvious extraction artifacts (broken "
    "spacing, split words) and trim whitespace, but do not rewrite, summarize, or translate.\n"
    "- Omit optional fields you cannot fill. Never insert empty strings, 'N/A', or "
    "placeholder text.\n"
    "- Preserve the order the CV presents things in (usually most-recent first)."
)


class ExtractionError(RuntimeError):
    """Raised when the model won't call the tool or its output can't be validated."""


def cv_tool_parameters() -> Dict[str, Any]:
    """The tool's parameter schema: the CV schema, dereferenced for portability."""
    return dereference_schema(CV.model_json_schema())


def _user_message(text: str, extra_instructions: Optional[str]) -> str:
    parts = ["Here is the raw CV text. Transcribe it by calling the tool."]
    if extra_instructions:
        parts.append("Additional instructions: " + extra_instructions)
    parts += ["---- BEGIN CV TEXT ----", text, "---- END CV TEXT ----"]
    return "\n\n".join(parts)


def _format_errors(exc: Optional[ValidationError]) -> str:
    """Compact, model-readable rendering of pydantic validation errors."""
    if exc is None:
        return "(unknown validation error)"
    lines: List[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        lines.append(f"- {loc}: {err.get('msg')}")
    return "\n".join(lines)


def extract_cv(
    text: str,
    *,
    provider: Union[str, Provider] = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Any = None,
    system_prompt: Optional[str] = None,
    extra_instructions: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_repair_attempts: int = 2,
    tool_choice: Any = None,
) -> CV:
    """Extract ``text`` into a validated :class:`CV` via forced tool use.

    Args:
        text: raw CV text (typically from ``cv_agent.parsers.*``). Keep the inline
            ``anchor <url>`` link annotations - the model reads real URLs from them.
        provider: provider name (``"anthropic"``, ``"gemini"``, ``"openrouter"``,
            ``"groq"``, ``"moonshot"``/``"kimi"``, ``"openai"``, ...) or a ready
            :class:`cv_agent.providers.Provider`. See ``providers.PRESETS``.
        model: model id; defaults to the provider preset's default. This is the
            user-facing model selection.
        api_key: provider API key, passed straight through (never stored). For
            OpenAI-compatible providers it falls back to the preset's env var.
        base_url: override the provider endpoint (advanced / self-hosted).
        client: inject a pre-built SDK client (reuse / testing).
        system_prompt: override the default extraction system prompt.
        extra_instructions: extra guidance appended to the user message.
        max_tokens: output token cap.
        max_repair_attempts: how many times to feed a validation error back and
            let the model correct its tool call (0 disables repair).
        tool_choice: override the provider's default forced tool selection.

    Returns:
        A validated ``CV``.

    Raises:
        ExtractionError: if the model never calls the tool, or its arguments still
            fail validation after all repair attempts.
        ValueError: if ``text`` is empty or the provider name is unknown.
    """
    if not text or not text.strip():
        raise ValueError("`text` is empty; nothing to extract.")

    prov = build_provider(provider, api_key=api_key, model=model, base_url=base_url, client=client)
    system = system_prompt or SYSTEM_PROMPT
    parameters = cv_tool_parameters()
    messages: List[Dict[str, Any]] = [prov.user_message(_user_message(text, extra_instructions))]

    last_error: Optional[ValidationError] = None
    for attempt in range(max_repair_attempts + 1):
        call = prov.call(
            messages,
            system=system,
            tool_name=TOOL_NAME,
            tool_description=TOOL_DESCRIPTION,
            parameters=parameters,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
        )
        if call.arguments is None:
            detail = f" Model said: {call.text!r}" if call.text else ""
            raise ExtractionError(f"{prov.label} did not return a {TOOL_NAME!r} tool call.{detail}")

        try:
            return CV.model_validate(call.arguments)
        except ValidationError as exc:
            last_error = exc
            if attempt >= max_repair_attempts:
                break
            messages.append(prov.assistant_message(call))
            messages.append(
                prov.error_message(
                    call,
                    "The tool arguments failed validation:\n"
                    + _format_errors(exc)
                    + f"\n\nCall {TOOL_NAME} again with corrected arguments, keeping "
                    "everything that was already valid.",
                )
            )

    raise ExtractionError(
        f"CV failed validation after {max_repair_attempts} repair attempt(s):\n"
        + _format_errors(last_error)
    ) from last_error


if __name__ == "__main__":  # pragma: no cover
    # Offline smoke test: no API key, no network. A fake Anthropic-shaped client
    # returns an INVALID tool call first (missing required date_range) then a
    # VALID one, exercising the tool schema, tool-call parsing, and repair loop.
    import types

    def _tool_use(payload, block_id):
        return types.SimpleNamespace(type="tool_use", name=TOOL_NAME, input=payload, id=block_id)

    class _FakeAnthropic:
        def __init__(self, payloads):
            self._responses = [
                types.SimpleNamespace(content=[_tool_use(p, f"toolu_{i}")], stop_reason="tool_use")
                for i, p in enumerate(payloads)
            ]
            self._i = 0
            self.messages = self

        def create(self, **_):
            resp = self._responses[self._i]
            self._i += 1
            return resp

    params = cv_tool_parameters()
    print(f"tool schema: type={params.get('type')!r}, {len(params.get('properties', {}))} top-level "
          f"properties, refs inlined ($defs present: {'$defs' in params}).")

    invalid = {"name": "Jordan Mercer", "experience": [{"company": "PayNova"}]}  # no date_range
    valid = {
        "name": "Jordan Mercer",
        "contact": {"location": "Berlin", "links": [{"label": "GitHub", "url": "https://github.com/example/jordan"}]},
        "experience": [{
            "company": "PayNova", "title": "Software Team Lead",
            "date_range": {"start": {"year": 2024, "month": 6}, "current": True},
            "description": "Led the payments platform team.",
        }],
    }

    cv = extract_cv("(fake CV text)", client=_FakeAnthropic([invalid, valid]), max_repair_attempts=2)
    print("repair loop recovered ->", cv.name, "|", cv.experience[0].company,
          "| link:", cv.contact.links[0].url)
    assert cv.name == "Jordan Mercer" and cv.experience[0].company == "PayNova"
    print("OK")

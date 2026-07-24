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

import re
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
    "description, or a reference's 'Title, Company'), optional `date_range`. For a "
    "reference, put phone in `phone`, email in `email`, and any website/profile in "
    "`links` (all verbatim) - do NOT put a reference's website in `url` (that hides it "
    "behind the name). Use `url` ONLY to make a project/certificate title itself a link. "
    "For a project entry, also fill `tech_stack` with the technologies/tools it used "
    "(named anywhere in its text), exactly as you would for an experience role.\n"
    "    * 'text' - a single paragraph; put it in `text`.\n"
    "  If a section ends with a short standalone note (e.g. 'References available on "
    "request.', a validation-letter remark), put that note in the section's `note` field.\n"
    "- Experience/education entries: record every descriptive line. Put each separate "
    "line/bullet as its own string in `highlights`; use `description` only for one flowing "
    "paragraph.\n"
    "- For EVERY experience role AND sub-role, fill `tech_stack` with the technologies, "
    "tools, frameworks, platforms and programming languages used in that role. Extract them "
    "even when they are embedded inside the description or the bullets - not only when the "
    "CV prints a separate technologies line. Include only technologies actually named in "
    "that role's text; leave `tech_stack` empty only when the role names none.\n"
    "- Experience shape: a normal role sets `title` and leaves `sub_roles` empty. An "
    "umbrella employer (an agency hosting several client roles) sets `sub_roles` and "
    "leaves its own top-level `title` empty.\n"
    "- `headline`: fill it ONLY if the CV literally shows a short title/role line under "
    "the name. If there is no such line, leave `headline` empty - do NOT infer or invent "
    "one from the person's degree or job titles.\n"
    "- `summary`: the profile / objective / cover-letter / about paragraph shown at the top, "
    "transcribed faithfully (do not shorten it). Put this text in `summary` even when the CV "
    "gives it a heading (e.g. 'ÖNYAZI', 'Cover Letter', 'Profile', 'About') - do NOT make it a "
    "section.\n"
    "- Set `language` to the ISO 639-1 code of the CV's primary language ('tr', 'en', "
    "...); it sets the EXPERIENCE / EDUCATION heading language.\n"
    "- NEVER invent a URL. A link is allowed ONLY when the source shows the URL explicitly - "
    "it appears inline as `anchor text <https://the-real-url>`; copy that exact URL and use "
    "the anchor as the label. If a label like 'LinkedIn', 'GitHub', 'Portfolio' or 'Website' "
    "appears with NO `<...>` URL after it, record NOTHING for it: do not add a link and do "
    "NOT guess 'https://linkedin.com', 'https://github.com', or any similar URL. A profile "
    "named without a URL is not a link.\n"
    "- Use ONLY information actually present. Never invent, guess, or embellish names, "
    "employers, dates, metrics, or descriptions. Do not reformat phone numbers.\n"
    "- Do NOT fill timeline gaps. If the dates leave an unexplained gap between roles (a "
    "period with no listed job, a career break, time studying), transcribe the CV as-is and "
    "leave the gap - never invent a role, employer, title, or dates to bridge it. Record "
    "exactly the experiences the CV lists, no more.\n"
    "- Transcribe wording faithfully. You may fix obvious extraction artifacts (broken "
    "spacing, split words) and trim whitespace, but do not rewrite, summarize, or translate.\n"
    "- Omit optional fields you cannot fill. Never insert empty strings, 'N/A', or "
    "placeholder text.\n"
    "- Preserve the order the CV presents things in (usually most-recent first)."
)


# --------------------------------------------------------------------------- #
# BUILD mode: turn a free-form self-description (a brain-dump) into a CV. Same
# anti-fabrication spine as extraction, but it may organize/rephrase loose prose.
# --------------------------------------------------------------------------- #
BUILD_SYSTEM_PROMPT = (
    "You help a person BUILD their CV from a free-form description they write about "
    "themselves - a brain-dump, often first person ('I worked at...', 'I know...'). Turn it "
    f"into a structured CV by calling the `{TOOL_NAME}` tool exactly once.\n\n"
    "Your job is to ORGANIZE and lightly REPHRASE what they tell you into a clean CV - never "
    "to invent facts.\n\n"
    "Rules:\n"
    "- Use ONLY what the person states. NEVER add a skill, employer, job title, date, degree, "
    "metric, certification, or achievement they did not mention. If a detail is missing (no "
    "dates for a job, no email), leave that field empty - do NOT guess or fill it in. "
    "Inventing anything is a failure.\n"
    "- You MAY rephrase their prose into concise CV wording (turn 'I made checkout faster' into "
    "a bullet 'Improved checkout performance'), convert first person to CV style, fix grammar, "
    "and organize loose text into the right places. This rephrasing must add NO new facts.\n"
    "- `name` is required: use the name they give. Structure the rest as: header (name, "
    "headline, summary, contact) + typed `experience` and `education` lists + `sections` "
    "(Skills, Projects, Languages, Certificates, Interests, ...). Group each described job into "
    "one experience entry (company, title, dates, description/highlights); each school into an "
    "education entry; skills into a 'skills' section; and so on. For a project you "
    "describe, fill its `tech_stack` with the technologies you say it used.\n"
    "- For every role, fill `tech_stack` with the technologies/tools the person says they used "
    "in it (only those they name).\n"
    "- `summary`: you MAY compose a short professional summary, but ONLY by rephrasing facts the "
    "person stated - add no seniority, adjectives, or claims they did not make. Omit it if they "
    "gave nothing to summarize.\n"
    "- `headline`: set it only if the person states a clear role/title for themselves; never "
    "invent one, and do not add 'Senior'/'Lead'/etc. unless they said so.\n"
    "- Contact: capture email, phone, location and any profile links (LinkedIn/GitHub/site) "
    "exactly as written. NEVER invent a URL or email - if they name a profile with no URL, "
    "record nothing for it.\n"
    "- Set `language` to the ISO 639-1 code of the language they wrote in ('en', 'tr', ...).\n"
    "- Omit optional fields you cannot fill. Never insert placeholders like 'N/A'. Keep their "
    "meaning faithful; do not exaggerate."
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


def _norm_url(u: str) -> str:
    """Normalize a URL for a lenient 'is this in the source?' check: drop the
    scheme, a leading ``www.`` and any trailing slash, then lowercase."""
    u = u.strip().lower()
    u = re.sub(r"^[a-z][a-z0-9+.\-]*://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u.rstrip("/")


#: URLs as they appear in the source: parser link markers ``<...>``, explicit
#: http(s) URLs, and bare domains (+ optional path).
_SOURCE_URL_RE = re.compile(
    r"<([^>\s]+)>"
    r"|(https?://[^\s<>()\"']+)"
    r"|([a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)+(?:/[^\s<>()\"']*)?)",
    re.IGNORECASE,
)


def _source_url_set(text: str) -> set:
    """The set of NORMALIZED URLs that literally appear in ``text``.

    Verifying a model URL by *equality* against this set (rather than a substring
    scan of the raw text) is what stops a fabricated bare ``github.com`` from
    passing just because a real ``github.com/jane`` is present - they normalize to
    different values."""
    out = set()
    for m in _SOURCE_URL_RE.finditer(text):
        raw = m.group(1) or m.group(2) or m.group(3)
        n = _norm_url(raw) if raw else ""
        if n:
            out.add(n)
    return out


def _strip_fabricated_urls(node: Any, source_urls: set) -> Any:
    """Recursively drop URLs the model invented - any ``url`` whose (normalized)
    value is not one of the URLs that literally appear in the source CV text.

    Real URLs are copied from the inline ``anchor <https://...>`` markers, so they
    appear in the source; hallucinated ones (e.g. ``https://linkedin.com`` for a
    plain-text 'LinkedIn') do not. A bare ``Link`` (has a ``label``) with a
    fabricated URL is removed outright; an optional ``url`` on a larger object is
    just cleared (so the field becomes empty rather than a fake link).
    """
    def real(u: Any) -> bool:
        return isinstance(u, str) and bool(u.strip()) and _norm_url(u) in source_urls

    if isinstance(node, list):
        out = []
        for item in node:
            if isinstance(item, dict) and "label" in item and not real(item.get("url")):
                continue  # a Link whose URL is fabricated -> drop the whole link
            out.append(_strip_fabricated_urls(item, source_urls))
        return out
    if isinstance(node, dict):
        return {
            k: _strip_fabricated_urls(v, source_urls)
            for k, v in node.items()
            if not (k == "url" and not real(v))  # clear a fabricated url field
        }
    return node


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
    verify_urls: bool = True,
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
        verify_urls: drop any URL the model produced that is not present in
            ``text`` (guards against fabricated links like ``https://linkedin.com``
            for a plain-text 'LinkedIn'). On by default; pass ``False`` to keep the
            model's URLs verbatim.

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

    source_urls = _source_url_set(text) if verify_urls else set()
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
            # A tool call whose arguments couldn't be parsed (bad JSON) sets a
            # call_id; that's recoverable - feed it back for repair like a
            # validation error. A missing call_id means no tool call at all.
            if call.call_id is not None and attempt < max_repair_attempts:
                messages.append(prov.assistant_message(call))
                messages.append(prov.error_message(
                    call, f"Your {TOOL_NAME} tool call's arguments were not valid JSON. "
                    f"Call {TOOL_NAME} again with well-formed JSON arguments."))
                continue
            detail = f" Model said: {call.text!r}" if call.text else ""
            raise ExtractionError(f"{prov.label} did not return a valid {TOOL_NAME!r} tool call.{detail}")

        arguments = call.arguments
        if verify_urls:
            arguments = _strip_fabricated_urls(arguments, source_urls)
        try:
            return CV.model_validate(arguments)
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


def build_cv(text: str, **kwargs: Any) -> CV:
    """Build a validated :class:`CV` from a free-form self-description (a brain-dump).

    A thin wrapper over :func:`extract_cv` with the build-tuned
    :data:`BUILD_SYSTEM_PROMPT`, so it inherits the repair loop, the
    URL-fabrication guard, and schema validation. Same faithfulness guarantee as
    extraction: it organizes and rephrases what the person states, and never invents
    a skill, employer, date, or metric they did not mention.

    All keyword args of :func:`extract_cv` are accepted (``provider``, ``model``,
    ``api_key``, ...). Do not pass ``system_prompt`` - the build prompt is fixed here.
    """
    return extract_cv(text, system_prompt=BUILD_SYSTEM_PROMPT, **kwargs)


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

    # verify_urls=False: the fake link URL isn't in the fake source, and here we're
    # exercising the repair loop, not the URL guard (which has its own asserts below).
    cv = extract_cv("(fake CV text)", client=_FakeAnthropic([invalid, valid]),
                    max_repair_attempts=2, verify_urls=False)
    print("repair loop recovered ->", cv.name, "|", cv.experience[0].company,
          "| link:", cv.contact.links[0].url)
    assert cv.name == "Jordan Mercer" and cv.experience[0].company == "PayNova"
    print("OK")

    # build_cv: same machinery, build-tuned prompt (fake client returns a valid CV).
    built = build_cv("(fake self-description)", client=_FakeAnthropic([valid]), verify_urls=False)
    assert built.name == "Jordan Mercer" and built.experience[0].company == "PayNova"
    print("build_cv OK ->", built.name)

    # URL guard: fabricated links dropped, real ones (present in source) kept.
    src_urls = _source_url_set("Necmettin GitHub <https://github.com/NecmettinG> LinkedIn | GitHub")
    args = {
        "name": "X",
        "contact": {"links": [
            {"label": "GitHub", "url": "https://github.com/NecmettinG"},   # real (full) -> keep
            {"label": "GitHub", "url": "https://github.com"},              # fabricated BARE -> drop
            {"label": "LinkedIn", "url": "https://linkedin.com"},          # fabricated -> drop
            {"label": "Site", "url": "https://"},                          # degenerate -> drop
        ]},
        "education": [{"institution": "U", "url": "https://fake.example/nope"}],  # fabricated -> clear
        "sections": [{"title": "Refs", "kind": "entries",
                      "entries": [{"title": "Kayhan", "url": "https://linkedin.com/in/x"}]}],
    }
    cleaned = _strip_fabricated_urls(args, src_urls)
    # only the real FULL github url survives; the bare github.com (finding #4) is
    # dropped even though 'github.com/necmetting' is in the source.
    kept = [(l["label"], l["url"]) for l in cleaned["contact"]["links"]]
    assert kept == [("GitHub", "https://github.com/NecmettinG")], kept
    assert "url" not in cleaned["education"][0]
    assert "url" not in cleaned["sections"][0]["entries"][0]
    print("URL guard OK (kept real link; dropped fabricated bare/degenerate + url fields)")

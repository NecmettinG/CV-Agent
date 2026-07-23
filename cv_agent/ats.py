"""ATS (Applicant Tracking System) analysis: score a CV for machine-readability
and keyword fit, and (opt-in) rephrase it toward a job description.

Real ATS software does two things to a resume: it **parses** the file into text
+ fields, then **ranks** it by how well its keywords match the job. This module
mirrors both, plus a guarded rewrite step:

    1. roundtrip_parse(pdf, cv)     re-extract our OWN generated PDF with
                                    pdfplumber and confirm the name, contact,
                                    every employer/school, and every section
                                    heading survived. If they don't, the TEMPLATE
                                    is broken - fix it. (Phase-4 step 1.)
    2. extract_job_keywords(jd)     LLM pulls the decisive keywords from a job
                                    description, tagged required / preferred.
       keyword_coverage(cv, kws)    deterministic: which of those the CV contains.
    3. ats_report(cv, pdf, jd)      combine into a transparent 0-100 score with a
                                    human-readable breakdown + recommendations.
    4. improve_cv(cv, keywords)     OPT-IN, GUARDED: rephrase existing prose to
                                    surface keywords the candidate genuinely has.
                                    The original employers / titles / schools / dates /
                                    tech lists are kept VERBATIM (grafting), so a
                                    rewrite cannot change them - only the summary,
                                    experience bullets/descriptions, and skills lines
                                    are taken from the model.

Scoring is a heuristic - there is no universal ATS formula - so the rubric here
is deliberately simple and transparent (see the WEIGHT constants), and every
sub-score is reported so a user can see exactly why they got the number.

Only the two LLM steps (keyword extraction, rewrite) need a provider/API key;
parsing, coverage, and scoring are fully offline and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cv_agent.providers import Provider, build_provider, dereference_schema
from cv_agent.schema import CV

# --------------------------------------------------------------------------- #
# Scoring weights - all in one place so the rubric is explicit and tunable.
# --------------------------------------------------------------------------- #
#: Within the format score, how much the round-trip (parse) vs the best-practice
#: checks count. They sum to 1.0.
PARSE_WEIGHT = 0.6
BEST_PRACTICE_WEIGHT = 0.4

#: With a job description, how much format vs keyword coverage count in the
#: overall score. Keywords dominate real ATS ranking, so they weigh more.
FORMAT_WEIGHT = 0.4
KEYWORD_WEIGHT = 0.6

#: A 'required' JD keyword counts this much more than a 'preferred' one.
REQUIRED_WEIGHT = 2.0
PREFERRED_WEIGHT = 1.0

#: Sensible CV length band (searchable words); outside it triggers a soft flag.
MIN_WORDS = 150
MAX_WORDS = 1200


class AtsError(RuntimeError):
    """Raised when an LLM ATS step fails (no tool call, bad output, unfaithful rewrite)."""


def _unwrap_envelope(args: Any, model_fields: set) -> Any:
    """Undo the ``{'cv': {...}}`` wrapper some models put around a tool payload.

    A few models (smaller ones especially) nest the whole object under a single
    outer key named after the tool/type instead of returning its fields at the top
    level. If ``args`` is a one-key dict whose key is NOT a real field of the target
    model and whose value is a dict, unwrap it; otherwise leave it untouched (so a
    legit single-field payload like ``{'keywords': [...]}`` is preserved).
    """
    if isinstance(args, dict) and len(args) == 1:
        (key, value), = args.items()
        if isinstance(value, dict) and key not in model_fields:
            return value
    return args


def _format_validation_errors(exc: Optional[ValidationError]) -> str:
    """Compact, model-readable rendering of pydantic validation errors (for repair)."""
    if exc is None:
        return "(unknown validation error)"
    return "\n".join(
        f"- {'.'.join(str(p) for p in e.get('loc', ())) or '(root)'}: {e.get('msg')}"
        for e in exc.errors()
    )


# --------------------------------------------------------------------------- #
# Report value objects (plain dataclasses - internal, not LLM I/O).
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    """One pass/fail signal that feeds a score."""

    name: str
    passed: bool
    detail: str = ""
    weight: float = 1.0


@dataclass
class Keyword:
    """A keyword an ATS would score on, from the job description."""

    text: str
    category: str = "skill"       # hard_skill | tool | soft_skill | qualification | domain
    importance: str = "required"  # required | preferred

    @property
    def weight(self) -> float:
        return REQUIRED_WEIGHT if self.importance.lower().startswith("req") else PREFERRED_WEIGHT


@dataclass
class KeywordHit:
    keyword: Keyword
    present: bool


@dataclass
class ParseReport:
    """Result of re-reading our own PDF: did the important data survive?"""

    text: str
    checks: List[Check]

    @property
    def score(self) -> float:
        return _weighted_pass_rate(self.checks)

    @property
    def missing(self) -> List[str]:
        return [c.name for c in self.checks if not c.passed]


@dataclass
class CoverageReport:
    """Which job-description keywords the CV contains."""

    hits: List[KeywordHit]

    @property
    def matched(self) -> List[Keyword]:
        return [h.keyword for h in self.hits if h.present]

    @property
    def missing(self) -> List[Keyword]:
        return [h.keyword for h in self.hits if not h.present]

    @property
    def score(self) -> float:
        """Weighted % of keyword weight the CV covers (required keywords weigh more)."""
        total = sum(h.keyword.weight for h in self.hits)
        if total == 0:
            return 100.0
        got = sum(h.keyword.weight for h in self.hits if h.present)
        return 100.0 * got / total


@dataclass
class AtsReport:
    """The full ATS analysis: sub-scores, the checks behind them, and advice."""

    format_score: float
    best_practices: List[Check]
    overall: float
    parse: Optional[ParseReport] = None
    coverage: Optional[CoverageReport] = None
    keyword_score: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)

    def summary_text(self) -> str:
        """A plain-text report card, safe for any console."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append(f"  ATS SCORE: {self.overall:5.1f} / 100")
        lines.append("=" * 60)
        lines.append(f"  Format / machine-readability : {self.format_score:5.1f} / 100")
        if self.keyword_score is not None:
            lines.append(f"  Job-description keyword match : {self.keyword_score:5.1f} / 100")
        lines.append("")

        if self.parse is not None:
            lines.append(f"-- Round-trip parse ({self.parse.score:.0f}/100) "
                         "- does an ATS see everything? --")
            for c in self.parse.checks:
                lines.append(f"   [{'PASS' if c.passed else 'FAIL'}] {c.name}"
                             + (f"  - {c.detail}" if c.detail and not c.passed else ""))
            lines.append("")

        lines.append("-- Best-practice checks --")
        for c in self.best_practices:
            lines.append(f"   [{'PASS' if c.passed else 'FAIL'}] {c.name}"
                         + (f"  - {c.detail}" if c.detail and not c.passed else ""))
        lines.append("")

        if self.coverage is not None:
            cov = self.coverage
            matched = cov.matched
            missing = cov.missing
            lines.append(f"-- Keyword coverage: {len(matched)}/{len(cov.hits)} present --")
            if matched:
                lines.append("   Present : " + ", ".join(k.text for k in matched))
            if missing:
                req = [k.text for k in missing if k.weight == REQUIRED_WEIGHT]
                pref = [k.text for k in missing if k.weight != REQUIRED_WEIGHT]
                if req:
                    lines.append("   MISSING (required)  : " + ", ".join(req))
                if pref:
                    lines.append("   Missing (preferred) : " + ", ".join(pref))
            lines.append("")

        if self.recommendations:
            lines.append("-- Recommendations --")
            for r in self.recommendations:
                lines.append(f"   * {r}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 1. Searchable text + keyword matching (deterministic core)
# --------------------------------------------------------------------------- #
#: model_dump keys whose string values are not meaningful search text.
_SKIP_KEYS = frozenset({"url", "email", "phone", "language"})


def cv_searchable_text(cv: CV) -> str:
    """All the human-readable text of a CV as one blob, for keyword scanning.

    Walks the dumped model so it stays correct as the schema grows; skips URLs /
    email / phone / language, which are not skills text. Scanning the CV object
    (not the re-parsed PDF) avoids any parse loss - round-trip already proves the
    PDF matches the object.
    """
    parts: List[str] = []

    def walk(node: Any, key: Optional[str]) -> None:
        if isinstance(node, str):
            if key not in _SKIP_KEYS and node.strip():
                parts.append(node)
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)

    walk(cv.model_dump(), None)
    return "\n".join(parts)


def keyword_present(keyword: str, text_lower: str) -> bool:
    """Is ``keyword`` present in ``text_lower`` (already lowercased)?

    Boundary-aware so 'R' doesn't match inside 'React', yet symbol-tokens like
    'C++', 'C#', '.NET' still match. A secondary normalized pass ('node.js' ~
    'nodejs') and a simple plural fold catch common surface variants without the
    over-matching that would inflate the score dishonestly.
    """
    kw = keyword.strip().lower()
    if not kw:
        return False

    # Exact, boundary-aware. \w boundaries on each side; works for c++, c#, .net
    # because their non-alnum edges sit next to whitespace/punctuation in the CV.
    for variant in _plural_variants(kw):
        if re.search(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", text_lower):
            return True

    # Normalized fold for single-token symbol/spacing variants only (no phrases),
    # e.g. 'node.js' -> 'nodejs', 'ci/cd' -> 'cicd'. Require length >= 3 to avoid
    # collapsing 'c#'/'c++' into a 1-char substring that matches everywhere.
    if " " not in kw:
        norm_kw = re.sub(r"[^a-z0-9]", "", kw)
        if len(norm_kw) >= 3:
            if norm_kw in re.sub(r"[^a-z0-9]", "", text_lower):
                return True
    return False


def _plural_variants(kw: str) -> List[str]:
    """The term plus a simple singular/plural fold of its TRAILING word, so
    'REST API' ~ 'REST APIs' and 'microservices' ~ 'microservice'. Leaves
    symbol-tails ('C++') untouched - the exact match already covers those."""
    out = {kw}
    m = re.search(r"[a-z0-9]+$", kw)  # trailing run of alphanumerics, if any
    if m:
        tail, base = m.group(0), kw[: m.start()]
        if tail.endswith("s") and len(tail) > 3:
            out.add(base + tail[:-1])
        else:
            out.add(base + tail + "s")
    return list(out)


def keyword_coverage(cv: CV, keywords: List[Keyword]) -> CoverageReport:
    """Deterministically check which ``keywords`` appear in ``cv``'s text."""
    text_lower = cv_searchable_text(cv).lower()
    hits = [KeywordHit(k, keyword_present(k.text, text_lower)) for k in keywords]
    return CoverageReport(hits)


# --------------------------------------------------------------------------- #
# Round-trip: re-parse our own PDF and check the important data survived.
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _contains_check(name: str, needle: Optional[str], haystack_norm: str,
                    weight: float = 1.0) -> Check:
    """Pass if ``needle`` (whitespace-normalized) is in ``haystack_norm``; falls
    back to 'all significant word tokens present' for names the PDF split oddly."""
    if not needle or not needle.strip():
        return Check(name, True, "(empty)", weight)
    n = _norm(needle)
    passed = n in haystack_norm
    if not passed:
        toks = [t for t in re.findall(r"\w+", n) if len(t) > 2]
        passed = bool(toks) and all(t in haystack_norm for t in toks)
    return Check(name, passed, "" if passed else f"'{needle}' not found in re-parsed PDF text", weight)


def roundtrip_parse(pdf_path: Union[str, Path], cv: CV, *,
                    x_tolerance: Optional[float] = None) -> ParseReport:
    """Re-extract text from our generated PDF and verify the CV's key data survives.

    This is the template quality gate: if a name, employer, or heading does not
    come back out of the PDF, an ATS won't see it either - the template is at
    fault and should be fixed (not the CV).
    """
    from cv_agent.parsers.pdf import DEFAULT_X_TOLERANCE
    from cv_agent.parsers.pdf import extract_text as _pdf_text

    xt = DEFAULT_X_TOLERANCE if x_tolerance is None else x_tolerance
    # annotate_links=False: the plain text layer, i.e. exactly what an ATS reads.
    text = _pdf_text(pdf_path, annotate_links=False, x_tolerance=xt)
    hay = _norm(text)
    digits = re.sub(r"\D", "", text)

    checks: List[Check] = [_contains_check("Name", cv.name, hay, weight=2.0)]
    if cv.contact.email:
        checks.append(_contains_check("Email", cv.contact.email, hay, weight=2.0))
    if cv.contact.phone:
        pd = re.sub(r"\D", "", cv.contact.phone)
        ok = len(pd) >= 6 and pd[-7:] in digits
        checks.append(Check("Phone", ok, "" if ok else "phone digits not found in re-parsed text", 1.5))
    if cv.contact.location:
        checks.append(_contains_check("Location", cv.contact.location, hay, weight=0.5))

    for e in cv.experience:
        checks.append(_contains_check(f"Experience: {e.company}", e.company, hay, weight=1.5))
        for sr in e.sub_roles:
            checks.append(_contains_check(f"Role: {sr.company}", sr.company, hay))
    for ed in cv.education:
        checks.append(_contains_check(f"Education: {ed.institution}", ed.institution, hay, weight=1.5))
    for s in cv.sections:
        if s.has_content:
            checks.append(_contains_check(f"Section: {s.title}", s.title, hay))

    return ParseReport(text=text, checks=checks)


# --------------------------------------------------------------------------- #
# Best-practice checks (format score, no job description needed).
# --------------------------------------------------------------------------- #
def _has_quantified_achievement(cv: CV) -> bool:
    """True if any bullet / description contains a digit (a metric, %, count, ...)."""
    def bodies() -> Any:
        for e in cv.experience:
            yield e.description or ""
            yield from e.highlights
            for sr in e.sub_roles:
                yield sr.description or ""
                yield from sr.highlights
        for s in cv.sections:
            for en in s.entries:
                yield en.detail or ""
                yield from en.highlights
    return any(re.search(r"\d", b) for b in bodies())


def _has_skills_section(cv: CV) -> bool:
    return any(s.kind == "skills" or "skill" in s.title.lower() for s in cv.sections)


def best_practice_checks(cv: CV) -> List[Check]:
    """CV-quality checks recruiters/ATS reward, computed from the CV object."""
    words = len(re.findall(r"\w+", cv_searchable_text(cv)))
    checks = [
        Check("Has email", bool(cv.contact.email),
              "add an email so recruiters (and the ATS) can reach you", 2.0),
        Check("Has phone", bool(cv.contact.phone), "add a phone number", 1.0),
        Check("Has location", bool(cv.contact.location), "add a city/region", 0.5),
        Check("Has work experience", bool(cv.experience), "no experience entries found", 2.0),
        Check("Has education", bool(cv.education), "no education entries found", 1.0),
        Check("Has a skills section", _has_skills_section(cv),
              "add an explicit Skills section - ATS keyword-match it heavily", 2.0),
        Check("Quantified achievements", _has_quantified_achievement(cv),
              "add numbers/metrics to bullets (e.g. 'cut latency 40%')", 1.0),
        Check("Reasonable length", MIN_WORDS <= words <= MAX_WORDS,
              f"{words} words - aim for {MIN_WORDS}-{MAX_WORDS}", 0.5),
    ]
    return checks


def _weighted_pass_rate(checks: List[Check]) -> float:
    total = sum(c.weight for c in checks)
    if total == 0:
        return 100.0
    return 100.0 * sum(c.weight for c in checks if c.passed) / total


# --------------------------------------------------------------------------- #
# 2. Job-description keyword extraction (LLM, forced tool use).
# --------------------------------------------------------------------------- #
class _KwModel(BaseModel):
    model_config = ConfigDict(extra="ignore")  # tolerate extra keys from lenient models
    text: str = Field(..., description="The keyword exactly as a CV would spell it.")
    category: str = Field(default="hard_skill")
    importance: str = Field(default="required", description="'required' or 'preferred'.")


class _KwList(BaseModel):
    model_config = ConfigDict(extra="ignore")
    keywords: List[_KwModel] = Field(default_factory=list)


_KW_TOOL = "record_keywords"
_KW_SYSTEM = (
    "You are an expert technical recruiter and ATS analyst. Read a job description and "
    "extract the concrete keywords an Applicant Tracking System scores candidates on: hard "
    "skills, technologies, tools, frameworks, methodologies, certifications, and explicit "
    "qualifications. For each keyword set `importance` to 'required' (a must-have the posting "
    "explicitly demands) or 'preferred' (nice-to-have / bonus), and `category` to one of "
    "'hard_skill', 'tool', 'soft_skill', 'qualification', 'domain'. Extract the 10-30 MOST "
    "decisive, checkable terms - prefer specific nouns a CV would literally contain "
    "('Kubernetes', 'REST APIs', 'CI/CD', \"Bachelor's in Computer Science\", 'B2 English') "
    "over vague phrases ('team player', 'fast learner'). Use the exact surface form. Call "
    f"`{_KW_TOOL}` exactly once."
)


def extract_job_keywords(
    job_description: str,
    *,
    provider: Union[str, Provider] = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Any = None,
    max_tokens: int = 2000,
    tool_choice: Any = None,
) -> List[Keyword]:
    """Extract the ATS keywords from a job description via forced tool use.

    Args mirror :func:`cv_agent.extract.extract_cv` (same provider system). Returns
    a list of :class:`Keyword`. Raises :class:`AtsError` if the model won't call the
    tool or its output can't be validated, :class:`ValueError` on empty input.
    """
    if not job_description or not job_description.strip():
        raise ValueError("`job_description` is empty; nothing to extract.")

    prov = build_provider(provider, api_key=api_key, model=model, base_url=base_url, client=client)
    parameters = dereference_schema(_KwList.model_json_schema())
    user = ("Extract the ATS keywords from this job posting by calling the tool.\n\n"
            "---- JOB DESCRIPTION ----\n" + job_description + "\n---- END ----")
    call = prov.call(
        [prov.user_message(user)],
        system=_KW_SYSTEM,
        tool_name=_KW_TOOL,
        tool_description="Record the job description's ATS keywords. Call once.",
        parameters=parameters,
        max_tokens=max_tokens,
        tool_choice=tool_choice,
    )
    if call.arguments is None:
        detail = f" Model said: {call.text!r}" if call.text else ""
        raise AtsError(f"{prov.label} did not return a {_KW_TOOL!r} tool call.{detail}")
    try:
        parsed = _KwList.model_validate(_unwrap_envelope(call.arguments, set(_KwList.model_fields)))
    except ValidationError as exc:
        raise AtsError(f"Keyword extraction returned invalid data:\n{exc}") from exc

    seen: set = set()
    out: List[Keyword] = []
    for k in parsed.keywords:
        text = k.text.strip()
        low = text.lower()
        if text and low not in seen:
            seen.add(low)
            out.append(Keyword(text=text, category=k.category.strip() or "skill",
                               importance=k.importance.strip() or "required"))
    return out


# --------------------------------------------------------------------------- #
# 3. The combined report.
# --------------------------------------------------------------------------- #
def ats_report(
    cv: CV,
    *,
    pdf_path: Optional[Union[str, Path]] = None,
    keywords: Optional[List[Keyword]] = None,
    job_description: Optional[str] = None,
    x_tolerance: Optional[float] = None,
    **extract_kwargs: Any,
) -> AtsReport:
    """Score ``cv`` for ATS-friendliness and (optionally) fit to a job.

    Args:
        cv: the CV to score.
        pdf_path: the CV's generated PDF. If given, its round-trip parse feeds the
            format score (and proves the template is machine-readable). If omitted,
            the format score uses the best-practice checks only.
        keywords: pre-extracted job keywords (skips the LLM - handy for testing).
        job_description: raw JD text; if given and ``keywords`` is not, the keywords
            are extracted from it via :func:`extract_job_keywords` (needs a provider
            + API key, passed through ``extract_kwargs``).
        x_tolerance: PDF word-gap tolerance for the round-trip parse.
        **extract_kwargs: forwarded to :func:`extract_job_keywords`
            (``provider``, ``model``, ``api_key``, ...).

    Returns:
        An :class:`AtsReport` with sub-scores, checks, and recommendations.
    """
    parse = roundtrip_parse(pdf_path, cv, x_tolerance=x_tolerance) if pdf_path else None
    bp = best_practice_checks(cv)

    bp_score = _weighted_pass_rate(bp)
    if parse is not None:
        format_score = PARSE_WEIGHT * parse.score + BEST_PRACTICE_WEIGHT * bp_score
    else:
        format_score = bp_score

    if keywords is None and job_description:
        keywords = extract_job_keywords(job_description, **extract_kwargs)

    coverage = None
    keyword_score = None
    if keywords:
        coverage = keyword_coverage(cv, keywords)
        keyword_score = coverage.score
        overall = FORMAT_WEIGHT * format_score + KEYWORD_WEIGHT * keyword_score
    else:
        overall = format_score

    report = AtsReport(
        format_score=round(format_score, 1),
        best_practices=bp,
        overall=round(overall, 1),
        parse=parse,
        coverage=coverage,
        keyword_score=None if keyword_score is None else round(keyword_score, 1),
    )
    report.recommendations = _recommendations(report)
    return report


def _recommendations(report: AtsReport) -> List[str]:
    """Actionable advice distilled from the failed checks + missing keywords."""
    recs: List[str] = []
    if report.parse is not None:
        for c in report.parse.checks:
            if not c.passed:
                recs.append(f"TEMPLATE BUG: '{c.name}' did not survive PDF parsing "
                            f"- an ATS can't read it. {c.detail}")
    for c in report.best_practices:
        if not c.passed and c.detail:
            recs.append(c.detail[0].upper() + c.detail[1:])
    if report.coverage is not None:
        req_missing = [k.text for k in report.coverage.missing if k.weight == REQUIRED_WEIGHT]
        if req_missing:
            recs.append("Cover these REQUIRED job keywords if you genuinely have them: "
                        + ", ".join(req_missing) + ".")
    return recs


# --------------------------------------------------------------------------- #
# 4. Opt-in, guarded rewrite toward the keywords.
# --------------------------------------------------------------------------- #
_REWRITE_TOOL = "record_cv"
_REWRITE_SYSTEM = (
    "You optimize a CV for ATS keyword coverage WITHOUT fabricating anything. You receive the "
    "candidate's existing CV as JSON plus target keywords. Rephrase ONLY the wording of "
    "existing content so that keywords the candidate GENUINELY demonstrates are stated with the "
    "standard term.\n"
    "ABSOLUTE RULES:\n"
    "- NEVER add a skill, technology, tool, employer, role, achievement, date, or qualification "
    "the candidate does not already demonstrate in the original CV. If there is no evidence for a "
    "target keyword, DO NOT add it - a keyword the person lacks must stay missing. Fabrication is "
    "a hard failure.\n"
    "- Do NOT change, add, or remove any company name, job title, institution, or date, and keep "
    "the same experience / education / section structure and counts.\n"
    "- You MAY: reword highlight bullets, descriptions and the summary; use the standard spelling "
    "of a technology already implied (e.g. write 'CI/CD' where a bullet clearly describes it); and "
    "move a technology already named in a bullet into that role's tech_stack.\n"
    "- Keep it truthful, concise, and natural - no keyword stuffing.\n"
    f"Return the full improved CV by calling `{_REWRITE_TOOL}` exactly once."
)


def _skeleton(cv: CV) -> Dict[str, Any]:
    """The factual bones a rewrite must NOT touch: employers, titles, schools, dates."""
    def dr(d: Any) -> Any:
        return d.model_dump() if d is not None else None
    return {
        "experience": [
            {"company": e.company, "title": e.title, "dates": dr(e.date_range),
             "sub": [(s.company, s.title) for s in e.sub_roles]}
            for e in cv.experience
        ],
        "education": [
            {"institution": ed.institution, "degree": ed.degree, "dates": dr(ed.date_range)}
            for ed in cv.education
        ],
        "name": cv.name,
    }


def _faithfulness_violations(original: CV, improved: CV) -> List[str]:
    """Non-empty if the rewrite changed the factual skeleton (a fabrication tripwire)."""
    a, b = _skeleton(original), _skeleton(improved)
    problems: List[str] = []
    if a["name"] != b["name"]:
        problems.append(f"name changed: {a['name']!r} -> {b['name']!r}")
    if len(a["experience"]) != len(b["experience"]):
        problems.append(f"experience count changed: {len(a['experience'])} -> {len(b['experience'])}")
    else:
        for i, (ea, eb) in enumerate(zip(a["experience"], b["experience"])):
            if ea != eb:
                problems.append(f"experience #{i + 1} employer/title/dates changed")
    if len(a["education"]) != len(b["education"]):
        problems.append(f"education count changed: {len(a['education'])} -> {len(b['education'])}")
    else:
        for i, (ea, eb) in enumerate(zip(a["education"], b["education"])):
            if ea != eb:
                problems.append(f"education #{i + 1} institution/degree/dates changed")
    return problems


def _graft_rewrite(original: CV, rewrite: CV) -> CV:
    """Build the improved CV from the ORIGINAL's factual skeleton + the model's
    reworded prose. Only the summary, each experience's description/highlights (and
    its sub-roles'), and same-length skills-section bullets are taken from
    ``rewrite``, matched by position. Everything factual - name, contact, employers,
    titles, dates, tech lists, education, and all other sections - is kept verbatim
    from ``original``. Any list whose length the model changed is left untouched.

    Result: the rewrite is structurally unable to alter employers/titles/schools/
    dates or to add a new skill line, no matter what the model returned.
    """
    result = original.model_copy(deep=True)

    if rewrite.summary and rewrite.summary.strip():
        result.summary = rewrite.summary

    if len(result.experience) == len(rewrite.experience):
        for o, n in zip(result.experience, rewrite.experience):
            if n.description and n.description.strip():
                o.description = n.description
            if n.highlights:
                o.highlights = n.highlights
            if len(o.sub_roles) == len(n.sub_roles):
                for os_, ns_ in zip(o.sub_roles, n.sub_roles):
                    if ns_.description and ns_.description.strip():
                        os_.description = ns_.description
                    if ns_.highlights:
                        os_.highlights = ns_.highlights

    # Skills lines may be reworded (e.g. normalize spelling) but not added/removed,
    # so the count must match - this stops the model padding skills with keywords.
    if len(result.sections) == len(rewrite.sections):
        for o, n in zip(result.sections, rewrite.sections):
            if o.kind == "skills" and n.bullets and len(n.bullets) == len(o.bullets):
                o.bullets = n.bullets

    return result


def improve_cv(
    cv: CV,
    keywords: List[Keyword],
    *,
    provider: Union[str, Provider] = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Any = None,
    max_tokens: int = 16000,
    max_repair_attempts: int = 2,
    tool_choice: Any = None,
) -> CV:
    """Rephrase ``cv`` to surface ``keywords`` the candidate genuinely has - GUARDED.

    This is opt-in and deliberately conservative. The model is asked to reword the
    CV, but its output is not trusted directly: :func:`_graft_rewrite` keeps the
    original's factual skeleton (name, contact, employers, titles, dates, tech
    lists, education, other sections) VERBATIM and grafts back only the reworded
    summary, experience bullets/descriptions, and skills lines. So a rewrite cannot
    change an employer, title, school, or date, and cannot add a skill line - it can
    only reword existing prose. (Prose honesty itself still rests on the system
    prompt; keywords the CV shows no evidence for should be left uncovered.)

    A schema-repair loop (like :func:`cv_agent.extract.extract_cv`) feeds any
    validation error back to the model, and a single-key envelope such as
    ``{'cv': {...}}`` is auto-unwrapped first.

    Returns the improved (validated) :class:`CV`. Raises :class:`AtsError` if the
    model won't call the tool or returns invalid data after all repair attempts.
    """
    if not keywords:
        return cv
    prov = build_provider(provider, api_key=api_key, model=model, base_url=base_url, client=client)
    parameters = dereference_schema(CV.model_json_schema())
    fields = set(CV.model_fields)
    kw_lines = "\n".join(f"- {k.text} ({k.importance})" for k in keywords)
    user = (
        "Rephrase this CV to surface any target keywords the candidate genuinely demonstrates, "
        "following the absolute rules. Call the tool once with the full improved CV, returning "
        "its fields (name, contact, experience, ...) at the TOP LEVEL - do NOT wrap them in an "
        "outer object.\n\n"
        "---- TARGET KEYWORDS ----\n" + kw_lines + "\n\n"
        "---- CURRENT CV (JSON) ----\n" + cv.model_dump_json() + "\n---- END ----"
    )
    messages: List[Dict[str, Any]] = [prov.user_message(user)]

    last_error: Optional[ValidationError] = None
    for attempt in range(max_repair_attempts + 1):
        call = prov.call(
            messages,
            system=_REWRITE_SYSTEM,
            tool_name=_REWRITE_TOOL,
            tool_description="Record the improved CV. Call once, fields at the top level.",
            parameters=parameters,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
        )
        if call.arguments is None:
            detail = f" Model said: {call.text!r}" if call.text else ""
            raise AtsError(f"{prov.label} did not return a {_REWRITE_TOOL!r} tool call.{detail}")
        try:
            rewrite = CV.model_validate(_unwrap_envelope(call.arguments, fields))
        except ValidationError as exc:
            last_error = exc
            if attempt >= max_repair_attempts:
                break
            messages.append(prov.assistant_message(call))
            messages.append(prov.error_message(
                call,
                "The improved CV failed validation:\n" + _format_validation_errors(exc)
                + "\n\nCall record_cv again with corrected arguments, keeping everything that was "
                "already valid, and put the CV fields at the TOP LEVEL (not wrapped in an outer key)."
            ))
            continue

        # Trust only the reworded prose; keep the factual skeleton from `cv`.
        improved = _graft_rewrite(cv, rewrite)
        # Grafting guarantees the skeleton is unchanged; this only trips on a bug
        # in _graft_rewrite, never on the model's output.
        violations = _faithfulness_violations(cv, improved)
        if violations:
            raise AtsError("internal: grafting did not preserve the CV skeleton:\n- "
                           + "\n- ".join(violations))
        return improved

    raise AtsError(
        "The rewritten CV failed schema validation after "
        f"{max_repair_attempts} repair attempt(s):\n" + _format_validation_errors(last_error)
    )


if __name__ == "__main__":  # pragma: no cover
    # Offline smoke tests: the deterministic core + the LLM steps via fake clients.
    import sys
    import types

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from examples.sample_data import sample_cv

    # --- keyword matcher edge cases -------------------------------------------
    t = "experienced with c++, c#, node.js, ci/cd pipelines and rest apis on aws.".lower()
    assert keyword_present("C++", t)
    assert keyword_present("C#", t)
    assert keyword_present("Node.js", t)      # normalized fold: node.js ~ nodejs? here literal
    assert keyword_present("CI/CD", t)
    assert keyword_present("REST API", t)     # plural fold: apis -> api
    assert keyword_present("AWS", t)
    assert not keyword_present("Java", t)     # must not false-positive
    assert not keyword_present("R", t)        # boundary: no bare 'r' inside words
    print("keyword matcher OK")

    # --- coverage on the sample CV --------------------------------------------
    kws = [
        Keyword("Java", "hard_skill", "required"),
        Keyword("Spring Boot", "hard_skill", "required"),
        Keyword("AWS", "tool", "required"),
        Keyword("Kubernetes", "tool", "required"),    # present in sample tech_stack
        Keyword("Rust", "hard_skill", "preferred"),   # NOT in sample -> missing
        Keyword("GraphQL", "tool", "preferred"),      # NOT in sample -> missing
    ]
    cov = keyword_coverage(sample_cv, kws)
    present = {k.text for k in cov.matched}
    missing = {k.text for k in cov.missing}
    assert {"Java", "Spring Boot", "AWS", "Kubernetes"} <= present, present
    assert {"Rust", "GraphQL"} <= missing, missing
    print(f"coverage OK -> {cov.score:.0f}% ({len(cov.matched)}/{len(cov.hits)} present)")

    # --- best-practice checks + format-only report ----------------------------
    report = ats_report(sample_cv, keywords=kws)
    assert 0 <= report.overall <= 100 and report.keyword_score is not None
    print(f"report OK -> overall {report.overall}, format {report.format_score}, "
          f"keywords {report.keyword_score}")

    # --- LLM keyword extraction via a fake Anthropic client -------------------
    def _fake_kw_client(payload):
        block = types.SimpleNamespace(type="tool_use", name=_KW_TOOL, input=payload, id="k1")
        resp = types.SimpleNamespace(content=[block], stop_reason="tool_use")
        client = types.SimpleNamespace()
        client.messages = types.SimpleNamespace(create=lambda **_: resp)
        return client

    fake = _fake_kw_client({"keywords": [
        {"text": "Kubernetes", "category": "tool", "importance": "required"},
        {"text": "Java", "category": "hard_skill", "importance": "required"},
        {"text": "java", "importance": "preferred"},  # dup (case) -> deduped
    ]})
    got = extract_job_keywords("(fake JD)", client=fake)
    assert [k.text for k in got] == ["Kubernetes", "Java"], got
    print("keyword extraction OK ->", [k.text for k in got])

    # --- rewrite faithfulness tripwire ----------------------------------------
    def _fake_cv_client(cv_payload):
        block = types.SimpleNamespace(type="tool_use", name=_REWRITE_TOOL, input=cv_payload, id="c1")
        resp = types.SimpleNamespace(content=[block], stop_reason="tool_use")
        client = types.SimpleNamespace()
        client.messages = types.SimpleNamespace(create=lambda **_: resp)
        return client

    # a faithful rewrite (skeleton untouched) passes...
    faithful = sample_cv.model_dump()
    faithful["summary"] = "Backend engineer with Java, Spring Boot and AWS experience."
    out = improve_cv(sample_cv, kws, client=_fake_cv_client(faithful))
    assert out.summary.startswith("Backend engineer")
    print("faithful rewrite OK")

    # ...a wrapped {'cv': {...}} envelope (as small models emit) is auto-unwrapped.
    out2 = improve_cv(sample_cv, kws, client=_fake_cv_client({"cv": faithful}))
    assert out2.summary.startswith("Backend engineer")
    print("envelope-wrapped rewrite unwrapped OK")

    # ...and a rewrite that tampers with the skeleton is NOT trusted: grafting
    # keeps the ORIGINAL employer/title/dates/tech and school, taking only prose.
    tampered = sample_cv.model_dump()
    tampered["experience"][0]["company"] = "FabricatedCorp"              # ignored
    tampered["experience"][0]["tech_stack"] = ["Terraform"]             # ignored (not grafted)
    tampered["experience"][0]["highlights"] = ["Built CI/CD pipelines with Java."]  # grafted
    tampered["education"][0]["institution"] = "Fake University"          # ignored
    out3 = improve_cv(sample_cv, kws, client=_fake_cv_client(tampered))
    assert out3.experience[0].company == sample_cv.experience[0].company
    assert out3.experience[0].tech_stack == sample_cv.experience[0].tech_stack
    assert out3.education[0].institution == sample_cv.education[0].institution
    assert out3.experience[0].highlights == ["Built CI/CD pipelines with Java."]
    print("skeleton tampering ignored; prose grafted OK")

    print("OK")

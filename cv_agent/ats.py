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
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cv_agent.providers import Provider, build_provider, dereference_schema
from cv_agent.schema import CV, Section

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

#: Word-gap tolerance for re-parsing OUR OWN generated PDF. Tectonic/XeTeX spaces
#: words tightly, so the parser's general 3.0 default merges them ("JavaSpring");
#: 2.0 recovers clean words. (Real ATS engines like pdfminer read these fine either
#: way - this just makes the round-trip check verify clean, honest text.)
ROUNDTRIP_X_TOLERANCE = 2.0


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
        imp = self.importance.lower()
        is_required = imp.startswith("req") or any(
            w in imp for w in ("must", "mandator", "critical", "essential")
        )
        return REQUIRED_WEIGHT if is_required else PREFERRED_WEIGHT


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

    # Pass 1 - exact, boundary-aware. \w boundaries on each side; works for c++,
    # c#, .net, node.js and phrases as written, and the plural fold catches
    # microservice(s) / API(s).
    for variant in _plural_variants(kw):
        if re.search(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", text_lower):
            return True

    # Pass 2 - separator-flexible but BOUNDARY-AWARE. A multi-token keyword's
    # alphanumeric tokens may be joined by any separator (or none), so 'node.js' ~
    # 'nodejs', 'rest api' ~ 'rest-api' ~ 'restapi', 'ci/cd' ~ 'cicd'. The alnum
    # lookarounds stop substring over-matches - crucially 'java' does NOT match
    # inside 'javascript', 'rust' not in 'trust', '.net' not in 'kubernetes'. A
    # single alnum token is already fully covered by Pass 1, so we skip it here.
    tokens = re.findall(r"[a-z0-9]+", kw)
    if len(tokens) >= 2 and sum(len(t) for t in tokens) >= 3:
        pat = r"[^a-z0-9]*".join(re.escape(t) for t in tokens)
        if re.search(r"(?<![a-z0-9])" + pat + r"s?(?![a-z0-9])", text_lower):
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


def newly_surfaced_keywords(original: CV, improved: CV,
                            keywords: List[Keyword]) -> List[Keyword]:
    """Keywords present in ``improved`` but ABSENT from ``original`` - i.e. terms an
    :func:`improve_cv` rewrite introduced.

    These are the claims a human must verify: grafting keeps the CV's facts intact,
    but the reworded prose can reach for a target keyword the source never supported
    (e.g. adding 'Gradle' or 'microservices'). Anything listed here should be kept
    ONLY if the candidate genuinely has it, and removed otherwise. It compares only
    against the supplied ``keywords`` (the optimization targets, where the risk is).
    """
    before = cv_searchable_text(original).lower()
    after = cv_searchable_text(improved).lower()
    return [k for k in keywords
            if keyword_present(k.text, after) and not keyword_present(k.text, before)]


def apply_keyword_decisions(original: CV, improved: CV, rejected: List[str]) -> CV:
    """Remove ``rejected`` keywords from an :func:`improve_cv` result cleanly.

    For each grafted prose field (summary, experience descriptions/highlights and
    their sub-roles', skills-section bullets) that contains a rejected keyword,
    revert THAT field to ``original``'s wording. Because the rejected terms were
    'newly surfaced' (absent from ``original``), they can only live in grafted
    fields, so reverting those fields removes every occurrence - deterministically,
    grammatically (original text is well-formed), and with no extra model call.

    ``improved`` must be an :func:`improve_cv` result of ``original`` (same
    structure). Returns a new CV; ``improved`` is not mutated.
    """
    rej = [r.strip().lower() for r in rejected if r and r.strip()]
    if not rej:
        return improved.model_copy(deep=True)   # honor 'returns a new CV' (no alias)
    result = improved.model_copy(deep=True)

    def tainted(text: Optional[str]) -> bool:
        low = (text or "").lower()
        return any(keyword_present(r, low) for r in rej)

    if tainted(result.summary):
        result.summary = original.summary

    if len(result.experience) == len(original.experience):
        for e, oe in zip(result.experience, original.experience):
            if tainted(e.description):
                e.description = oe.description
            if any(tainted(h) for h in e.highlights):
                e.highlights = list(oe.highlights)
            if len(e.sub_roles) == len(oe.sub_roles):
                for sr, osr in zip(e.sub_roles, oe.sub_roles):
                    if tainted(sr.description):
                        sr.description = osr.description
                    if any(tainted(h) for h in sr.highlights):
                        sr.highlights = list(osr.highlights)

    if len(result.sections) == len(original.sections):
        for s, os_ in zip(result.sections, original.sections):
            if s.kind == "skills" and any(tainted(b) for b in s.bullets):
                s.bullets = list(os_.bullets)

    return result


# --------------------------------------------------------------------------- #
# Gap-closer: add skills the CANDIDATE affirms they genuinely have.
#
# The rewrite (improve_cv) can only surface keywords already implied in the text.
# A skill the candidate genuinely has but never wrote down stays invisible - and
# the guard rightly won't let the model assert it. The honest fix: let the person
# DECLARE such skills (they are the source of truth about their own experience) and
# record them - to the Skills section, and optionally onto a specific role/project's
# tech line. All deterministic: the tool never invents, it only writes down what the
# user affirmed.
# --------------------------------------------------------------------------- #
def add_declared_skills(cv: CV, skills: List[str], *, section_title: str = "Skills") -> CV:
    """Add candidate-affirmed ``skills`` to the CV's skills section.

    Appends each skill (deduped, case-insensitively) to the first ``kind='skills'``
    section, creating one titled ``section_title`` if the CV has none. Truthful by
    construction - the candidate declares these; the tool only records them. Returns
    a new CV; ``cv`` is not mutated.
    """
    additions = [s.strip() for s in skills if s and s.strip()]
    if not additions:
        return cv.model_copy(deep=True)   # honor 'returns a new CV' (no alias)
    result = cv.model_copy(deep=True)
    target = next((s for s in result.sections if s.kind == "skills"), None)
    if target is None:
        target = Section(title=section_title, kind="skills", bullets=[])
        result.sections.append(target)
    have = {b.strip().lower() for b in target.bullets}
    for s in additions:
        if s.lower() not in have:
            target.bullets.append(s)
            have.add(s.lower())
    return result


#: Section-title markers (any language) that mark a references section, whose
#: entries are people - never a place to attach a skill. 'referans' covers the
#: Turkish 'Referanslar'; 'reference' covers 'References'.
_REFERENCE_TITLE_MARKERS = ("reference", "referans")


def _is_reference_section(title: str) -> bool:
    low = (title or "").lower()
    return any(m in low for m in _REFERENCE_TITLE_MARKERS)


def weavable_entries(cv: CV) -> List[Tuple[str, tuple]]:
    """``(label, locator)`` for every entry a declared skill can be attached to:
    each experience role / umbrella sub-role, and each project-style section entry.

    References are skipped - a skill does not belong on a reference - both by
    SECTION TITLE ('References'/'Referanslar') and, as a fallback, any single entry
    that carries an email/phone. The opaque ``locator`` tuples are consumed by
    :func:`weave_skills`.
    """
    out: List[Tuple[str, tuple]] = []
    for i, e in enumerate(cv.experience):
        if e.sub_roles:
            for j, sr in enumerate(e.sub_roles):
                out.append((f"{e.company} / {sr.company} - {sr.title}", ("exp", i, "sub", j)))
        else:
            out.append(((f"{e.company} - {e.title}" if e.title else e.company), ("exp", i)))
    for si, s in enumerate(cv.sections):
        if s.kind == "entries" and not _is_reference_section(s.title):
            for ei, en in enumerate(s.entries):
                if en.email or en.phone:  # a stray reference-like entry
                    continue
                out.append((f"[{s.title}] {en.title}", ("sec", si, ei)))
    return out


def _resolve_locator(cv: CV, locator: tuple):
    """Return the entry object a ``weavable_entries`` locator points to."""
    if locator[0] == "exp":
        e = cv.experience[locator[1]]
        if len(locator) >= 4 and locator[2] == "sub":
            return e.sub_roles[locator[3]]
        return e
    if locator[0] == "sec":
        return cv.sections[locator[1]].entries[locator[2]]
    raise ValueError(f"bad locator {locator!r}")


def weave_skills(cv: CV, assignments: List[Tuple[tuple, List[str]]]) -> CV:
    """Attach declared skills to specific entries' tech lines (deterministic).

    ``assignments`` pairs a locator (from :func:`weavable_entries`) with the skills
    to add to that entry's ``tech_stack`` (deduped, case-insensitively). Experience
    entries, sub-roles, and project-style section entries all carry a ``tech_stack``.
    Returns a new CV; ``cv`` is not mutated.
    """
    result = cv.model_copy(deep=True)
    for locator, skills in assignments:
        target = _resolve_locator(result, locator)
        have = {t.strip().lower() for t in target.tech_stack}
        for s in skills:
            s = s.strip()
            if s and s.lower() not in have:
                target.tech_stack.append(s)
                have.add(s.lower())
    return result


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
    from cv_agent.parsers.pdf import extract_text as _pdf_text

    xt = ROUNDTRIP_X_TOLERANCE if x_tolerance is None else x_tolerance
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
        yield cv.summary or ""
        for e in cv.experience:
            yield e.description or ""
            yield from e.highlights
            for sr in e.sub_roles:
                yield sr.description or ""
                yield from sr.highlights
        for ed in cv.education:
            yield from ed.highlights
        for s in cv.sections:
            yield s.text or ""            # a 'text'/profile section can carry a metric
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


def _merge_honest_tech(target: Any, source: Any, orig_text_lower: str) -> None:
    """Append ``source``'s tech_stack items onto ``target``'s ONLY when the tech
    already appears somewhere in the original CV text - honest surfacing into the
    tech line (per the rewrite prompt), never fabrication. Mutates ``target``."""
    have = {t.strip().lower() for t in target.tech_stack}
    for t in source.tech_stack:
        t = t.strip()
        if t and t.lower() not in have and keyword_present(t, orig_text_lower):
            target.tech_stack.append(t)
            have.add(t.lower())


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
    orig_text = cv_searchable_text(original).lower()

    if rewrite.summary and rewrite.summary.strip():
        result.summary = rewrite.summary

    if len(result.experience) == len(rewrite.experience):
        for o, n in zip(result.experience, rewrite.experience):
            if n.description and n.description.strip():
                o.description = n.description
            # Same-length only: bullets may be reworded, never added/removed - this
            # stops the model padding a role with fabricated highlight bullets.
            if n.highlights and len(n.highlights) == len(o.highlights):
                o.highlights = n.highlights
            _merge_honest_tech(o, n, orig_text)   # surface genuinely-present tech
            if len(o.sub_roles) == len(n.sub_roles):
                for os_, ns_ in zip(o.sub_roles, n.sub_roles):
                    if ns_.description and ns_.description.strip():
                        os_.description = ns_.description
                    if ns_.highlights and len(ns_.highlights) == len(os_.highlights):
                        os_.highlights = ns_.highlights
                    _merge_honest_tech(os_, ns_, orig_text)

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
    # no substring over-match (Pass-2 boundary guard)
    assert not keyword_present("Java", "i use javascript daily")
    assert not keyword_present("Rust", "building trust with clients")
    assert not keyword_present(".NET", "we run kubernetes")
    assert not keyword_present("React", "a reactive system")
    # separator-flexible phrase matching (no false negatives)
    assert keyword_present("REST API", "we build rest-api services")   # hyphen
    assert keyword_present("REST API", "our restapi layer")            # concatenated
    assert keyword_present("Node.js", "built with nodejs")            # fold still works
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
    tampered["experience"][0]["company"] = "FabricatedCorp"              # ignored (skeleton)
    tampered["experience"][0]["tech_stack"] = ["Terraform"]             # ignored (not grafted)
    tampered["experience"][0]["highlights"] = ["Fabricated extra bullet."]  # PADDING (0->1): blocked
    # Kartos (exp[2]) has 2 highlights; a SAME-COUNT reword IS grafted.
    tampered["experience"][2]["highlights"] = ["Reworded one.", "Reworded two."]
    tampered["education"][0]["institution"] = "Fake University"          # ignored
    out3 = improve_cv(sample_cv, kws, client=_fake_cv_client(tampered))
    assert out3.experience[0].company == sample_cv.experience[0].company
    assert out3.experience[0].tech_stack == sample_cv.experience[0].tech_stack
    assert out3.education[0].institution == sample_cv.education[0].institution
    assert out3.experience[0].highlights == sample_cv.experience[0].highlights   # padding blocked
    assert out3.experience[2].highlights == ["Reworded one.", "Reworded two."]   # reword grafted
    print("skeleton tampering ignored; padding blocked; same-count reword grafted OK")

    # --- newly-surfaced keyword flag (terms to verify) ------------------------
    imp = sample_cv.model_copy(deep=True)
    imp.experience[0].highlights = ["Built GraphQL APIs."]   # GraphQL absent from original
    flagged = newly_surfaced_keywords(sample_cv, imp, kws)
    assert [k.text for k in flagged] == ["GraphQL"], flagged   # Java etc. already present
    print("surfaced-keyword flag OK ->", [k.text for k in flagged])

    # --- apply_keyword_decisions: rejecting a term reverts its field cleanly ----
    imp2 = sample_cv.model_copy(deep=True)
    imp2.summary = "Backend engineer skilled in Java, AWS and GraphQL."  # GraphQL absent
    kept = apply_keyword_decisions(sample_cv, imp2, rejected=["GraphQL"])
    assert kept.summary == sample_cv.summary                # tainted field reverted
    assert not newly_surfaced_keywords(sample_cv, kept, kws)  # GraphQL fully gone
    # a field WITHOUT a rejected term is left as improved
    imp3 = sample_cv.model_copy(deep=True)
    imp3.summary = "Backend engineer skilled in Java and Kubernetes."
    kept3 = apply_keyword_decisions(sample_cv, imp3, rejected=["GraphQL"])
    assert kept3.summary == imp3.summary                    # untouched (no rejected term)
    print("apply_keyword_decisions OK (rejected reverted, others kept)")

    # --- declare + attach genuine skills (deterministic gap-closer) -----------
    with_skill = add_declared_skills(sample_cv, ["GraphQL", "graphql"])  # dup folded
    sk = next(s for s in with_skill.sections if s.kind == "skills")
    assert sk.bullets.count("GraphQL") == 1, sk.bullets
    print("add_declared_skills OK")

    targets = weavable_entries(sample_cv)
    labels = [lab for lab, _ in targets]
    assert any("SmartShop" in lab for lab in labels), labels          # project present
    assert not any("Alex Thompson" in lab for lab in labels), labels  # reference skipped

    # a references section is skipped by TITLE even when its entries carry no
    # email/phone (the case that used to leak into the attach menu).
    from cv_agent.schema import SectionEntry as _SE
    cv_ref = sample_cv.model_copy(deep=True)
    cv_ref.sections.append(Section(title="Referanslar", kind="entries",
                                   entries=[_SE(title="Ada Lovelace")]))  # no email/phone
    assert not any("Ada Lovelace" in lab for lab, _ in weavable_entries(cv_ref))
    print("reference section skipped by title OK")
    proj_loc = next(loc for lab, loc in targets if "SmartShop" in lab)
    exp_loc = next(loc for lab, loc in targets if lab.startswith("PayNova"))
    woven = weave_skills(sample_cv, [(proj_loc, ["GraphQL"]), (exp_loc, ["GraphQL"])])
    smartshop = next(en for s in woven.sections for en in s.entries if en.title == "SmartShop")
    assert "GraphQL" in smartshop.tech_stack                          # onto project tech line
    assert "GraphQL" in woven.experience[0].tech_stack                # onto role tech line
    print("weavable_entries + weave_skills OK")

    print("OK")

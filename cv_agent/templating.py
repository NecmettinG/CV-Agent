r"""Render a validated :class:`~cv_agent.schema.CV` into a LaTeX document.

LaTeX already uses ``{``, ``}`` and ``%``, which collide with Jinja2's default
``{{ }}`` / ``{% %}`` / ``{# #}``. So we configure Jinja2 with LaTeX-safe
delimiters instead:

    variable   \VAR{ ... }
    block       \BLOCK{ ... }
    comment     \#{ ... }

That way a ``.tex.j2`` file is still (almost) valid LaTeX you can read, and the
template engine only reacts to the ``\VAR`` / ``\BLOCK`` markers.

This module also owns:

* ``escape_latex`` - CV text may contain ``&``, ``%``, ``#`` etc.; those must be
  escaped or the PDF won't compile. Registered as the ``e`` filter.
* ``uc`` - Turkish-aware uppercasing for section headings (Python's ``str.upper``
  turns 'i' into 'I' rather than 'İ').
* date formatting and the small localized label set (``labels_for``) - kept here,
  out of the schema, so the data model stays pure.

There is one output style, modelled on ``samples/CV Example.pdf`` (``STYLE_A``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from cv_agent.schema import CV, DateRange, MonthYear

TEMPLATES_DIR = Path(__file__).parent / "templates"

_MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# --------------------------------------------------------------------------- #
# Localization: the two fixed headings (EXPERIENCE / EDUCATION) and the ongoing
# date-word. Every OTHER section heading comes verbatim from the CV, so it is not
# translated here. Unknown languages fall back to English; add a block per lang.
# --------------------------------------------------------------------------- #
DEFAULT_LANG = "en"

LABELS = {
    "en": {"experience": "Experience", "education": "Education", "profile": "Profile",
           "current": "Current", "present": "Present"},
    "tr": {"experience": "Deneyim", "education": "Eğitim", "profile": "Önyazı",
           "current": "Hâlen", "present": "Hâlen"},
}

# A profile/summary longer than this many characters is rendered as its own
# top section ("PROFILE" / "ÖNYAZI") instead of the centered header tagline.
SUMMARY_SECTION_THRESHOLD = 450


def labels_for(language: Optional[str]) -> dict:
    """The label set for a CV's ``language`` (matched on its first two letters);
    falls back to English for an unknown or missing language."""
    key = (language or "").strip().lower()[:2]
    return LABELS.get(key, LABELS[DEFAULT_LANG])


def _word(language: Optional[str], key: str) -> str:
    return labels_for(language)[key]


# --------------------------------------------------------------------------- #
# Uppercasing (heading style) - Turkish-aware
# --------------------------------------------------------------------------- #
def uc(value, lang: Optional[str] = None) -> str:
    """Uppercase ``value`` for a section heading. For Turkish, map i->İ and ı->I
    first (Python's ``str.upper`` would otherwise produce a dotless 'I')."""
    s = "" if value is None else str(value)
    if (lang or "").strip().lower().startswith("tr"):
        s = s.replace("ı", "I").replace("i", "İ")
    return s.upper()


# --------------------------------------------------------------------------- #
# LaTeX escaping
# --------------------------------------------------------------------------- #
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
# Longest keys first so multi-char keys (none here, but future-proof) win; the
# single-pass sub means replacements we insert (which contain { }) are NOT
# re-escaped.
_ESCAPE_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(_LATEX_SPECIAL, key=len, reverse=True))
)


def escape_latex(value) -> str:
    """Escape the LaTeX special characters in ``value`` (None -> '')."""
    if value is None:
        return ""
    return _ESCAPE_RE.sub(lambda m: _LATEX_SPECIAL[m.group()], str(value))


# --------------------------------------------------------------------------- #
# Date formatting
# --------------------------------------------------------------------------- #
def _my_a(my: MonthYear) -> str:
    """Point-in-time: ``'24 Jun`` (curly apostrophe + 2-digit year)."""
    s = "’" + f"{my.year % 100:02d}"
    if my.month:
        s += " " + _MONTHS[my.month - 1]
    return s


def range_a(dr: Optional[DateRange], lang: Optional[str] = None) -> str:
    """Experience/entry range, e.g. ``'23 Jun - '24 Jun`` / ``'24 Jun - Current``.
    The ongoing word is localized by ``lang`` (Turkish -> ``Hâlen``)."""
    if dr is None:
        return ""
    start = _my_a(dr.start) if dr.start else ""
    if dr.current:
        cur = _word(lang, "current")
        return f"{start} - {cur}" if start else cur
    if dr.end:
        return f"{start} - {_my_a(dr.end)}"
    return start


def range_years(dr: Optional[DateRange], lang: Optional[str] = None) -> str:
    """Year-only range with spaced hyphen, e.g. ``2022 - 2023`` (education).
    The ongoing word is localized by ``lang``."""
    if dr is None:
        return ""
    start = str(dr.start.year) if dr.start else ""
    if dr.current:
        pres = _word(lang, "present")
        return f"{start} - {pres}" if start else pres
    if dr.end:
        return f"{start} - {dr.end.year}"
    return start


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def build_environment(templates_dir: Path = TEMPLATES_DIR) -> Environment:
    """Create the LaTeX-flavoured Jinja2 environment."""
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
        line_statement_prefix="%%",
        line_comment_prefix="%#",
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        undefined=StrictUndefined,  # a typo'd field name fails loudly, not silently
    )
    env.filters["e"] = escape_latex
    env.filters["uc"] = uc
    env.filters["range_a"] = range_a
    env.filters["range_years"] = range_years
    return env


_ENV: Optional[Environment] = None


def render_cv(cv: CV, template_name: str) -> str:
    """Render ``cv`` through ``template_name`` and return the LaTeX source.

    The fixed EXPERIENCE/EDUCATION headings and the ongoing date word come from
    ``t`` (the label set for ``cv.language``); every other section heading is the
    section's own ``title``, uppercased by the ``uc`` filter."""
    global _ENV
    if _ENV is None:
        _ENV = build_environment()
    return _ENV.get_template(template_name).render(
        cv=cv, t=labels_for(cv.language), summary_threshold=SUMMARY_SECTION_THRESHOLD
    )


# The single built-in style.
STYLE_A = "resume.tex.j2"

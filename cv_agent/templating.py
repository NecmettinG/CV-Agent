r"""Render a validated :class:`~cv_agent.schema.CV` into a LaTeX document.

LaTeX already uses ``{``, ``}`` and ``%``, which collide with Jinja2's default
``{{ }}`` / ``{% %}`` / ``{# #}``. So we configure Jinja2 with LaTeX-safe
delimiters instead:

    variable   \VAR{ ... }
    block       \BLOCK{ ... }
    comment     \#{ ... }

That way a ``.tex.j2`` file is still (almost) valid LaTeX you can read, and the
template engine only reacts to the ``\VAR`` / ``\BLOCK`` markers.

Two more things this module owns:

* ``escape_latex`` - CV text may contain ``&``, ``%``, ``#`` etc.; those must be
  escaped or the PDF won't compile. Registered as the ``e`` filter.
* date formatting - the two styles print dates differently, so the formatting
  lives here (as filters) and NOT in the schema, keeping the data model pure.
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
# Date formatting (one helper per visual style)
# --------------------------------------------------------------------------- #
def _my_a(my: MonthYear) -> str:
    """Style A point-in-time: ``'24 Jun`` (curly apostrophe + 2-digit year)."""
    s = "’" + f"{my.year % 100:02d}"
    if my.month:
        s += " " + _MONTHS[my.month - 1]
    return s


def _my_b(my: MonthYear) -> str:
    """Style B point-in-time: ``06.2024`` or a bare ``2024``."""
    if my.month:
        return f"{my.month:02d}.{my.year}"
    return str(my.year)


def range_a(dr: Optional[DateRange]) -> str:
    """Style A range, e.g. ``'23 Jun - '24 Jun`` / ``'24 Jun - Current``."""
    if dr is None:
        return ""
    start = _my_a(dr.start) if dr.start else ""
    if dr.current:
        return f"{start} - Current" if start else "Current"
    if dr.end:
        return f"{start} - {_my_a(dr.end)}"
    return start


def range_b(dr: Optional[DateRange]) -> str:
    """Style B range, e.g. ``07.2024 – 09.2024`` / ``06.2025- ongoing`` /
    year-only ``2021-2026``."""
    if dr is None:
        return ""
    start = _my_b(dr.start) if dr.start else ""
    if dr.current:
        return f"{start}- ongoing" if start else "ongoing"
    if dr.end:
        end = _my_b(dr.end)
        # Year-only ranges use a plain hyphen (2021-2026); dated ranges an en dash.
        if dr.start and dr.start.month is None and dr.end and dr.end.month is None:
            return f"{start}-{end}"
        return f"{start} – {end}"
    return start


def range_years(dr: Optional[DateRange]) -> str:
    """Year-only range with spaced hyphen, e.g. ``2022 - 2023`` (Style A edu)."""
    if dr is None:
        return ""
    start = str(dr.start.year) if dr.start else ""
    if dr.current:
        return f"{start} - Present" if start else "Present"
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
    env.filters["range_a"] = range_a
    env.filters["range_b"] = range_b
    env.filters["range_years"] = range_years
    return env


_ENV: Optional[Environment] = None


def render_cv(cv: CV, template_name: str) -> str:
    """Render ``cv`` through ``template_name`` and return the LaTeX source."""
    global _ENV
    if _ENV is None:
        _ENV = build_environment()
    return _ENV.get_template(template_name).render(cv=cv)


# Convenience aliases for the two built-in styles.
STYLE_A = "resume.tex.j2"
STYLE_B = "resume2.tex.j2"

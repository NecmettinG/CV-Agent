"""CV-Agent: turn PDF/DOCX/raw-text CVs into an ATS-friendly LaTeX PDF.

The public data contract lives in :mod:`cv_agent.schema`; templating to a LaTeX
string lives in :mod:`cv_agent.templating`; compiling a CV all the way to a PDF
lives in :mod:`cv_agent.render` (:func:`render_pdf`).
"""

from cv_agent.render import render_pdf
from cv_agent.templating import STYLE_A, STYLE_B, render_cv
from cv_agent.schema import (
    CV,
    Competition,
    Contact,
    DateRange,
    EducationEntry,
    ExperienceEntry,
    Language,
    Link,
    MonthYear,
    Project,
    Reference,
    SkillCategory,
    Skills,
    SubRole,
)

__all__ = [
    "CV",
    "Competition",
    "Contact",
    "DateRange",
    "EducationEntry",
    "ExperienceEntry",
    "Language",
    "Link",
    "MonthYear",
    "Project",
    "Reference",
    "SkillCategory",
    "Skills",
    "SubRole",
    # rendering / compilation
    "render_cv",
    "render_pdf",
    "STYLE_A",
    "STYLE_B",
]

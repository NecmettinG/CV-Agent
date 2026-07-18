"""CV-Agent: turn PDF/DOCX/raw-text CVs into an ATS-friendly LaTeX PDF.

The public data contract lives in :mod:`cv_agent.schema`; templating to a LaTeX
string lives in :mod:`cv_agent.templating`; compiling a CV all the way to a PDF
lives in :mod:`cv_agent.render` (:func:`render_pdf`).
"""

from cv_agent.extract import ExtractionError, extract_cv
from cv_agent.pipeline import cv_from_file, file_to_pdf, parse_file
from cv_agent.providers import PRESETS, Provider, build_provider
from cv_agent.render import render_pdf
from cv_agent.templating import STYLE_A, render_cv
from cv_agent.schema import (
    CV,
    Contact,
    DateRange,
    EducationEntry,
    ExperienceEntry,
    Link,
    MonthYear,
    Section,
    SectionEntry,
    SubRole,
)

__all__ = [
    # schema (the CV data contract)
    "CV",
    "Contact",
    "DateRange",
    "EducationEntry",
    "ExperienceEntry",
    "Link",
    "MonthYear",
    "Section",
    "SectionEntry",
    "SubRole",
    # extraction (LLM: text -> CV) + provider selection
    "extract_cv",
    "ExtractionError",
    "PRESETS",
    "Provider",
    "build_provider",
    # end-to-end pipeline (file -> CV -> PDF)
    "parse_file",
    "cv_from_file",
    "file_to_pdf",
    # rendering / compilation
    "render_cv",
    "render_pdf",
    "STYLE_A",
]

"""CV-Agent: turn PDF/DOCX/raw-text CVs into an ATS-friendly LaTeX PDF.

The public data contract lives in :mod:`cv_agent.schema`; templating to a LaTeX
string lives in :mod:`cv_agent.templating`; compiling a CV all the way to a PDF
lives in :mod:`cv_agent.render` (:func:`render_pdf`).
"""

from cv_agent.ats import (
    AtsError,
    AtsReport,
    Keyword,
    add_declared_skills,
    apply_keyword_decisions,
    ats_report,
    extract_job_keywords,
    improve_cv,
    keyword_coverage,
    newly_surfaced_keywords,
    roundtrip_parse,
    weavable_entries,
    weave_skills,
)
from cv_agent.extract import ExtractionError, build_cv, extract_cv
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
    "build_cv",
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
    # ATS analysis (round-trip parse, keyword coverage, score, guarded rewrite)
    "ats_report",
    "AtsReport",
    "AtsError",
    "Keyword",
    "roundtrip_parse",
    "keyword_coverage",
    "newly_surfaced_keywords",
    "apply_keyword_decisions",
    "add_declared_skills",
    "weavable_entries",
    "weave_skills",
    "extract_job_keywords",
    "improve_cv",
]

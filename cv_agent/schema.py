"""Pydantic data model for a CV, shaped after the Cambridge-Oxford format(s).

This module is the single source of truth for what a CV *is* inside CV-Agent.
Everything flows through it:

    raw input (PDF / DOCX / text)  ->  LLM  ->  CV (this schema)  ->  LaTeX -> PDF

Because the LLM returns free-form text, we never trust it directly: we validate
its output into these typed models first. If a field is missing or malformed,
validation fails loudly here instead of producing a broken PDF later.

The model is deliberately a SUPERSET that can feed either output style:

* Style A (samples/CV Example.pdf) - ALL-CAPS section headings, an italic tech
  line + justified paragraph per role, two-column skills, umbrella employers.
* Style B (samples/CV Example 2.pdf) - "Title:" headings, bulleted descriptions,
  extra sections (Competitions, Projects, Languages, References), bold-label
  skills, lots of clickable links.

Each Jinja2 template renders only the fields its style uses, so one CV object
can be poured into either template.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    """Shared config for every model in the schema.

    * ``extra="forbid"`` -> reject unknown keys, so a hallucinated field from the
      LLM surfaces as a validation error rather than being silently dropped.
    * ``str_strip_whitespace`` -> trim stray whitespace the LLM often adds.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #
class Link(_Base):
    """A labelled hyperlink, e.g. a LinkedIn / GitHub / project / repo link.

    ``url`` is what makes links actually *clickable* in the output PDF - the
    templates render every Link as ``\\href{url}{label}``.
    """

    label: str = Field(..., description="Text shown to the reader, e.g. 'LinkedIn'.")
    url: str = Field(..., description="Destination URL, e.g. 'https://...'.")


class MonthYear(_Base):
    """A point in time. ``month`` is optional so the same type serves both
    experience dates (``'24 Jun`` / ``06.2024``) and year-only education dates."""

    year: int = Field(..., ge=1900, le=2100)
    month: Optional[int] = Field(default=None, ge=1, le=12)


class DateRange(_Base):
    """A start/end span. Rendering (not the schema) decides the visual style:
    Style A uses ``'YY Mon``, Style B uses ``MM.YYYY`` or a bare year.

    * A single date (e.g. a graduation year): set ``start`` only.
    * An ongoing role: set ``current=True`` and leave ``end`` empty.
    """

    start: Optional[MonthYear] = None
    end: Optional[MonthYear] = None
    current: bool = Field(
        default=False, description="True for a still-active role (renders 'Current'/'ongoing')."
    )

    @model_validator(mode="after")
    def _clear_end_when_current(self) -> "DateRange":
        if self.current:
            self.end = None
        return self


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
class Contact(_Base):
    """The contact line under the name. All fields optional so we render only
    what the input provides. ``email``/``phone`` become mailto:/tel: links;
    ``links`` holds extra profiles (LinkedIn, GitHub, website)."""

    location: Optional[str] = Field(default=None, description="e.g. 'Istanbul'.")
    phone: Optional[str] = None
    email: Optional[str] = None
    links: List[Link] = Field(
        default_factory=list, description="Extra profile links (LinkedIn, GitHub, site)."
    )


# --------------------------------------------------------------------------- #
# Experience
# --------------------------------------------------------------------------- #
class SubRole(_Base):
    """A single position nested under an umbrella employer (the agency case in
    Style A). Carries its own company/title/tech/description but NOT its own
    dates - those live on the parent :class:`ExperienceEntry`."""

    company: str = Field(..., description="Sub-employer name, e.g. 'Finwave'.")
    title: str = Field(..., description="Role title, e.g. 'Senior Backend Engineer'.")
    tech_stack: List[str] = Field(default_factory=list, description="Italic tech line.")
    description: Optional[str] = None
    highlights: List[str] = Field(
        default_factory=list, description="Bullet points (used by bulleted styles)."
    )
    links: List[Link] = Field(
        default_factory=list, description="Trailing clickable links (e.g. project names)."
    )


class ExperienceEntry(_Base):
    """One block in the EXPERIENCE section. Two shapes, mutually exclusive:

    * **Normal role** - ``title`` set, ``sub_roles`` empty.
    * **Umbrella employer** - ``sub_roles`` non-empty, ``title`` empty (an agency).

    Style A uses ``tech_stack`` + ``description`` (paragraph); Style B uses
    ``highlights`` (bullets) plus ``location`` / ``work_mode`` /
    ``employment_type``. A role may fill whichever its target style needs.
    """

    company: str = Field(..., description="Employer name, e.g. 'PayNova' or 'TalentBridge'.")
    title: Optional[str] = Field(
        default=None, description="Role title for a normal entry; empty for umbrella."
    )
    location: Optional[str] = Field(default=None, description="e.g. 'Istanbul, Turkiye'.")
    work_mode: Optional[str] = Field(default=None, description="e.g. 'Hybrid', 'Remote'.")
    employment_type: Optional[str] = Field(
        default=None, description="Shown in parens, e.g. 'Volunteer', 'Intern'."
    )
    date_range: DateRange
    tech_stack: List[str] = Field(default_factory=list, description="Italic tech line (Style A).")
    description: Optional[str] = Field(default=None, description="Paragraph body (Style A).")
    highlights: List[str] = Field(default_factory=list, description="Bullet points (Style B).")
    links: List[Link] = Field(
        default_factory=list, description="Trailing clickable links (e.g. project names)."
    )
    sub_roles: List[SubRole] = Field(
        default_factory=list, description="Nested positions for an umbrella employer."
    )

    @property
    def is_umbrella(self) -> bool:
        """True when this entry groups several sub-roles (the umbrella case)."""
        return bool(self.sub_roles)

    @model_validator(mode="after")
    def _check_shape(self) -> "ExperienceEntry":
        if self.sub_roles and self.title:
            raise ValueError(
                "An umbrella employer (with sub_roles) must not set its own 'title'; "
                "put role titles on each sub-role."
            )
        if not self.sub_roles and not self.title:
            raise ValueError(
                "A normal experience entry needs a 'title' "
                "(or provide 'sub_roles' for an umbrella employer)."
            )
        return self


# --------------------------------------------------------------------------- #
# Education
# --------------------------------------------------------------------------- #
class EducationEntry(_Base):
    """One block in the EDUCATION section.

    Style A uses ``degree`` (italic line) + ``date_range`` + ``location``.
    Style B uses the header + ``highlights`` (bullets), an optional ``url``
    (e.g. a course link), and often no separate ``degree``."""

    institution: str = Field(..., description="e.g. 'Metropolitan Technical University'.")
    degree: Optional[str] = Field(default=None, description="e.g. \"Computer Engineering\".")
    location: Optional[str] = Field(default=None, description="e.g. 'Istanbul, Turkey'.")
    date_range: Optional[DateRange] = None
    highlights: List[str] = Field(default_factory=list, description="Bullet points (Style B).")
    links: List[Link] = Field(
        default_factory=list, description="Nested sub-links under the last bullet (e.g. course repos)."
    )
    url: Optional[str] = Field(default=None, description="Makes the header a clickable link.")


# --------------------------------------------------------------------------- #
# Competitions / Projects / Languages / References (mostly Style B)
# --------------------------------------------------------------------------- #
class Competition(_Base):
    """One entry in a COMPETITIONS section."""

    title: str = Field(..., description="e.g. 'Su Hackathonu (Water Hackathon): 2026'.")
    location: Optional[str] = None
    url: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)


class Project(_Base):
    """One entry in a PROJECTS section: a bold (optionally linked) name, a
    description, and optional nested links / bullets."""

    name: str = Field(..., description="Project name, shown bold; linked if 'url' set.")
    description: Optional[str] = None
    url: Optional[str] = Field(default=None, description="Makes the name a clickable link.")
    links: List[Link] = Field(
        default_factory=list, description="Nested sub-links (e.g. component repos)."
    )
    highlights: List[str] = Field(default_factory=list)


class Language(_Base):
    """One entry in a LANGUAGES section."""

    name: str = Field(..., description="e.g. 'English'.")
    level: Optional[str] = Field(default=None, description="e.g. 'C1', 'Native'.")


class Reference(_Base):
    """One entry in a REFERENCES section."""

    name: str = Field(..., description="e.g. 'Alex Thompson'.")
    detail: Optional[str] = Field(
        default=None, description="e.g. 'Software Engineer III, ING Hubs Turkey'."
    )
    url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Skills
# --------------------------------------------------------------------------- #
class SkillCategory(_Base):
    """A bold-label skill bullet (Style B), e.g. **Java:** Spring Boot, JPA..."""

    name: str = Field(..., description="Bold label, e.g. 'Java'.")
    detail: Optional[str] = Field(default=None, description="Text after the label.")
    emphasis: bool = Field(default=False, description="Render the label bold-italic.")


class Skills(_Base):
    """The skills section. Style A uses ``primary`` + ``good_to_mention``;
    Style B uses ``categories``. Templates read whichever they need."""

    primary: List[str] = Field(
        default_factory=list, description="Bulleted skills (Style A two-column list)."
    )
    good_to_mention: List[str] = Field(
        default_factory=list, description="Secondary '(Good to mention: ...)' skills (Style A)."
    )
    categories: List[SkillCategory] = Field(
        default_factory=list, description="Bold-label skill bullets (Style B)."
    )


# --------------------------------------------------------------------------- #
# Top-level document
# --------------------------------------------------------------------------- #
class CV(_Base):
    """A complete CV: the object the renderer turns into LaTeX -> PDF."""

    name: str = Field(..., min_length=1, description="Full name, shown bold.")
    summary: Optional[str] = Field(
        default=None, description="One-line italic tagline under the contact row."
    )
    contact: Contact = Field(default_factory=Contact)
    experience: List[ExperienceEntry] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    competitions: List[Competition] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    skills: Optional[Skills] = None
    languages: List[Language] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)
    references_note: Optional[str] = Field(
        default=None, description="Bold-italic note under references, e.g. 'shared on request'."
    )


if __name__ == "__main__":
    # Smoke test: build a fragment covering the tricky "umbrella" case and
    # a Style-B section, then prove it validates and round-trips as JSON.
    cv = CV(
        name="Jordan Mercer",
        summary="Proactive problem solver with a passion for continuous improvement.",
        contact=Contact(location="Berlin", phone="+49 30 5550100", email="jordan.mercer@example.com"),
        experience=[
            ExperienceEntry(
                company="PayNova",
                title="Software Team Lead",
                date_range=DateRange(start=MonthYear(year=2024, month=6), current=True),
                tech_stack=["Project Management", "Java 8-11-21", "Spring Boot 2-3"],
                description="Leading a cross-functional team on the payback lifecycle.",
            ),
            ExperienceEntry(
                company="TalentBridge",
                date_range=DateRange(
                    start=MonthYear(year=2022, month=3), end=MonthYear(year=2023, month=6)
                ),
                sub_roles=[
                    SubRole(
                        company="Finwave",
                        title="Senior Backend Engineer",
                        tech_stack=["Java", "Spring Boot", "AWS"],
                        description="Designed microservices from beginning to end.",
                        links=[Link(label="Product One", url="https://example.com/product-one")],
                    )
                ],
            ),
        ],
        projects=[
            Project(
                name="SmartShop",
                description="E-commerce infrastructure with Spring Boot and AWS.",
                url="https://example.com/smartshop",
            )
        ],
        skills=Skills(categories=[SkillCategory(name="Java", detail="Spring Boot, JPA, JUnit.")]),
        languages=[Language(name="English", level="C1")],
        references=[Reference(name="Alex Thompson", detail="Software Engineer III, TechCorp")],
        references_note="Contact information will be shared upon request.",
    )

    print("Validated OK.")
    print("Umbrella entry is umbrella:", cv.experience[1].is_umbrella)
    payload = cv.model_dump_json(indent=2)
    reparsed = CV.model_validate_json(payload)
    assert reparsed == cv, "round-trip mismatch"
    print("JSON round-trip OK ({} chars).".format(len(payload)))

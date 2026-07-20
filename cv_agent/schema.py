"""Pydantic data model for a CV, shaped after the Cambridge-Oxford format.

This module is the single source of truth for what a CV *is* inside CV-Agent.
Everything flows through it:

    raw input (PDF / DOCX / text)  ->  LLM  ->  CV (this schema)  ->  LaTeX -> PDF

Because the LLM returns free-form text, we never trust it directly: we validate
its output into these typed models first. If a field is missing or malformed,
validation fails loudly here instead of producing a broken PDF later.

The output format is a single style, modelled on ``samples/CV Example.pdf``:
ALL-CAPS section headings with a full-width rule, an italic tech line + paragraph
(or bullets) per role, two-column skills, umbrella employers.

**Header + experience + education are typed** (they are universal and need a
specific layout). **Everything else is a dynamic** :class:`Section`: its heading
is taken verbatim from the CV, so any section a CV happens to have - Projects,
Certificates, Community, Volunteering, References, Interests, ... - is preserved
under its own name instead of being forced into a fixed slot.
"""

from __future__ import annotations

from typing import Annotated, List, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    """Shared config for every model in the schema.

    * ``extra="forbid"`` -> reject unknown keys, so a hallucinated field from the
      LLM surfaces as a validation error rather than being silently dropped.
    * ``str_strip_whitespace`` -> trim stray whitespace the LLM often adds.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _with_scheme(u: str) -> str:
    """Ensure a web URL has a scheme so ``\\href`` treats it as absolute. A CV
    often writes a bare domain ('kayhanakbay.com'); without ``https://`` hyperref
    makes it a broken relative link."""
    u = u.strip()
    if not u or "://" in u or u.startswith(("mailto:", "tel:")):
        return u
    return "https://" + u


#: A URL string, normalized to always carry a scheme.
Url = Annotated[str, AfterValidator(_with_scheme)]

#: Known profiles: a link whose *label* is a bare URL of one of these is shown by
#: its platform name instead (e.g. 'linkedin.com/in/jane' -> 'LinkedIn').
_PLATFORM_LABELS = {
    "linkedin.com": "LinkedIn", "github.com": "GitHub", "gitlab.com": "GitLab",
    "bitbucket.org": "Bitbucket", "kaggle.com": "Kaggle", "medium.com": "Medium",
    "behance.net": "Behance", "dribbble.com": "Dribbble", "stackoverflow.com": "Stack Overflow",
    "twitter.com": "Twitter", "x.com": "X", "instagram.com": "Instagram",
    "youtube.com": "YouTube", "facebook.com": "Facebook",
}


def _tidy_label(label: str) -> str:
    """Shorten a link label that is itself a bare platform URL to the platform
    name ('linkedin.com/in/jane' -> 'LinkedIn'). Descriptive labels (with spaces)
    and non-platform domains are left untouched."""
    lab = label.strip()
    if lab and " " not in lab and ("." in lab or "://" in lab):
        low = lab.lower()
        for domain, name in _PLATFORM_LABELS.items():
            if domain in low:
                return name
    return label


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #
class Link(_Base):
    """A labelled hyperlink, e.g. a LinkedIn / GitHub / project / repo link.

    ``url`` is what makes links actually *clickable* in the output PDF - the
    template renders every Link as ``\\href{url}{label}``.
    """

    label: str = Field(..., description="Text shown to the reader, e.g. 'LinkedIn'.")
    url: Url = Field(..., description="Destination URL, e.g. 'https://...'.")

    @model_validator(mode="after")
    def _shorten_url_label(self) -> "Link":
        self.label = _tidy_label(self.label)
        return self


class MonthYear(_Base):
    """A point in time. ``month`` is optional so the same type serves both dated
    experience (``'24 Jun``) and year-only education dates."""

    year: int = Field(..., ge=1900, le=2100)
    month: Optional[int] = Field(default=None, ge=1, le=12)


class DateRange(_Base):
    """A start/end span rendered as ``'23 Jun - '24 Jun`` / ``'24 Jun - Current``.

    * A single date (e.g. a graduation year): set ``start`` only.
    * An ongoing role: set ``current=True`` and leave ``end`` empty.
    """

    start: Optional[MonthYear] = None
    end: Optional[MonthYear] = None
    current: bool = Field(
        default=False, description="True for a still-active role (renders 'Current')."
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
    """A single position nested under an umbrella employer (the agency case).
    Carries its own company/title/tech/description but NOT its own dates - those
    live on the parent :class:`ExperienceEntry`."""

    company: str = Field(..., description="Sub-employer name, e.g. 'Finwave'.")
    title: str = Field(..., description="Role title, e.g. 'Senior Backend Engineer'.")
    tech_stack: List[str] = Field(default_factory=list, description="Italic tech line.")
    description: Optional[str] = Field(default=None, description="Paragraph body.")
    highlights: List[str] = Field(default_factory=list, description="Bullet points.")
    links: List[Link] = Field(
        default_factory=list, description="Trailing clickable links (e.g. project names)."
    )


class ExperienceEntry(_Base):
    """One block in the EXPERIENCE section. Two shapes, mutually exclusive:

    * **Normal role** - ``title`` set, ``sub_roles`` empty.
    * **Umbrella employer** - ``sub_roles`` non-empty, ``title`` empty (an agency).

    A role's body may be a ``description`` paragraph, ``highlights`` bullets, or
    both - use whichever the source CV uses.
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
    tech_stack: List[str] = Field(default_factory=list, description="Italic tech/skills line.")
    description: Optional[str] = Field(default=None, description="Paragraph body.")
    highlights: List[str] = Field(default_factory=list, description="Bullet points.")
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
    """One block in the EDUCATION section: bold ``institution`` (+ ``location``)
    on line one, italic ``degree`` (+ ``date_range``) on line two, optional
    ``highlights`` bullets, and an optional ``url`` that links the header."""

    institution: str = Field(..., description="e.g. 'Metropolitan Technical University'.")
    degree: Optional[str] = Field(default=None, description="e.g. 'Computer Engineering'.")
    location: Optional[str] = Field(default=None, description="e.g. 'Istanbul, Turkey'.")
    date_range: Optional[DateRange] = None
    highlights: List[str] = Field(default_factory=list, description="Bullet points.")
    links: List[Link] = Field(
        default_factory=list, description="Sub-links (e.g. course repos) under the bullets."
    )
    url: Optional[Url] = Field(default=None, description="Makes the header a clickable link.")


# --------------------------------------------------------------------------- #
# Dynamic sections (everything that is not experience/education)
# --------------------------------------------------------------------------- #
class SectionEntry(_Base):
    """One titled entry inside a ``kind='entries'`` :class:`Section` - a project,
    certificate, award, competition, or a reference."""

    title: str = Field(
        ..., description="Bold lead text: a project/certificate/award name, or a reference's name."
    )
    detail: Optional[str] = Field(
        default=None,
        description="Text under the title: a description, or a reference's 'Title, Company'.",
    )
    date_range: Optional[DateRange] = Field(
        default=None, description="Optional date/year, shown on the right."
    )
    url: Optional[Url] = Field(
        default=None, description="If set, the TITLE becomes a link to this URL (e.g. a project repo)."
    )
    phone: Optional[str] = Field(
        default=None, description="Phone (e.g. a reference's), copied verbatim -> tel: link."
    )
    email: Optional[str] = Field(
        default=None, description="Email (e.g. a reference's) -> mailto: link."
    )
    links: List[Link] = Field(
        default_factory=list, description="Extra clickable links (repo sub-links, a reference's site)."
    )
    highlights: List[str] = Field(default_factory=list, description="Bullet points under the entry.")


_SECTION_KINDS = {"list", "skills", "entries", "text"}


class Section(_Base):
    """A dynamic CV section. Its heading comes from the CV itself, so ANY section
    is supported (Projects, Certificates, Community, Volunteering, References,
    Interests, Personal Details, ...). ``kind`` selects how the body is rendered.
    """

    title: str = Field(
        ...,
        description="Section heading, VERBATIM from the CV in its own language "
        "(e.g. 'Projects', 'Community', 'References', 'Certificates').",
    )
    kind: str = Field(
        default="list",
        description="Render shape: 'list' (bullet list: interests, languages, community), "
        "'skills' (two-column bullet list), 'entries' (titled entries: projects, references, "
        "certificates, competitions), or 'text' (one paragraph).",
    )
    bullets: List[str] = Field(
        default_factory=list, description="Items for kind 'list'/'skills' (one string per bullet)."
    )
    entries: List[SectionEntry] = Field(
        default_factory=list, description="Entries for kind 'entries'."
    )
    text: Optional[str] = Field(default=None, description="Paragraph for kind 'text'.")
    note: Optional[str] = Field(
        default=None,
        description="A short trailing note under the section, e.g. 'References available on "
        "request.' Rendered bold-italic after the section body.",
    )

    @model_validator(mode="after")
    def _normalize_kind(self) -> "Section":
        k = (self.kind or "").strip().lower()
        if k not in _SECTION_KINDS:
            # Unknown kind from the model: infer a safe one from the populated payload.
            if self.entries:
                k = "entries"
            elif self.text and not self.bullets:
                k = "text"
            else:
                k = "list"
        self.kind = k
        return self

    @property
    def has_content(self) -> bool:
        """True if the section carries anything worth rendering."""
        return bool(self.bullets or self.entries or self.text or self.note)


# --------------------------------------------------------------------------- #
# Top-level document
# --------------------------------------------------------------------------- #
class CV(_Base):
    """A complete CV: the object the renderer turns into LaTeX -> PDF."""

    name: str = Field(..., min_length=1, description="Full name, shown bold.")
    headline: Optional[str] = Field(
        default=None,
        description="Professional title line shown under the name (e.g. 'Customer Support "
        "Representative'). Fill ONLY if the CV explicitly has such a line under the name - "
        "never invent one.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Profile / objective / cover-letter text shown italic under the header. "
        "A short paragraph is fine - transcribe it faithfully, do not shorten it.",
    )
    language: Optional[str] = Field(
        default=None,
        description="ISO 639-1 code of the CV's PRIMARY language ('en', 'tr', ...). Sets the "
        "language of the EXPERIENCE / EDUCATION headings. Omit only if unsure.",
    )
    contact: Contact = Field(default_factory=Contact)
    experience: List[ExperienceEntry] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    sections: List[Section] = Field(
        default_factory=list,
        description="Every other section, in the CV's own order, each under its own heading "
        "(Skills, Projects, Languages, Community, Interests, References, ...).",
    )


if __name__ == "__main__":
    # Smoke test: cover the umbrella experience case + a few dynamic sections,
    # then prove it validates and round-trips as JSON.
    cv = CV(
        name="Jordan Mercer",
        headline="Software Team Lead",
        summary="Proactive problem solver with a passion for continuous improvement.",
        language="en",
        contact=Contact(location="Berlin", phone="+49 30 5550100", email="jordan.mercer@example.com"),
        experience=[
            ExperienceEntry(
                company="PayNova",
                title="Software Team Lead",
                date_range=DateRange(start=MonthYear(year=2024, month=6), current=True),
                tech_stack=["Java 8-11-21", "Spring Boot 2-3"],
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
        education=[
            EducationEntry(
                institution="Metropolitan Technical University",
                degree="Computer Engineering",
                location="Berlin",
                date_range=DateRange(start=MonthYear(year=2016), end=MonthYear(year=2020)),
            )
        ],
        sections=[
            Section(title="Skills", kind="skills", bullets=["Java, Spring Boot", "AWS", "PostgreSQL"]),
            Section(
                title="Projects",
                kind="entries",
                entries=[
                    SectionEntry(
                        title="SmartShop",
                        detail="E-commerce infrastructure with Spring Boot and AWS.",
                        url="https://example.com/smartshop",
                    )
                ],
            ),
            Section(title="Languages", kind="list", bullets=["English (Native)", "German (B2)"]),
            Section(
                title="References",
                kind="entries",
                entries=[
                    SectionEntry(
                        title="Alex Thompson",
                        detail="Software Engineer III, TechCorp",
                        email="alex.thompson@example.com",
                    )
                ],
            ),
        ],
    )

    print("Validated OK.")
    print("Umbrella entry is umbrella:", cv.experience[1].is_umbrella)
    print("Section kinds:", [s.kind for s in cv.sections])
    payload = cv.model_dump_json(indent=2)
    reparsed = CV.model_validate_json(payload)
    assert reparsed == cv, "round-trip mismatch"
    print("JSON round-trip OK ({} chars).".format(len(payload)))

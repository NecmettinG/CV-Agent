"""One anonymous sample CV, expressed as a validated :class:`CV` object.

Everything here is FICTIONAL - all names, contact details, employers, schools,
references and URLs are invented (``example.com``). The sample is modelled on the
output format's *structure* so it exercises every template feature:

* a header with headline + summary + profile links,
* a normal role, an umbrella employer with sub-roles, and a bullet-only role,
* education,
* dynamic sections of every ``kind``: ``skills`` (two-column), ``entries``
  (projects / references, with links + a reference email), and ``list``
  (languages / community / interests).

It doubles as (a) a fixture to verify the template compiles and (b) a worked
example of how to populate the schema.
"""

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


def _my(year: int, month: int | None = None) -> MonthYear:
    return MonthYear(year=year, month=month)


sample_cv = CV(
    name="Jordan Mercer",
    headline="Senior Backend Engineer",
    summary=(
        "Proactive problem solver with a passion for continuous improvement, a strong "
        "communicator, and an enthusiastic knowledge sharer."
    ),
    language="en",
    contact=Contact(
        location="Berlin",
        phone="+49 30 5550100",
        email="jordan.mercer@example.com",
        links=[
            Link(label="LinkedIn", url="https://example.com/in/jordan-mercer"),
            Link(label="GitHub", url="https://example.com/jmercer"),
        ],
    ),
    experience=[
        ExperienceEntry(
            company="PayNova",
            title="Software Team Lead",
            date_range=DateRange(start=_my(2024, 6), current=True),
            tech_stack=["Project Management", "Java 8-11-21", "Spring Boot 2-3", "Kubernetes", "Docker", "CI/CD"],
            description=(
                "Leading a cross-functional team on the company-wide payback lifecycle, "
                "processing millions of transactions per day without a hitch."
            ),
        ),
        ExperienceEntry(
            company="TalentBridge",
            date_range=DateRange(start=_my(2021, 3), end=_my(2024, 6)),
            sub_roles=[
                SubRole(
                    company="Finwave",
                    title="Senior Backend Engineer",
                    tech_stack=["Java", "Spring Boot", "AWS", "Kafka"],
                    description="Designed and built fintech microservices from the ground up.",
                    links=[Link(label="Finwave BNPL", url="https://example.com/finwave-bnpl")],
                ),
                SubRole(
                    company="Marketly",
                    title="Backend Engineer",
                    tech_stack=["Java", "PostgreSQL", "RabbitMQ"],
                    description="Improved warehouse-management and payment features for an online marketplace.",
                ),
            ],
        ),
        ExperienceEntry(
            company="Kartos",
            title="Software Engineer",
            date_range=DateRange(start=_my(2019, 7), end=_my(2021, 1)),
            highlights=[
                "Worked on a mobile app with over 5 million active users.",
                "Helped transform a monolith into microservices.",
            ],
        ),
    ],
    education=[
        EducationEntry(
            institution="Metropolitan Technical University",
            degree="MSc Computer Engineering",
            location="Berlin, Germany",
            date_range=DateRange(start=_my(2017), end=_my(2019)),
        ),
        EducationEntry(
            institution="Riverside University",
            degree="BSc Computer Science (Erasmus)",
            location="Krakow, Poland",
            date_range=DateRange(start=_my(2013), end=_my(2017)),
        ),
    ],
    sections=[
        Section(
            title="Skills",
            kind="skills",
            bullets=[
                "Java, Spring Boot", "Redis", "SQL (PostgreSQL, MySQL)", "RabbitMQ, Kafka",
                "Unit Testing (JUnit, Mockito)", "AWS", "Docker, Kubernetes", "CI/CD",
            ],
        ),
        Section(
            title="Projects",
            kind="entries",
            entries=[
                SectionEntry(
                    title="SmartShop",
                    detail="E-commerce infrastructure with Spring Boot and AWS.",
                    url="https://example.com/smartshop",
                    links=[Link(label="backend", url="https://example.com/smartshop-backend")],
                ),
                SectionEntry(
                    title="Waterwatch",
                    detail="Water-consumption tracker built for a civic hackathon.",
                    date_range=DateRange(start=_my(2023)),
                    highlights=["4th place among 21 teams."],
                ),
            ],
        ),
        Section(title="Languages", kind="list", bullets=["English (Native)", "German (C1)", "Polish (B1)"]),
        Section(
            title="Community",
            kind="list",
            bullets=["Mentor at Code Bridge Foundation.", "Volunteer at Open Source Weekend."],
        ),
        Section(title="Interests", kind="list", bullets=["Chess", "Trail running", "Model building"]),
        Section(
            title="References",
            kind="entries",
            entries=[
                SectionEntry(
                    title="Alex Thompson",
                    detail="Engineering Manager, PayNova",
                    phone="+49 30 5550199",
                    email="alex.thompson@example.com",
                ),
                SectionEntry(
                    title="Sam Rivera",
                    detail="CTO, Finwave",
                    email="sam.rivera@example.com",
                    links=[Link(label="samrivera.example.com", url="https://samrivera.example.com")],
                ),
            ],
        ),
    ],
)


# Keyed by template file name so render_samples.py can render each.
SAMPLES = {"resume.tex.j2": sample_cv}

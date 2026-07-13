"""Two anonymous sample CVs, expressed as validated :class:`CV` objects.

These are FICTIONAL fixtures - all names, contact details, employers, schools,
references and URLs are invented (``example.com``). They are modelled only on the
*structure* of the two output styles, so they exercise every template feature:

* ``style_a_cv`` - Style A (resume.tex.j2): umbrella employer with sub-roles,
  italic tech lines + paragraph descriptions, trailing links, two-column skills.
* ``style_b_cv`` - Style B (resume2.tex.j2): bulleted entries, Competitions /
  Projects / Languages / References, nested sub-links, bold-label skills.

They double as (a) fixtures to verify the templates compile and (b) worked
examples of how to populate the schema.
"""

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


def _my(year: int, month: int | None = None) -> MonthYear:
    return MonthYear(year=year, month=month)


# =========================================================================== #
# Style A - fictional senior-engineer persona
# =========================================================================== #
style_a_cv = CV(
    name="Jordan Mercer",
    summary=(
        "Proactive problem solver with a passion for continuous improvement, "
        "strong communicator, and knowledge sharer."
    ),
    contact=Contact(
        location="Berlin",
        phone="+49 30 5550100",
        email="jordan.mercer@example.com",
    ),
    experience=[
        ExperienceEntry(
            company="PayNova",
            title="Software Team Lead",
            date_range=DateRange(start=_my(2024, 6), current=True),
            tech_stack=[
                "Project Management", "People Management", "Java 8-11-21",
                "Spring Boot 2-3", "Kubernetes", "Docker", "CI/CD",
            ],
            description=(
                "Leading a cross-functional team and handling operations with business "
                "partners across multiple projects that control the company-wide payback "
                "lifecycle. With more than 125k+ customers and 300k+ daily transactions, the "
                "team gathers, prepares, calculates and correctly pays more than ₺5 billion "
                "per month to customers daily, without any problems."
            ),
        ),
        ExperienceEntry(
            company="Helix Group",
            title="Software Team Lead",
            date_range=DateRange(start=_my(2023, 6), end=_my(2024, 6)),
            tech_stack=[
                "Project Management", "People Management", "Java 8-11-21",
                "Spring Boot 3", "Angular 15-17", "Docker", "CI/CD",
            ],
            description=(
                "Managing multiple projects with a team of frontend and backend engineers, "
                "making architectural decisions, and taking the initiative for larger success "
                "in technical and customer-oriented aspects of product development. Hiring and "
                "mentoring new team members while still actively contributing to code."
            ),
        ),
        ExperienceEntry(
            company="TalentBridge",
            date_range=DateRange(start=_my(2022, 3), end=_my(2023, 6)),
            sub_roles=[
                SubRole(
                    company="Finwave",
                    title="Senior Backend Engineer",
                    tech_stack=[
                        "Java", "Spring Boot", "Snowflake", "MongoDB", "OpenAPI",
                        "Swagger", "AWS", "Kafka", "Docker", "CI/CD", "TDD",
                    ],
                    description=(
                        "Main developer to design, develop and create microservices from "
                        "beginning to end. Built the core application that provides data to "
                        "other microservices, and owned the back end, structuring and "
                        "deployments. Worked on financial solutions and insights services to "
                        "collect, manage and present customer data, and maintained shared "
                        "libraries."
                    ),
                    links=[
                        Link(label="Product One", url="https://example.com/product-one"),
                        Link(label="Product Two", url="https://example.com/product-two"),
                    ],
                ),
                SubRole(
                    company="ShopSphere",
                    title="Senior Backend Engineer",
                    tech_stack=[
                        "Java", "Spring Boot", "PostgreSQL", "AWS", "RabbitMQ",
                        "Docker", "CI/CD", "TDD",
                    ],
                    description=(
                        "Designed and developed features for an online marketplace. Worked on "
                        "warehouse management, last-mile optimization, payment system "
                        "integration and document management systems. Responsible for releases "
                        "and mentorship for newcomers."
                    ),
                ),
            ],
        ),
        ExperienceEntry(
            company="MicroChip Systems",
            title="Senior Software Engineer",
            date_range=DateRange(start=_my(2019, 9), end=_my(2022, 3)),
            tech_stack=["C#", ".Net", "Kotlin", "Android", "Java", "React", "CI/CD"],
            description=(
                "Designed and implemented configuration software and backend APIs for "
                "integrated circuits, mostly for AI, audio and mobile power. Prepared demo "
                "applications for released ICs, initiated projects from scratch to completion, "
                "managed customer communication, reviewed GUIs before release and mentored "
                "newly recruited engineers."
            ),
            links=[
                Link(label="SoC-A100", url="https://example.com/soc-a100"),
                Link(label="Codec-B200", url="https://example.com/codec-b200"),
                Link(label="PMIC-C300 (Series)", url="https://example.com/pmic-c300"),
                Link(label="PMIC-D400 (Series)", url="https://example.com/pmic-d400"),
            ],
        ),
        ExperienceEntry(
            company="SafeGuard Insurance",
            title="Software Engineer",
            date_range=DateRange(start=_my(2018, 1), end=_my(2019, 9)),
            tech_stack=["Java EE", "Oracle Weblogic", "ADF", "JSF", "PL/SQL", "CI/CD"],
            description=(
                "Developed a claims handling system and automated insurance claims before the "
                "actuary to reduce the time (total process reduced by 85%). Also worked with a "
                "rules engine and various middleware products."
            ),
        ),
        ExperienceEntry(
            company="StreamData",
            title="Software Engineer",
            date_range=DateRange(start=_my(2017, 7), end=_my(2018, 1)),
            tech_stack=[
                "Java", "Spring Boot", "MySql", "Redis", "Elastic Search",
                "MongoDB", "Kafka", "RabbitMQ",
            ],
            description=(
                "Worked in the team that developed a loyalty app with over 5 million active "
                "users. Played a role in transforming the system into microservices."
            ),
        ),
        ExperienceEntry(
            company="ClassicList",
            title="Internship",
            date_range=DateRange(start=_my(2016, 6), end=_my(2016, 9)),
            tech_stack=["Java", "Spring Boot", "PHP", "Javascript", "Ajax"],
            description=(
                "Redesigned an existing project that helps customers build their websites "
                "using provided modules with drag-and-drop actions, customize the layout and "
                "configure their websites."
            ),
        ),
    ],
    education=[
        EducationEntry(
            institution="Metropolitan Technical University",
            degree="MBA",
            location="Berlin, Germany",
            date_range=DateRange(start=_my(2022), end=_my(2023)),
        ),
        EducationEntry(
            institution="Metropolitan Technical University",
            degree="Computer Engineering - Bachelor’s (with English prep school)",
            location="Berlin, Germany",
            date_range=DateRange(start=_my(2012), end=_my(2017)),
        ),
        EducationEntry(
            institution="Nordic Institute of Technology",
            degree="Computer Engineering - Bachelor’s (Exchange)",
            location="Oslo, Norway",
            date_range=DateRange(start=_my(2014), end=_my(2015)),
        ),
    ],
    skills=Skills(
        primary=[
            "Java, Spring Boot",
            "Redis",
            "RabbitMQ",
            "Angular",
            "C#, .Net Core",
            "React",
            "Continuous Integration & Continuous Delivery",
            "SQL (PostgreSQL, MySQL, PL/SQL + RDBMS)",
            "Unit Testing (JUnit 4 and 5, JMockit, Mockito)",
            "Oracle ADF and Fusion Middleware products",
            "RabbitMQ",
            "AWS",
        ],
        good_to_mention=[
            "Message Brokers (Kafka)", "MongoDB", "Snowflake", "Python (+Django)",
            "JavaScript (+JQuery)", "PHP", "Assembly x86", "C", "C++", "VB",
        ],
    ),
)


# =========================================================================== #
# Style B - fictional junior-developer persona
# =========================================================================== #
style_b_cv = CV(
    name="Sam Rivera",
    summary=(
        "A technology enthusiast who is hardworker and eager to learn. Performing "
        "tasks with care and responsibility. Filled with determination."
    ),
    contact=Contact(
        email="sam.rivera@example.com",
        phone="+1 555 0142",
        links=[
            Link(label="LinkedIn", url="https://example.com/linkedin"),
            Link(label="GitHub", url="https://example.com/github"),
        ],
    ),
    experience=[
        ExperienceEntry(
            company="Quantum Robotics Club",
            title="Full Stack Developer - UAV Communication - Project Member",
            location="Metropolis",
            work_mode="Hybrid",
            employment_type="Volunteer",
            date_range=DateRange(start=_my(2025, 6), current=True),
            highlights=[
                "Writing Backend with Java Spring Boot and a Ground Control Station user "
                "interface with PySide6.",
                "Responsible for the MAVLink protocol connection between software and UAV with "
                "PyMAVLink.",
                "Also worked with Thymeleaf template engine, HTML, JavaScript, RTSP camera "
                "feed, Linux, Threads, JPA, Redis, and more.",
            ],
        ),
        ExperienceEntry(
            company="SkyTech Aviation",
            title="Backend Developer Intern",
            location="Metropolis",
            work_mode="Hybrid",
            date_range=DateRange(start=_my(2024, 7), end=_my(2024, 9)),
            highlights=[
                "Worked with Java Spring Boot, Spring Framework, PostgreSQL.",
                "Improved Backend knowledge and learned fundamental project structure.",
                "Took part in a project converting a Backend structure from Spring to Spring "
                "Boot.",
                "Also worked with SOAP and RESTful; JPA and JDBC for hibernation.",
            ],
        ),
        ExperienceEntry(
            company="Logix Logistics",
            title="Backend Developer Intern",
            location="Metropolis",
            work_mode="Hybrid",
            date_range=DateRange(start=_my(2023, 6), end=_my(2023, 10)),
            highlights=[
                "Worked with Python Flask Framework, MSSQL and learned the basics of the "
                "Backend field.",
                "Made different projects like sorting a database with 76k rows according to a "
                "numerical attribute.",
                "Improved problem solving skills with exercises from an online judge.",
            ],
        ),
    ],
    education=[
        EducationEntry(
            institution="METROPOLIS UNIVERSITY",
            location="METROPOLIS",
            date_range=DateRange(start=_my(2021), end=_my(2026)),
            highlights=[
                "Bachelor of Computer Engineering, June 2026. GPA: 3.31.",
                "Honor student in Computer Engineering (English) Department.",
            ],
        ),
        EducationEntry(
            institution="Data Science and AI Academy: 2025 Summer",
            location="VIRTUAL",
            highlights=[
                "Learned general information and made practices about Data Science and AI.",
                "Python, Data Preprocessing, Data Visualization, Machine Learning and Deep "
                "Learning were taught.",
                "Aims to use AI-ML fields as a tool in Backend projects (adding intelligence "
                "to the Backend).",
            ],
        ),
        EducationEntry(
            institution="RESTful Web Services with Spring Boot",
            location="Online Course",
            date_range=DateRange(start=_my(2026)),
            url="https://example.com/course",
            highlights=[
                "Learned how to build a RESTful Web Service with Spring Boot, implement User "
                "Sign-up, Token-Based Authentication, Spring Data JPA Query Methods, MySQL, "
                "deploy to Apache Tomcat, use Postman, deploy to a cloud server and Elastic "
                "Beanstalk, use an H2 in-memory database, test endpoints with Rest Assured, "
                "protect services with Spring Security, add Password Reset and Email "
                "Verification, use Native SQL Queries, build with Maven, and test with JUnit 5.",
                "Tutorial project folders:",
            ],
            links=[
                Link(label="web-service-demo", url="https://example.com/repo/web-service-demo"),
                Link(label="verification-service", url="https://example.com/repo/verification-service"),
                Link(label="rest-assured-tests", url="https://example.com/repo/rest-assured-tests"),
                Link(label="mvc-example", url="https://example.com/repo/mvc-example"),
            ],
        ),
    ],
    competitions=[
        Competition(
            title="Tech Hackathon: 2026",
            location="METROPOLIS",
            highlights=[
                "4th Position among 21 teams.",
                "Organized by a regional technology platform together with the municipality "
                "and national ministries of industry and environment.",
            ],
        ),
    ],
    projects=[
        Project(
            name="SmartShop",
            url="https://example.com/repo/smartshop",
            description=(
                "E-commerce infrastructure with Java Spring Boot, Python FastAPI, React "
                "Next.js, and AWS technologies."
            ),
        ),
        Project(
            name="web-service-demo",
            url="https://example.com/repo/web-service-demo",
            description=(
                "A tutorial project for learning a complete institutional project structure "
                "with Java Spring Boot, AUTH, AWS, and unit-integration testing (links were "
                "given above in the “Education” heading)."
            ),
        ),
        Project(
            name="stock-tracker",
            url="https://example.com/repo/stock-tracker",
            description=(
                "A stock market project for enhancing Java Spring Boot skills and stock market "
                "business logic."
            ),
        ),
        Project(
            name="water-monitor",
            url="https://example.com/repo/water-monitor",
            description=(
                "The backend code for the “Tech Hackathon” competition, written with Java "
                "Spring Boot. Tracks water consumption."
            ),
        ),
        Project(
            name="Ground control station",
            description=(
                "Ground control station codes for a UAV project. Written with Java Spring "
                "Boot for the cargo interface backend, Python PyMAVLink and PySide6 for the "
                "UAV ground control interface with GUI."
            ),
            links=[
                Link(label="MAVLink and GUI layer", url="https://example.com/repo/gcs-gui"),
                Link(label="Cargo backend layer", url="https://example.com/repo/gcs-cargo"),
            ],
        ),
        Project(
            name="Others",
            description="You can also inspect other projects here:",
            links=[Link(label="Repos", url="https://example.com/repos")],
        ),
    ],
    skills=Skills(
        categories=[
            SkillCategory(name="Java", detail=(
                "Spring Boot and Spring Framework, JPA and JDBC for hibernation, RESTful and "
                "SOAP for API, Spring Security for AUTH (Json Web Token), Thymeleaf for "
                "Template Engine (Spring MVC, also has knowledge on JSP), Maven for "
                "dependency, HATEOAS, JUnit-Mockito-Rest Assured for Unit-Integration tests, "
                "CORS-AJAX configurations. Also has a very solid understanding of "
                "authentication-authorization flows, Spring Security Context, Spring "
                "Application Context and more.")),
            SkillCategory(name="SQL", detail=(
                "MSSQL, PostgreSQL, MySQL, SQLite3, H2 In-Memory Database. Also knows Query "
                "Method structure in JPA, Native SQL, Java Persistence Query Language(JPQL).")),
            SkillCategory(name="Swagger Interactive Documentation"),
            SkillCategory(name="Postman"),
            SkillCategory(name="Redis", detail="For caching."),
            SkillCategory(name="Python", detail="Flask Framework, Pymavlink, PySide6, FastAPI, FastHTML and more."),
            SkillCategory(name="AWS", detail="EC2 Linux Server, S3 Bucket, Elastic Beanstalk, SES, MariaDB (MySQL equivalent in AWS), RDS."),
            SkillCategory(name="Docker"),
            SkillCategory(name="Git"),
            SkillCategory(name="Apache Tomcat 11", detail="For deploying the app."),
            SkillCategory(name="C++", detail="Data Structures and Algorithms, Omnet++."),
            SkillCategory(name="C"),
            SkillCategory(name="Assembly"),
            SkillCategory(name="C# MVC", detail="(Low experience)"),
            SkillCategory(name="Linux", detail="Ubuntu, WSL."),
            SkillCategory(name="General", emphasis=True, detail=(
                "Has the ability to read code well. Has data structures and algorithms "
                "knowledge. Can write layered clean code. Has problem solving skills. Very "
                "good at Mathematics.")),
        ],
    ),
    languages=[
        Language(name="English", level="C1"),
        Language(name="Spanish", level="Native"),
    ],
    references=[
        Reference(name="Alex Thompson", detail="Software Engineer III, TechCorp"),
        Reference(name="Chris Palmer", detail="Senior Software Developer, Logix Logistics"),
        Reference(name="Pat Morgan", detail="Senior Software Engineer at SkyTech Aviation"),
        Reference(name="Jamie Lee", detail="Software Engineer at DevHouse"),
    ],
    references_note="NOTE: Contact information will be shared upon request.",
)


SAMPLES = {
    "resume.tex.j2": style_a_cv,
    "resume2.tex.j2": style_b_cv,
}

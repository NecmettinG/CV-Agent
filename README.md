# CV-Agent

Turn any CV — a PDF, a DOCX, or pasted text — into a clean, ATS-friendly, typeset
PDF, then **score it against a job description and improve it — without ever
fabricating anything.**

CV-Agent reads a messy real-world CV, understands it with an LLM, re-renders it in
one consistent professional layout, and can measure and raise how well it matches a
specific job. Every step that could invent a fact is fenced off by deterministic
guards and human confirmation, so the output stays truthful.

```
  input CV                                                     ATS report
 (PDF/DOCX/txt)                                              (0–100 + advice)
      │                                                            ▲
      ▼            ┌──────────────┐     validated CV              │
   parse  ───────▶ │  LLM extract │ ───▶ (typed, Pydantic) ──┬──▶ score / analyse
 (text +           └──────────────┘                          │
  links)                                                      ▼
                                                      render (LaTeX → Tectonic)
                                                             │
                                                             ▼
                                                     ATS-friendly PDF
```

---

## Why it exists

Most CV tools either (a) reformat blindly, mangling structure that an Applicant
Tracking System (ATS) then can’t read, or (b) “optimize” your CV by quietly
inventing skills to hit keywords. CV-Agent does neither:

- **It re-renders into a layout that machines can actually parse** — and *proves*
  it by re-reading its own output PDF and checking every field survived.
- **It never lies.** The one step that rewrites your wording is wrapped in three
  independent safeguards (below), and anything the model adds that wasn’t already
  in your CV is shown to you for a yes/no before it’s kept.

---

## Features

- **Any input → one clean output.** PDF (via `pdfplumber`), DOCX (`python-docx`),
  or `.txt`/`.md` in; a Cambridge-Oxford–style LaTeX PDF out (via Tectonic).
- **Structured extraction via forced tool-use.** The LLM fills a strict
  [Pydantic](https://docs.pydantic.dev) schema; invalid output is fed back for
  self-repair. The model’s raw text is never trusted directly.
- **Multi-provider, including free options.** Anthropic (default), OpenAI, Google
  Gemini, Moonshot/Kimi, OpenRouter, Groq — one flag to switch.
- **Dynamic sections.** Any heading a CV happens to have (Projects, Certificates,
  Community, References, Interests, Referanslar…) is preserved under its own name,
  in its own language. English and Turkish headings/dates are localized.
- **ATS scoring.** A transparent 0–100 report: a round-trip parse check + best
  practices, plus job-description keyword coverage when you supply a posting.
- **Guarded improvement.** An opt-in rewrite that can only *surface skills you
  already have*, plus an honest gap-closer that lets *you* add genuine skills the
  CV forgot to mention.
- **Robust rendering.** LaTeX-special characters and URLs are escaped/handled
  correctly; a failed compile retries once with a font-free fallback and reports a
  single clean error instead of a LaTeX traceback.
- **No secrets on disk.** API keys are passed as parameters or read from env vars,
  otherwise prompted (hidden). Real input CVs stay out of git.

---

## The faithfulness guarantee (the core idea)

Raising an ATS score is one keystroke away from lying on a CV, so honesty is
enforced *structurally*, not by trusting the model:

1. **Grafting.** The rewrite keeps your original skeleton — name, employers, job
   titles, dates, education, tech lists — **verbatim**. Only reworded prose
   (summary, bullets, skills lines) is taken from the model. It is *structurally
   impossible* for a rewrite to change a job, title, or date.
2. **Remove-fabricated gate.** Any job keyword the rewrite introduces that wasn’t
   already in your CV is flagged and confirmed with you (`y`/`N`). Reject one and
   its wording is reverted cleanly to your original.
3. **Declare-and-attach gap-closer.** For skills the job wants that your CV lacks,
   *you* affirm the ones you genuinely have; they’re added to your Skills section
   and, optionally, onto a specific role/project’s tech line. The tool never
   asserts a skill on its own — you are the source of truth, it only records it.

---

## Architecture

CV-Agent is an **agentic workflow**: bounded LLM steps (extraction, keyword
extraction, guarded rewrite) orchestrated by deterministic Python, with the
fabrication-sensitive logic living in code and human gates rather than in the model.

| Module | Responsibility |
|---|---|
| `cv_agent/parsers/` | `pdf.py` (pdfplumber; keeps hyperlinks as `anchor <url>`, de-interleaves two-column skill grids) and `docx.py`. |
| `cv_agent/schema.py` | The `CV` Pydantic model — the single source of truth. Typed header + experience + education, plus dynamic `Section`s. URL/label normalization. |
| `cv_agent/providers.py` | One tool-use interface over Anthropic + any OpenAI-compatible endpoint. Named presets wire up base URL, default model, and env var. |
| `cv_agent/extract.py` | Text → validated `CV` via forced tool-use, a provider-agnostic repair loop, and a URL-fabrication guard. |
| `cv_agent/templating.py` + `templates/resume.tex.j2` | LaTeX-safe Jinja2 (uses `\VAR`/`\BLOCK`) → LaTeX. Escaping, Turkish-aware uppercasing, date/localization. |
| `cv_agent/render.py` | LaTeX → PDF via Tectonic. Clean error distillation + font-free fallback retry. |
| `cv_agent/ats.py` | Round-trip parse check, keyword coverage, 0–100 scoring, guarded rewrite, the two gates, and declared-skill weaving. |
| `cv_agent/pipeline.py` | One-call glue: `parse_file` → `cv_from_file` → `file_to_pdf`. |

Every module has an offline smoke test in its `__main__` (see [Development](#development)).

---

## Installation

**Requirements:** Python 3.10+ and [Tectonic](https://tectonic-typesetting.github.io)
(the LaTeX engine that produces the PDF).

```bash
# 1. Clone and create a virtual environment
git clone <your-repo-url> CV-Agent
cd CV-Agent
python -m venv .venv
# Windows:  .venv\Scripts\activate       (or call .venv\Scripts\python.exe directly)
# macOS/Linux:  source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt
```

**Tectonic** is located at runtime in this order: an explicit `tectonic_path`
argument → the `TECTONIC_PATH` env var → your system `PATH` → a binary placed at
`tools/tectonic.exe` (or `tools/tectonic`). Install it however you like and put it
on `PATH`, or drop the binary in `tools/`. The first compile downloads Tectonic’s
support bundle (fonts/packages) over the network; later runs are offline.

> No LaTeX distribution (TeX Live/MiKTeX) is required — Tectonic is self-contained.

---

## Quick start

The `examples/` scripts drive the whole pipeline from the command line. API keys are
read from the provider’s env var if set, otherwise prompted (input hidden). Nothing
is written except the output PDF(s).

**Extract a CV and render it to PDF:**

```bash
python examples/extract_demo.py "my_cv.pdf" --render
python examples/extract_demo.py "my_cv.docx" --provider gemini --render   # free tier
python examples/extract_demo.py --list-providers
```

**Score a CV for ATS-friendliness (and fit to a job):**

```bash
# format + round-trip score only (no job description needed):
python examples/ats_demo.py "my_cv.pdf" --provider anthropic --model claude-haiku-4-5-20251001

# add a job description for keyword coverage:
python examples/ats_demo.py "my_cv.pdf" --jd job.txt --provider anthropic

# guarded improvement with the confirm/declare gates:
python examples/ats_demo.py "my_cv.pdf" --jd job.txt --provider anthropic --improve

# try it with the built-in fictional sample (no LLM needed for the CV):
python examples/ats_demo.py --sample
```

---

## Python API

```python
from cv_agent import file_to_pdf, cv_from_file, ats_report, improve_cv, extract_job_keywords

# One call: file → LLM extract → PDF
pdf_path = file_to_pdf("my_cv.pdf", provider="anthropic", model="claude-haiku-4-5-20251001")

# Or step by step, then score against a job description
cv = cv_from_file("my_cv.pdf", provider="anthropic")
keywords = extract_job_keywords(open("job.txt").read(), provider="anthropic")
report = ats_report(cv, pdf_path=pdf_path, keywords=keywords)
print(report.summary_text())          # full score card

# Guarded rewrite (structure is locked; only prose is reworded)
better = improve_cv(cv, keywords, provider="anthropic")
```

Lower-level entry points are also exported: `parse_file`, `extract_cv`,
`render_cv`, `render_pdf`, `roundtrip_parse`, `keyword_coverage`,
`newly_surfaced_keywords`, `apply_keyword_decisions`, `add_declared_skills`,
`weavable_entries`, `weave_skills`.

---

## Providers

Pick with `--provider` / `provider=`, and override the model with `--model` /
`model=`. Keys come from the listed env var, or are prompted. (Model IDs and free
tiers change — run `--list-providers` for the current defaults.)

| Provider | Env var | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Default. Most reliable tool-use. |
| `openai` | `OPENAI_API_KEY` | Paid. |
| `gemini` (`google`) | `GEMINI_API_KEY` | Free tier via Google AI Studio. |
| `openrouter` | `OPENROUTER_API_KEY` | One key, many models incl. `:free` ones. |
| `groq` | `GROQ_API_KEY` | Free tier, very fast. |
| `moonshot` (`kimi`) | `MOONSHOT_API_KEY` | Low-cost Kimi; strong tool-use. |

---

## How the ATS score is computed

The rubric is deliberately simple and transparent (all weights are constants in
`cv_agent/ats.py`):

- **Format score** — works with no job description:
  - **Round-trip parse** (60%): render the CV, re-read that PDF with `pdfplumber`
    (the plain-text layer an ATS actually sees), and check the name, contact, every
    employer/school, and every heading survived. Misses point to a template bug.
  - **Best practices** (40%): contact completeness, has skills/experience/education,
    quantified achievements, sane length.
- **Keyword score** — when a job description is supplied: the LLM extracts the
  posting’s decisive keywords (tagged *required* / *preferred*), then a
  deterministic matcher checks coverage (handles `C++`, `Node.js`, plurals, word
  boundaries). Required keywords weigh 2× preferred.
- **Overall** = format only, or `0.4 × format + 0.6 × keyword` when a job
  description is present (keywords dominate real ATS ranking).

---

## Development

Each module ships an offline smoke test (no network, no API key) in its `__main__`:

```bash
python -m cv_agent.schema      # model + JSON round-trip
python -m cv_agent.extract     # repair loop + URL guard (fake client)
python -m cv_agent.render      # error distiller + fallback compile (needs Tectonic)
python -m cv_agent.ats         # matcher, coverage, grafting, gates, weaving
```

Project layout:

```
cv_agent/            # the package
  parsers/           #   pdf.py, docx.py
  templates/         #   resume.tex.j2  (the one output layout)
  schema.py providers.py extract.py templating.py render.py ats.py pipeline.py
examples/            # runnable CLI demos + a fictional sample CV
requirements.txt
```

---

## Notes & limitations

- **Scanned / image-only PDFs need OCR**, which the parsers do not do — a PDF with
  no text layer yields no text.
- **The improved-prose honesty** (beyond the locked skeleton) rests on the system
  prompt plus the confirm gate: review reworded bullets before using the output.
- **ATS scoring is a heuristic** — there is no universal ATS formula; the rubric
  here is transparent and every sub-score is reported so you can see exactly why.
- Committed sample/fixture data is fully fictional. Real CVs (`test inputs/`,
  `samples/`) and generated `output/` are gitignored.

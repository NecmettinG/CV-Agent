"""Phase-4 demo: score a CV for ATS-friendliness (and fit to a job description).

Three things happen here:

  1. Render the CV to a PDF, then RE-PARSE that PDF and check the name, contact,
     every employer/school and every heading survived - proof the template is
     machine-readable (a real ATS reads the text layer, just like this).
  2. Score format + best practices, and - if you pass a job description - the
     coverage of that job's keywords (extracted by the LLM).
  3. Optionally (--improve) rephrase the CV toward the job's keywords WITHOUT
     fabricating (a tripwire rejects any rewrite that changes an employer/title/
     school/date), then re-score to show the before/after.

Run it (the built-in fictional sample needs no API key unless you add a JD)::

    python examples/ats_demo.py --sample
    python examples/ats_demo.py --sample --jd job.txt --provider gemini
    python examples/ats_demo.py my_cv.pdf --jd job.txt --provider anthropic --improve
    python examples/ats_demo.py --list-providers

A real input file (.pdf/.docx/.txt) is parsed + extracted by the LLM first (needs
a key); --sample skips that. The key comes from the provider's env var if set,
else you are prompted (hidden). Nothing is written except the PDF(s).
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cv_agent.ats import ats_report, extract_job_keywords, improve_cv
from cv_agent.pipeline import SUPPORTED_SUFFIXES, parse_file
from cv_agent.providers import PRESETS, resolve_provider_name
from cv_agent.render import render_pdf


def print_providers() -> None:
    print("Available providers (pick with --provider, override model with --model):\n")
    for name, preset in PRESETS.items():
        default = " default" if name == "anthropic" else ""
        print(f"  {name:<11}{default:<8} model: {preset.default_model}")
        print(f"  {'':<11}        key:   {preset.env_var}")
        print(f"  {'':<11}        {preset.cost}\n")


def _api_key(name: str, env_var: str) -> str:
    """The key from the provider's env var, else prompted (hidden). Never stored."""
    key = os.environ.get(env_var) or getpass.getpass(
        f"API key for {name} ({env_var}, hidden): "
    ).strip()
    if not key:
        raise SystemExit("No API key provided.")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a CV for ATS-friendliness and (optionally) fit to a job.",
        epilog="Supported inputs: " + ", ".join(sorted(SUPPORTED_SUFFIXES)),
    )
    parser.add_argument("path", type=Path, nargs="?",
                        help="CV file (.pdf/.docx/.txt/.md). Omit and use --sample for a demo.")
    parser.add_argument("--sample", action="store_true",
                        help="score the built-in fictional sample CV (no LLM needed for the CV)")
    parser.add_argument("--jd", type=Path, default=None,
                        help="job-description text file -> keyword coverage score")
    parser.add_argument("--improve", action="store_true",
                        help="also rephrase the CV toward the JD keywords (guarded; needs --jd)")
    parser.add_argument("--provider", default="anthropic",
                        help="LLM provider: " + ", ".join(PRESETS) + " (aliases: kimi, google)")
    parser.add_argument("--model", default=None, help="model id (defaults to the provider's default)")
    parser.add_argument("--list-providers", action="store_true", help="list providers and exit")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="where PDFs go")
    parser.add_argument("--x-tolerance", type=float, default=None,
                        help="PDF only: word-gap tolerance; lower (e.g. 1.5) if words merge")
    parser.add_argument("--password", default=None, help="PDF only: password for an encrypted file")
    args = parser.parse_args()

    if args.list_providers:
        print_providers()
        return
    if not args.sample and args.path is None:
        raise SystemExit("Provide a CV file path or --sample. See --help.")
    if args.improve and not args.jd:
        raise SystemExit("--improve needs a --jd (it rewrites toward the job's keywords).")

    name = resolve_provider_name(args.provider)
    preset = PRESETS.get(name)
    if preset is None:
        raise SystemExit(f"Unknown provider {args.provider!r}. Try --list-providers.")
    model = args.model or preset.default_model

    # 1. Get the CV object -----------------------------------------------------
    if args.sample:
        from examples.sample_data import sample_cv
        cv = sample_cv
        print(f"[1] Using the built-in sample CV: {cv.name}.")
    else:
        if not args.path.exists():
            raise SystemExit(f"No such file: {args.path}")
        text = parse_file(args.path, pdf_x_tolerance=args.x_tolerance, password=args.password)
        if not text.strip():
            raise SystemExit("No text extracted - empty or image-only PDF? (Scanned CVs need OCR.)")
        from cv_agent.extract import extract_cv
        key = _api_key(name, preset.env_var)
        print(f"[1] Extracting {args.path.name} with {name} / {model} ...")
        cv = extract_cv(text, provider=name, model=model, api_key=key)
        print(f"    -> {cv.name}: {len(cv.experience)} experience, {len(cv.education)} education, "
              f"{len(cv.sections)} section(s).")

    # 2. Render -> PDF (needed for the round-trip parse) -----------------------
    pdf = render_pdf(cv, output_dir=args.output_dir)
    print(f"[2] Rendered -> {pdf}")

    # 3. Keywords from the JD (optional) ---------------------------------------
    keywords = None
    if args.jd:
        if not args.jd.exists():
            raise SystemExit(f"No such job-description file: {args.jd}")
        jd_text = args.jd.read_text(encoding="utf-8")
        key = _api_key(name, preset.env_var)
        print(f"[3] Extracting job keywords with {name} / {model} ...")
        keywords = extract_job_keywords(jd_text, provider=name, model=model, api_key=key)
        print(f"    -> {len(keywords)} keywords "
              f"({sum(k.importance.startswith('req') for k in keywords)} required).")

    # 4. Score + report --------------------------------------------------------
    report = ats_report(cv, pdf_path=pdf, keywords=keywords, x_tolerance=args.x_tolerance)
    print()
    print(report.summary_text())

    # 5. Guarded improvement (optional) ----------------------------------------
    if args.improve and keywords:
        key = _api_key(name, preset.env_var)
        print(f"\n[5] Rewriting toward the JD keywords with {name} / {model} (guarded) ...")
        improved = improve_cv(cv, keywords, provider=name, model=model, api_key=key)
        imp_pdf = render_pdf(improved, output_dir=args.output_dir,
                             basename=(pdf.stem + "-improved"))
        after = ats_report(improved, pdf_path=imp_pdf, keywords=keywords,
                           x_tolerance=args.x_tolerance)
        print(f"    Rendered improved CV -> {imp_pdf}")
        print(f"    Keyword match: {report.keyword_score} -> {after.keyword_score}  "
              f"(overall {report.overall} -> {after.overall})")


if __name__ == "__main__":
    main()

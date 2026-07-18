"""End-to-end Phase-2 demo: a CV file -> parsed text -> LLM -> validated CV -> PDF.

Runs the three stages explicitly (parse / extract / render) so you can see each
one. For the one-call version, use ``cv_agent.pipeline.file_to_pdf`` instead.

Run it yourself (needs network + a real key; the key is requested on screen and
never stored)::

    python examples/extract_demo.py cv.pdf  --render                     # default (Anthropic)
    python examples/extract_demo.py cv.docx --provider gemini --render   # free: Google AI Studio
    python examples/extract_demo.py cv.pdf  --provider openrouter --model moonshotai/kimi-k2:free
    python examples/extract_demo.py --list-providers

Supported inputs: .docx, .pdf, .txt/.md. For a PDF whose words run together in
the parse step, pass a smaller --x-tolerance (e.g. 1.5). The API key comes from
the provider's env var if set, otherwise you are prompted (input hidden).
Nothing is written to disk except the optional PDF.
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cv_agent.extract import extract_cv
from cv_agent.pipeline import SUPPORTED_SUFFIXES, parse_file
from cv_agent.providers import PRESETS, resolve_provider_name
from cv_agent.render import render_pdf
from cv_agent.templating import STYLE_A


def print_providers() -> None:
    print("Available providers (pick with --provider, override model with --model):\n")
    for name, preset in PRESETS.items():
        default = " default" if name == "anthropic" else ""
        print(f"  {name:<11}{default:<8} model: {preset.default_model}")
        print(f"  {'':<11}        key:   {preset.env_var}")
        print(f"  {'':<11}        {preset.cost}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a CV file into a validated CV (and optionally a PDF).",
        epilog="Supported inputs: " + ", ".join(sorted(SUPPORTED_SUFFIXES)),
    )
    parser.add_argument("path", type=Path, nargs="?", help="CV file (.docx / .pdf / .txt / .md)")
    parser.add_argument("--provider", default="anthropic",
                        help="LLM provider: " + ", ".join(PRESETS) + " (aliases: kimi, google)")
    parser.add_argument("--model", default=None, help="model id (defaults to the provider's default)")
    parser.add_argument("--list-providers", action="store_true", help="list providers and exit")
    parser.add_argument("--render", action="store_true", help="also compile a PDF via Tectonic")
    parser.add_argument("--x-tolerance", type=float, default=None,
                        help="PDF only: word-gap tolerance; lower (e.g. 1.5) if words merge")
    parser.add_argument("--password", default=None, help="PDF only: password for an encrypted file")
    args = parser.parse_args()

    if args.list_providers:
        print_providers()
        return
    if args.path is None:
        raise SystemExit("Provide a CV file path (or --list-providers). See --help.")
    if not args.path.exists():
        raise SystemExit(f"No such file: {args.path}")

    name = resolve_provider_name(args.provider)
    preset = PRESETS.get(name)
    if preset is None:
        raise SystemExit(f"Unknown provider {args.provider!r}. Try --list-providers.")

    # 1. Parse: any supported format -> text (links preserved as `anchor <url>`).
    text = parse_file(args.path, pdf_x_tolerance=args.x_tolerance, password=args.password)
    print(f"[1/3] Parsed {args.path.name} ({args.path.suffix.lower() or 'no-ext'}): {len(text)} chars.")
    if not text.strip():
        raise SystemExit("No text extracted - empty or image-only PDF? (Scanned CVs need OCR.)")

    # 2. Extract: text -> validated CV via forced tool use.
    #    Key from the provider's env var, else prompted (hidden) and passed as a param.
    api_key = os.environ.get(preset.env_var) or getpass.getpass(
        f"API key for {name} ({preset.env_var}, hidden): "
    ).strip()
    if not api_key:
        raise SystemExit("No API key provided.")
    model = args.model or preset.default_model
    print(f"[2/3] Extracting with {name} / {model} (forced tool use + validation)...")
    cv = extract_cv(text, provider=name, model=model, api_key=api_key)
    print(f"      -> {cv.name}: {len(cv.experience)} experience, {len(cv.education)} education, "
          f"{len(cv.sections)} other section(s): {', '.join(s.title for s in cv.sections) or '-'}.")

    # 3. Render (optional): CV -> PDF.
    if args.render:
        pdf = render_pdf(cv, template_name=STYLE_A)
        print(f"[3/3] Rendered -> {pdf}")
    else:
        print("[3/3] Skipped render (pass --render to compile a PDF).")


if __name__ == "__main__":
    main()

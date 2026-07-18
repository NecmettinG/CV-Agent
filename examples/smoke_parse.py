"""Parse-only smoke test: read every CV in a folder to text. No API key, no network.

This isolates the *parsing* stage (the part most likely to break on real-world
PDFs: multi-column layouts, embedded fonts, non-ASCII names) from the LLM stage.
Run it first; if the text below looks right, the extraction step has a fair shot.

    python examples/smoke_parse.py                 # defaults to "test inputs"
    python examples/smoke_parse.py "test inputs"   # explicit folder
    python examples/smoke_parse.py path\to\one.pdf # a single file

For a PDF whose words run together, note it and we can lower the x-tolerance.
Nothing is written to disk and nothing leaves your machine.
"""

import sys
from pathlib import Path

# Real CVs carry non-ASCII names (Turkish ç/ğ/ı/ş/ü ...). The Windows console is
# cp1252 by default and would crash on them; force UTF-8 output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cv_agent.pipeline import SUPPORTED_SUFFIXES, parse_file

PREVIEW_CHARS = 400


def collect(target: Path):
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.iterdir()
                      if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    return []


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "test inputs"
    target = Path(arg)
    if not target.is_absolute():
        target = ROOT / target

    files = collect(target)
    if not files:
        raise SystemExit(f"No supported files ({', '.join(sorted(SUPPORTED_SUFFIXES))}) under {target}")

    print(f"Parsing {len(files)} file(s) from {target}\n" + "=" * 70)
    ok = 0
    for f in files:
        print(f"\n### {f.name}")
        try:
            text = parse_file(f)
        except Exception as exc:  # noqa: BLE001 - we want to see every failure
            print(f"  !! FAILED: {type(exc).__name__}: {exc}")
            continue
        stripped = text.strip()
        links = text.count(" <http")  # inline `anchor <url>` annotations we captured
        print(f"  chars: {len(text):>6} | non-empty: {bool(stripped)} | links captured: {links}")
        if not stripped:
            print("  !! EMPTY — image-only/scanned PDF? (would need OCR)")
            continue
        preview = " ".join(stripped[:PREVIEW_CHARS].split())
        print(f"  preview: {preview}...")
        ok += 1

    print("\n" + "=" * 70)
    print(f"Parsed cleanly: {ok}/{len(files)}. "
          "Spot-check the previews — right name, sane order, no gibberish, links present?")


if __name__ == "__main__":
    main()

"""Render the sample CV through the template -> LaTeX in output/.

Run from the repo root:
    .venv/Scripts/python.exe examples/render_samples.py

Then compile with Tectonic, e.g.:
    tools/tectonic.exe -X compile output/resume_gen.tex --outdir output
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make 'cv_agent' importable when run as a script

from cv_agent.templating import render_cv  # noqa: E402
from sample_data import SAMPLES  # noqa: E402  (examples/ is on sys.path[0])

OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

for template_name, cv in SAMPLES.items():
    tex = render_cv(cv, template_name)
    out_name = template_name.replace(".tex.j2", "_gen.tex")  # resume.tex.j2 -> resume_gen.tex
    (OUT / out_name).write_text(tex, encoding="utf-8")
    print(f"Rendered {template_name:16s} -> output/{out_name} ({len(tex)} chars)")

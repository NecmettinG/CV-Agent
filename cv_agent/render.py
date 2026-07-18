"""End-to-end rendering: a :class:`~cv_agent.schema.CV` object -> compiled PDF.

This is the high-level entry point. It orchestrates the full pipeline:

    CV object
      -> cv_agent.templating.render_cv(...)   (Jinja2 template -> LaTeX string)
      -> write <output_dir>/<basename>.tex
      -> subprocess: `tectonic -X compile <tex> --outdir <output_dir>`
      -> return Path to <output_dir>/<basename>.pdf

(The lower-level Jinja machinery - delimiters, escaping, date filters - lives in
:mod:`cv_agent.templating`. This module only adds the "write + compile" steps.)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Optional, Union

from cv_agent.templating import STYLE_A, render_cv
from cv_agent.schema import CV

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
_BUNDLED_TECTONIC = _REPO_ROOT / "tools" / ("tectonic.exe" if os.name == "nt" else "tectonic")


class TectonicError(RuntimeError):
    """Raised when the Tectonic compile step fails or produces no PDF."""


def find_tectonic(explicit: Optional[str] = None) -> str:
    """Locate the Tectonic binary.

    Search order (first hit wins): the ``explicit`` argument, the
    ``TECTONIC_PATH`` env var, the system ``PATH``, then the binary bundled at
    ``tools/`` in the repo. Raises :class:`FileNotFoundError` if none exist.
    """
    for candidate in (explicit, os.environ.get("TECTONIC_PATH")):
        if candidate and Path(candidate).exists():
            return str(candidate)
    on_path = shutil.which("tectonic")
    if on_path:
        return on_path
    if _BUNDLED_TECTONIC.exists():
        return str(_BUNDLED_TECTONIC)
    raise FileNotFoundError(
        "Tectonic not found. Install it and put it on PATH, set TECTONIC_PATH, "
        f"or place the binary at {_BUNDLED_TECTONIC}."
    )


def _slug(text: str) -> str:
    """A filesystem-safe slug, e.g. 'Jordan Mercer' -> 'jordan-mercer'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "-", text)
    return text or "cv"


def render_pdf(
    cv: CV,
    template_name: str = STYLE_A,
    output_dir: Union[str, Path] = "output",
    basename: Optional[str] = None,
    tectonic_path: Optional[str] = None,
    keep_tex: bool = True,
    timeout: Optional[float] = None,
) -> Path:
    """Render ``cv`` through a template and compile it to a PDF.

    Args:
        cv: the validated CV to render.
        template_name: which template file to use (``templating.STYLE_A``, or
            any ``*.tex.j2`` in ``cv_agent/templates/``).
        output_dir: directory for the ``.tex`` and ``.pdf`` (created if needed).
        basename: output filename stem; defaults to a slug of ``cv.name``.
        tectonic_path: explicit path to the Tectonic binary (else auto-located).
        keep_tex: keep the intermediate ``.tex`` next to the PDF (default True).
        timeout: seconds before the compile is aborted (default: no limit - the
            first run may download Tectonic's support bundle over the network).

    Returns:
        Path to the generated PDF.

    Raises:
        TectonicError: if compilation fails or the PDF is not produced.
        FileNotFoundError: if the Tectonic binary cannot be located.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = basename or _slug(cv.name)
    tex_path = out_dir / f"{stem}.tex"
    pdf_path = out_dir / f"{stem}.pdf"

    # 1. Jinja2 template -> LaTeX source, then write it out.
    tex_path.write_text(render_cv(cv, template_name), encoding="utf-8")

    # 2. Compile with Tectonic.
    tectonic = find_tectonic(tectonic_path)
    proc = subprocess.run(
        [tectonic, "-X", "compile", str(tex_path), "--outdir", str(out_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise TectonicError(
            f"Tectonic failed (exit {proc.returncode}) compiling {tex_path}:\n"
            f"{proc.stderr.strip()}"
        )
    if not pdf_path.exists():
        raise TectonicError(
            f"Tectonic reported success but {pdf_path} was not produced."
        )

    # 3. Optionally drop the intermediate .tex.
    if not keep_tex:
        tex_path.unlink(missing_ok=True)

    return pdf_path

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


# Tectonic stderr noise that carries no diagnostic value for the user.
_STDERR_NOISE = (
    "halted on potentially-recoverable error",
    "fontconfig error",
    "the compilation failed",
)


def _extract_latex_error(log_text: str, stderr: str) -> str:
    """Distill a failed compile into one short, human-readable reason.

    Prefers the LaTeX log's first ``! ...`` block (the message plus the ``l.NN``
    source-line context); falls back to Tectonic's own ``error:`` lines. Always
    returns a trimmed single string - never the whole multi-hundred-line dump, and
    never a raw traceback.
    """
    if log_text:
        lines = log_text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("!"):
                parts = [line.lstrip("! ").rstrip()]
                for follow in lines[i + 1 : i + 8]:
                    if follow.startswith("l."):  # the offending source line
                        parts.append(follow.strip())
                        break
                    if not follow.strip():
                        break
                    parts.append(follow.strip())
                distilled = " ".join(p for p in parts if p).strip()
                if distilled:
                    return distilled[:500]

    picks = []
    for raw in stderr.splitlines():
        s = raw.strip()
        low = s.lower()
        if low.startswith("error:") and not any(n in low for n in _STDERR_NOISE):
            picks.append(s[len("error:"):].strip())
    if picks:
        return " | ".join(picks)[:500]
    return (stderr.strip() or "unknown LaTeX error")[:500]


def _run_tectonic(tectonic: str, tex_path: Path, out_dir: Path,
                  timeout: Optional[float]) -> "subprocess.CompletedProcess[str]":
    """Run one Tectonic compile of ``tex_path`` into ``out_dir`` (keeping the log
    so :func:`_extract_latex_error` can read it) and return the completed process."""
    return subprocess.run(
        [tectonic, "-X", "compile", str(tex_path), "--outdir", str(out_dir),
         "--keep-logs"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


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
    log_path = out_dir / f"{stem}.log"

    tectonic = find_tectonic(tectonic_path)

    # Try the full layout first; if it won't compile, retry once with a
    # stripped-down, font-free fallback (rescues font-resolution failures on
    # minimal environments). On total failure we raise ONE distilled reason -
    # never a raw LaTeX traceback.
    reason = ""
    for fallback in (False, True):
        # 1. Jinja2 template -> LaTeX source, then write it out.
        tex_path.write_text(render_cv(cv, template_name, fallback=fallback),
                            encoding="utf-8")

        # 2. Compile with Tectonic.
        proc = _run_tectonic(tectonic, tex_path, out_dir, timeout)
        if proc.returncode == 0 and pdf_path.exists():
            # 3. Success: tidy the output dir (drop the log, and the .tex unless
            #    asked to keep it).
            log_path.unlink(missing_ok=True)
            if not keep_tex:
                tex_path.unlink(missing_ok=True)
            return pdf_path

        log_text = log_path.read_text(encoding="utf-8", errors="replace") \
            if log_path.exists() else ""
        reason = _extract_latex_error(log_text, proc.stderr)

    raise TectonicError(
        f"Could not compile {cv.name!r}'s CV to a PDF (tried the standard layout "
        f"and a font-free fallback). LaTeX reported: {reason}"
    )


if __name__ == "__main__":  # pragma: no cover
    # Smoke tests for the hardening layer. The error-distiller checks are fully
    # offline; the compile checks need the Tectonic binary and are skipped if it
    # is not installed.
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 1. The distiller turns a LaTeX log into ONE line (message + source line).
    log = "\n".join([
        "This is XeTeX ...", "(preamble noise)",
        "! Undefined control sequence.",
        "l.42 Hello \\nope",
        "               world",
        "No pages of output.",
    ])
    got = _extract_latex_error(log, "irrelevant stderr")
    assert got == "Undefined control sequence. l.42 Hello \\nope", repr(got)

    # 2. With no log it falls back to stderr's error: lines, dropping the noise
    #    (Fontconfig warnings, the generic 'halted ...' tail).
    stderr = (
        "Fontconfig error: Cannot load default config file: No such file\n"
        "error: foo.tex:3: Missing $ inserted\n"
        "error: halted on potentially-recoverable error as specified"
    )
    got2 = _extract_latex_error("", stderr)
    assert got2 == "foo.tex:3: Missing $ inserted", repr(got2)
    print("error distiller OK ->", got, "||", got2)

    # 3. End-to-end (needs Tectonic): a CV whose fields AND link URLs are packed
    #    with LaTeX-special characters must still compile to a PDF.
    try:
        find_tectonic()
    except FileNotFoundError:
        print("Tectonic not found; skipping compile smoke tests.")
        raise SystemExit(0)

    import tempfile

    from cv_agent.schema import (CV, Contact, DateRange, ExperienceEntry, Link,
                                 MonthYear)

    cv = CV(
        name="AT&T Research",
        contact=Contact(
            location="Berlin 50% #hq",
            email="a_b@example.com",
            phone="+49 30 1234",
            links=[Link(label="Site 100%", url="https://ex.com/a%20b#frag_x?y_z&k=1")],
        ),
        experience=[ExperienceEntry(
            company="R&D Lab",
            title="Engineer 50% #lead",
            date_range=DateRange(start=MonthYear(year=2020, month=1), current=True),
            description=r"Used C# & F#; saved 50% via a_b{c}~d^e$f.",
        )],
    )
    tmp = Path(tempfile.mkdtemp(prefix="cvhardening-"))
    pdf = render_pdf(cv, output_dir=tmp, basename="specials", keep_tex=False)
    assert pdf.exists() and pdf.stat().st_size > 0
    assert not (tmp / "specials.log").exists(), "log should be cleaned up on success"
    assert not (tmp / "specials.tex").exists(), "tex should be dropped when keep_tex=False"
    print("specials + URL compile OK ->", pdf.name, pdf.stat().st_size, "bytes")

    # 4. The font-free fallback layout renders (without the Termes/newunicodechar
    #    block) and compiles on its own.
    tex_fb = render_cv(cv, STYLE_A, fallback=True)
    assert "setmainfont" not in tex_fb and "newunicodechar" not in tex_fb, \
        "fallback layout must drop the custom-font block"
    (tmp / "fb.tex").write_text(tex_fb, encoding="utf-8")
    proc = _run_tectonic(find_tectonic(), tmp / "fb.tex", tmp, None)
    assert proc.returncode == 0 and (tmp / "fb.pdf").exists(), proc.stderr[-400:]
    print("fallback layout compile OK ->", (tmp / "fb.pdf").stat().st_size, "bytes")
    print("OK")

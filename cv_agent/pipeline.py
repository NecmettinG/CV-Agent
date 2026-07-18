"""End-to-end pipeline: an input file -> validated CV -> compiled PDF.

One-call glue over the three stages, dispatching on file type:

    cv_agent.parsers.{docx,pdf}   file  -> text   (links kept as `anchor <url>`)
    cv_agent.extract.extract_cv   text  -> CV      (LLM forced tool use)
    cv_agent.render.render_pdf    CV    -> PDF      (Tectonic)

Entry points, lowest to highest level:

    parse_file(path)    -> str   just the text (any supported format)
    cv_from_file(path)  -> CV    text + LLM extraction
    file_to_pdf(path)   -> Path  the whole chain

Supported inputs: ``.docx`` (python-docx), ``.pdf`` (pdfplumber), and plain
``.txt`` / ``.md``. Extraction knobs (``api_key``, ``model``, ``client``,
``max_repair_attempts`` ...) flow through to :func:`cv_agent.extract.extract_cv`;
rendering knobs (``template_name``, ``output_dir`` ...) to
:func:`cv_agent.render.render_pdf`. The PDF-only ``pdf_x_tolerance`` /
``password`` knobs are ignored for other formats.

Parsers are imported lazily inside :func:`parse_file` so that merely importing
``cv_agent`` does not drag in both python-docx and pdfplumber - you only pay for
the format you actually read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from cv_agent.extract import extract_cv
from cv_agent.render import render_pdf
from cv_agent.schema import CV
from cv_agent.templating import STYLE_A

TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_SUFFIXES = {".docx", ".pdf"} | TEXT_SUFFIXES


def parse_file(
    source: Union[str, Path],
    *,
    pdf_x_tolerance: Optional[float] = None,
    password: Optional[str] = None,
) -> str:
    """Read a supported CV file into text, hyperlinks preserved as ``anchor <url>``.

    Args:
        source: path to a ``.docx`` / ``.pdf`` / ``.txt`` / ``.md`` file.
        pdf_x_tolerance: PDF only - word-gap tolerance for pdfplumber. Lower it
            (e.g. ``1.5``) if words run together; ``None`` uses the parser default.
        password: PDF only - password for an encrypted document.

    Raises:
        FileNotFoundError: if ``source`` does not exist.
        ValueError: for an unsupported file type.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".docx":
        from cv_agent.parsers.docx import extract_text as _docx_text

        return _docx_text(path)
    if suffix == ".pdf":
        from cv_agent.parsers.pdf import DEFAULT_X_TOLERANCE
        from cv_agent.parsers.pdf import extract_text as _pdf_text

        xt = DEFAULT_X_TOLERANCE if pdf_x_tolerance is None else pdf_x_tolerance
        return _pdf_text(path, x_tolerance=xt, password=password)
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")

    raise ValueError(
        f"Unsupported input {suffix or '(no extension)'!r} for {path.name}. "
        f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
    )


def cv_from_file(
    source: Union[str, Path],
    *,
    client: Any = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    pdf_x_tolerance: Optional[float] = None,
    password: Optional[str] = None,
    **extract_kwargs: Any,
) -> CV:
    """Parse any supported CV file and extract it into a validated :class:`CV`.

    Extra keyword args are forwarded to :func:`cv_agent.extract.extract_cv`
    (e.g. ``provider``, ``system_prompt``, ``max_repair_attempts``). Pick the LLM
    with ``provider=``/``model=``; ``model=None`` uses the provider's default.

    Raises:
        ValueError: if the file yields no text (empty or image-only/scanned).
    """
    text = parse_file(source, pdf_x_tolerance=pdf_x_tolerance, password=password)
    if not text.strip():
        raise ValueError(
            f"No text extracted from {Path(source).name!r} - is it empty or "
            "image-only? (Scanned PDFs need OCR, which the parsers do not do.)"
        )
    return extract_cv(text, client=client, api_key=api_key, model=model, **extract_kwargs)


def file_to_pdf(
    source: Union[str, Path],
    *,
    client: Any = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    pdf_x_tolerance: Optional[float] = None,
    password: Optional[str] = None,
    template_name: str = STYLE_A,
    output_dir: Union[str, Path] = "output",
    basename: Optional[str] = None,
    keep_tex: bool = True,
    tectonic_path: Optional[str] = None,
    timeout: Optional[float] = None,
    **extract_kwargs: Any,
) -> Path:
    """Run the full file -> text -> LLM -> CV -> PDF pipeline; return the PDF path.

    Args:
        source: input CV file (``.docx`` / ``.pdf`` / ``.txt`` / ``.md``).
        client / api_key / model / **extract_kwargs: passed to the extraction
            step (see :func:`cv_agent.extract.extract_cv`).
        pdf_x_tolerance / password: PDF-only parsing knobs (see :func:`parse_file`).
        template_name: output template (``STYLE_A``).
        output_dir, basename, keep_tex, tectonic_path, timeout: passed to the
            render step (see :func:`cv_agent.render.render_pdf`).

    Returns:
        Path to the generated PDF.
    """
    cv = cv_from_file(
        source,
        client=client,
        api_key=api_key,
        model=model,
        pdf_x_tolerance=pdf_x_tolerance,
        password=password,
        **extract_kwargs,
    )
    return render_pdf(
        cv,
        template_name=template_name,
        output_dir=output_dir,
        basename=basename,
        keep_tex=keep_tex,
        tectonic_path=tectonic_path,
        timeout=timeout,
    )

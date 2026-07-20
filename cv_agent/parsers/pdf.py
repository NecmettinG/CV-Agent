"""PDF -> plain text (with hyperlink URLs preserved).

Mirrors :mod:`cv_agent.parsers.docx` on purpose: same ``extract_text`` /
``extract_hyperlinks`` surface, same inline ``anchor <url>`` convention, so the
LLM step downstream never has to care which format a CV arrived in.

The hard part is that **PDF hyperlinks are positional, not textual**. A DOCX
stores a link as a run of text with an href attached; a PDF stores it as a
rectangle on the page plus a URL (``page.hyperlinks`` gives ``{x0, top, x1,
bottom, uri}``). There is no built-in link between that rectangle and the text
under it - we have to recover it geometrically: find the words whose centre
falls inside the link rectangle, and splice the URL in after them.

That makes **word segmentation the load-bearing step**. pdfplumber groups
characters into words by horizontal gap (``x_tolerance``); if that value is too
large for a given PDF the words merge ("SAM RIVERA" -> "SAMRIVERA"), a whole
line collapses into one word, and only one of several links on it will match.
Most real CVs (exported from Word/Google Docs) encode real spaces and the
default is fine; some (LaTeX/XeTeX exports) omit space glyphs and need a smaller
``x_tolerance``. It is exposed for exactly that reason.

A two-column *bulleted list* (typically a skills grid) is de-interleaved so each
column reads top-to-bottom (see ``merge_columns`` / :func:`_deinterleave_columns`).
Not handled: scanned/image-only PDFs (no text layer -> needs OCR), and general
multi-column *prose* layouts (only sustained two-column bulleted blocks are
recovered - a full page-layout analysis is out of scope).

Smoke test::

    python -m cv_agent.parsers.pdf <path/to/cv.pdf> [x_tolerance]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pdfplumber

# pdfplumber's own defaults; exposed because some PDFs need them tightened.
DEFAULT_X_TOLERANCE = 3.0
DEFAULT_Y_TOLERANCE = 3.0

_BLANK_RUN = re.compile(r"\n{3,}")

# A link rectangle in page coordinates: (x0, top, x1, bottom).
_Rect = Tuple[float, float, float, float]


def _link_rects_and_uris(page: Any) -> Tuple[List[_Rect], List[str]]:
    """Pull (rectangle, uri) for every external hyperlink on a page."""
    rects: List[_Rect] = []
    uris: List[str] = []
    for h in page.hyperlinks:
        uri = h.get("uri")
        if not uri:
            continue  # internal GoTo links carry no URL we can keep
        rects.append((h["x0"], h["top"], h["x1"], h["bottom"]))
        uris.append(uri)
    return rects, uris


def _word_link_index(word: Dict[str, Any], rects: List[_Rect]) -> Optional[int]:
    """Index of the first link rectangle containing the word's centre, else None."""
    cx = (word["x0"] + word["x1"]) / 2
    cy = (word["top"] + word["bottom"]) / 2
    for i, (x0, top, x1, bottom) in enumerate(rects):
        if x0 <= cx <= x1 and top <= cy <= bottom:
            return i
    return None


def _group_lines(words: List[Dict[str, Any]], y_tolerance: float) -> List[List[Dict[str, Any]]]:
    """Group words into visual lines by vertical position, left-to-right."""
    lines: List[List[Dict[str, Any]]] = []
    for w in sorted(words, key=lambda w: w["top"]):
        if lines and abs(w["top"] - lines[-1][0]["top"]) <= y_tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w["x0"])
    return lines


# --------------------------------------------------------------------------- #
# Two-column de-interleaving
# --------------------------------------------------------------------------- #
# Some CVs put a section (typically skills) in two side-by-side columns. Read
# naively, each visual line becomes "left-col-item  right-col-item", interleaving
# the two lists. We detect a sustained block with a clear, consistent vertical
# gutter and re-read it one column at a time. Deliberately conservative: a normal
# single-column line, or a "Title .......... Date" header, is never split (a
# header is one line, not a sustained block, and its right cell has no consistent
# left edge), so the working single-column parse is untouched.
_COL_MIN_GUTTER = 16.0        # a column gutter is far wider than a word space
_COL_MIN_LINES = 3            # need a sustained run to call it a two-column block
_COL_LEFT_ALIGN_TOL = 40.0    # a line jutting further left than this ends the block
_COL_RIGHT_EDGE_SPREAD = 26.0  # right-column items must start at a consistent x
# Only split when BOTH sides are bulleted lists. This is what separates a real
# two-column *list* (skills) from single-column text with a right-aligned date or
# location (education/experience), which must never be split.
_BULLET_CHARS = {"-", "–", "—", "•", "●", "*", "◦", "·", "‣", "▪", "▸", "►"}


def _mostly_bulleted(cells: List[List[Dict[str, Any]]]) -> bool:
    """True if a majority of the non-empty cells start with a bullet marker."""
    filled = [c for c in cells if c]
    if len(filled) < 2:
        return False
    hits = sum(1 for c in filled if c[0]["text"].strip() in _BULLET_CHARS)
    return hits >= 0.6 * len(filled)


def _merge_spans(spans: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping/touching x-spans into disjoint intervals (sorted)."""
    merged: List[Tuple[float, float]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1] + 0.5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _corridor_x(band: List[List[Dict[str, Any]]]) -> Optional[float]:
    """Split-x of the widest internal clear vertical corridor across ``band``
    (a gap wide enough to be a gutter, with words on both sides), else None."""
    spans = [(w["x0"], w["x1"]) for line in band for w in line]
    if not spans:
        return None
    merged = _merge_spans(spans)
    best: Optional[Tuple[float, float]] = None  # (split_x, width)
    for (_, b), (c, _) in zip(merged, merged[1:]):
        if c - b >= _COL_MIN_GUTTER and (best is None or c - b > best[1]):
            best = ((b + c) / 2, c - b)
    return best[0] if best else None


def _deinterleave_columns(
    lines: List[List[Dict[str, Any]]]
) -> List[List[Dict[str, Any]]]:
    """Reorder two-column list blocks so each column reads top-to-bottom."""
    out: List[List[Dict[str, Any]]] = []
    i, n = 0, len(lines)
    while i < n:
        left_edge = min(w["x0"] for w in lines[i])
        # Grow a candidate block while a common corridor survives and lines align.
        j = i + 1
        while j < n:
            if min(w["x0"] for w in lines[j]) < left_edge - _COL_LEFT_ALIGN_TOL:
                break
            if _corridor_x(lines[i:j + 1]) is None:
                break
            j += 1
        block = lines[i:j]
        split = _corridor_x(block)
        handled = False
        if split is not None and len(block) >= _COL_MIN_LINES:
            left = [[w for w in line if w["x1"] <= split] for line in block]
            right = [[w for w in line if w["x0"] >= split] for line in block]
            right_edges = [min(w["x0"] for w in r) for r in right if r]
            if (sum(bool(l) for l in left) >= 2 and len(right_edges) >= 2
                    and max(right_edges) - min(right_edges) <= _COL_RIGHT_EDGE_SPREAD
                    and _mostly_bulleted(left) and _mostly_bulleted(right)):
                out.extend(l for l in left if l)
                out.extend(r for r in right if r)
                handled = True
        if handled:
            i = j
        else:
            out.append(lines[i])
            i += 1
    return out


def _render_line(line_words: List[Dict[str, Any]], uris: List[str], annotate: bool) -> str:
    """Join a line's words, appending ``<url>`` after each linked word-run."""
    out: List[str] = []
    i, n = 0, len(line_words)
    while i < n:
        link = line_words[i].get("_link") if annotate else None
        if link is not None:
            # Consume the whole run of adjacent words sharing this link.
            run: List[str] = []
            while i < n and line_words[i].get("_link") == link:
                run.append(line_words[i]["text"])
                i += 1
            out.append(" ".join(run) + f" <{uris[link]}>")
        else:
            out.append(line_words[i]["text"])
            i += 1
    return " ".join(out)


def _open(source: Union[str, Path], password: Optional[str]):
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    try:
        return pdfplumber.open(str(path), password=password or "")
    except Exception as exc:  # pdfplumber/pdfminer raise a variety of types
        raise ValueError(f"Could not open {path} as a PDF: {exc}") from exc


def extract_text(
    source: Union[str, Path],
    *,
    annotate_links: bool = True,
    x_tolerance: float = DEFAULT_X_TOLERANCE,
    y_tolerance: float = DEFAULT_Y_TOLERANCE,
    password: Optional[str] = None,
    merge_columns: bool = True,
) -> str:
    """Extract text from a PDF, with hyperlink URLs spliced in as ``anchor <url>``.

    Args:
        source: path to the PDF.
        annotate_links: splice link URLs inline (default True). When False, the
            fast path returns pdfplumber's own ``extract_text`` per page (best
            text fidelity, no link URLs).
        x_tolerance: max horizontal gap (points) for characters to count as one
            word. Lower it if words are running together (some XeTeX PDFs); the
            quality of inline links depends directly on this.
        y_tolerance: vertical tolerance for grouping words into a line.
        password: password for an encrypted PDF.
        merge_columns: de-interleave a two-column *bulleted list* (e.g. a
            two-column skills grid) so each column reads top-to-bottom instead of
            zig-zagging line-by-line. Conservative - only fires on a sustained
            block with a clear gutter where both sides are bulleted, so ordinary
            single-column text and title/date headers are left as-is. Default True.

    Returns:
        The document text, one line per visual line, blank runs collapsed.

    Raises:
        FileNotFoundError: if ``source`` does not exist.
        ValueError: if ``source`` cannot be opened as a PDF.
    """
    out_lines: List[str] = []
    with _open(source, password) as pdf:
        for page in pdf.pages:
            if not annotate_links:
                txt = page.extract_text(x_tolerance=x_tolerance, y_tolerance=y_tolerance) or ""
                if txt:
                    out_lines.append(txt)
                continue

            rects, uris = _link_rects_and_uris(page)
            words = page.extract_words(x_tolerance=x_tolerance, y_tolerance=y_tolerance)
            for w in words:
                w["_link"] = _word_link_index(w, rects) if rects else None
            lines = _group_lines(words, y_tolerance)
            if merge_columns:
                lines = _deinterleave_columns(lines)
            for line_words in lines:
                out_lines.append(_render_line(line_words, uris, annotate_links))

    text = "\n".join(out_lines)
    return _BLANK_RUN.sub("\n\n", text).strip()


def _page_links(page: Any, x_tolerance: float, y_tolerance: float) -> List[Tuple[str, str]]:
    """(anchor, uri) for each hyperlink on a page, anchor built from its words.

    Uses the same word-centre-in-rectangle rule as the inline text path, so the
    anchor here matches what gets spliced into :func:`extract_text` - and stays
    clean where a crop of the raw rectangle would pull in neighbouring text.
    """
    rects, uris = _link_rects_and_uris(page)
    if not rects:
        return []
    buckets: List[List[Dict[str, Any]]] = [[] for _ in rects]
    for w in page.extract_words(x_tolerance=x_tolerance, y_tolerance=y_tolerance):
        idx = _word_link_index(w, rects)
        if idx is not None:
            buckets[idx].append(w)
    out: List[Tuple[str, str]] = []
    for words, uri in zip(buckets, uris):
        words.sort(key=lambda w: (w["top"], w["x0"]))
        out.append((" ".join(w["text"] for w in words), uri))
    return out


def extract_hyperlinks(
    source: Union[str, Path],
    *,
    x_tolerance: float = DEFAULT_X_TOLERANCE,
    y_tolerance: float = DEFAULT_Y_TOLERANCE,
    password: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Return every external hyperlink as ``(anchor_text, url)``, in page order.

    The anchor is the text of the words sitting under the link's rectangle
    (empty string if none do). Internal (non-URL) links are skipped. Maps
    straight onto the schema's ``Link(label, url)``.
    """
    links: List[Tuple[str, str]] = []
    with _open(source, password) as pdf:
        for page in pdf.pages:
            links.extend(_page_links(page, x_tolerance, y_tolerance))
    return links


if __name__ == "__main__":  # pragma: no cover
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if len(sys.argv) < 2:
        print("usage: python -m cv_agent.parsers.pdf <path/to/cv.pdf> [x_tolerance]")
        raise SystemExit(2)

    path = sys.argv[1]
    xt = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_X_TOLERANCE

    text = extract_text(path, x_tolerance=xt)
    links = extract_hyperlinks(path, x_tolerance=xt)
    print(f"--- extracted {len(text)} chars, {len(links)} hyperlink(s) (x_tolerance={xt}) ---\n")
    print(text)
    if links:
        print("\n--- hyperlinks ---")
        for anchor, url in links:
            print(f"  {anchor!r} -> {url}")

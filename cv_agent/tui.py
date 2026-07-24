"""Interactive terminal UI for CV-Agent - no flags to memorize.

Launch it with either of:

    python -m cv_agent
    python cv_agent/tui.py

It wraps the same library the CLI examples use (parse -> extract -> render, ATS
scoring, and the guarded improvement gates) behind numbered menus: pick a provider
and model once, choose an action, pick a file from an auto-discovered list, and go.
API keys are read from the provider's env var if set, otherwise prompted (hidden)
and cached for the session only - never written to disk.

UX niceties: a live spinner/timer during slow LLM + render calls, a per-session
cache so re-using a CV skips re-extraction, an offer to open the finished PDF, and
friendly hints for the common failures.
"""

from __future__ import annotations

import getpass
import itertools
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cv_agent.providers import PRESETS, resolve_provider_name

# --------------------------------------------------------------------------- #
# Colour (optional; colorama is already a dependency, but degrade gracefully).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - cosmetic only
    import colorama

    colorama.init()
    _STYLES = {"h": "\033[96m", "ok": "\033[92m", "warn": "\033[93m",
               "err": "\033[91m", "dim": "\033[90m", "b": "\033[1m", "r": "\033[0m"}
except Exception:  # pragma: no cover
    _STYLES = {k: "" for k in ("h", "ok", "warn", "err", "dim", "b", "r")}


def c(text: str, style: str) -> str:
    return f"{_STYLES.get(style, '')}{text}{_STYLES['r']}"


#: Convenience model shortlists per provider so you can pick without typing a long
#: id. Model ids change over time - edit freely; "custom" is always available.
KNOWN_MODELS = {
    "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-4-8"],
}

_CV_EXTS = {".pdf", ".docx", ".txt", ".md"}
_JD_EXTS = {".txt", ".md", ".pdf", ".docx"}
_SEARCH_DIRS = ("test inputs", "samples", ".", "output")
#: Repo files that share a CV-ish extension but are obviously not CVs/JDs.
_IGNORE_NAMES = {"readme.md", "requirements.txt", "license", "license.md",
                 "license.txt", "changelog.md", "contributing.md"}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    provider: str = "anthropic"
    model: Optional[str] = None          # None -> the provider preset's default (Haiku)
    api_key: Optional[str] = None        # cached for this run only
    output_dir: str = "output"
    x_tolerance: Optional[float] = None
    #: (resolved path, provider, model, x_tol) -> extracted CV, to avoid paying for
    #: the same extraction twice in one session.
    cv_cache: Dict[tuple, object] = field(default_factory=dict)

    @property
    def preset(self):
        return PRESETS[resolve_provider_name(self.provider)]

    @property
    def effective_model(self) -> str:
        return self.model or self.preset.default_model

    @property
    def model_label(self) -> str:
        return self.model or f"{self.preset.default_model} (default)"


# --------------------------------------------------------------------------- #
# Spinner: a live status line for the slow (LLM / Tectonic) calls.
# --------------------------------------------------------------------------- #
class spinner:
    """Context manager showing ``message ... |  3s`` while a blocking call runs.

    Animates only on a real terminal; when output is piped it just prints the
    message once (so logs stay clean and tests don't hang on animation).
    """

    _FRAMES = "|/-\\"

    def __init__(self, message: str):
        self.message = message
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "spinner":
        if sys.stdout.isatty():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            print(c(f"  {self.message} ...", "dim"))
        return self

    def _run(self) -> None:
        start = time.time()
        for frame in itertools.cycle(self._FRAMES):
            if self._stop.is_set():
                break
            elapsed = time.time() - start
            sys.stdout.write(f"\r  {c(frame, 'h')} {self.message} ... {elapsed:4.0f}s ")
            sys.stdout.flush()
            time.sleep(0.1)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            sys.stdout.write("\r" + " " * (len(self.message) + 24) + "\r")
            sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Tiny input helpers
# --------------------------------------------------------------------------- #
def _input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        raise SystemExit("\nBye.")


#: Bracketed-paste markers some terminals wrap pasted text in.
_PASTE_MARKS = ("\x1b[200~", "\x1b[201~")


def read_multiline(end: str = "END") -> str:
    """Read many lines of pasted/typed text until a terminator.

    Finishes on EOF (Ctrl-Z then Enter on Windows, Ctrl-D on macOS/Linux) or on the
    sentinel ``end``. The sentinel is matched robustly: on its own line (any case /
    stray spaces) AND when it arrives glued to the last line - e.g. ``…grow.END`` -
    which happens when a paste has no trailing newline. Bracketed-paste escape codes
    are stripped so they don't corrupt the text or hide the sentinel.
    """
    print(c(f"  Paste or type your text. To finish: press Ctrl-Z then Enter (Windows) "
            f"or Ctrl-D (Mac/Linux) - or type {end} on a new line.", "dim"))
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        for mark in _PASTE_MARKS:
            line = line.replace(mark, "")
        if line.strip().upper() == end.upper():         # END on its own line
            break
        rs = line.rstrip()
        if rs.upper().endswith(end.upper()):            # END glued to the last line
            i = len(rs) - len(end)
            if i == 0 or not rs[i - 1].isalnum():        # a standalone token, not "backend"
                head = rs[:i].rstrip()
                if head:
                    lines.append(head)
                break
        lines.append(line)
    return "\n".join(lines).strip()


def choose(title: str, options: List[Tuple[str, object]], *, back: str = "Back"):
    """Show a numbered menu; return the chosen option's value, or None for Back."""
    print("\n" + c(title, "h"))
    for i, (label, _) in enumerate(options, 1):
        print(f"  {c(str(i), 'b')}) {label}")
    if back:
        print(f"  {c('0', 'b')}) {back}")
    while True:
        raw = _input("Select: ")
        if raw == "0" and back:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print(c("  Please enter one of the numbers above.", "warn"))


def confirm(prompt: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    ans = _input(f"{prompt} {suffix}: ").lower()
    if not ans:
        return default
    return ans in ("y", "yes")


def discover_files(exts) -> List[Path]:
    """CV / JD files found in the usual folders, de-duplicated, sorted."""
    found: List[Path] = []
    seen = set()
    for d in _SEARCH_DIRS:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in exts and f.name.lower() not in _IGNORE_NAMES:
                rp = f.resolve()
                if rp not in seen:
                    seen.add(rp)
                    found.append(f)
    return found


def pick_file(prompt: str, exts, *, none_label: Optional[str] = None) -> Tuple[str, Optional[Path]]:
    """Pick a file from the discovered list, enter a path, or (optionally) none.

    Returns ``(kind, path)`` where kind is 'file' | 'none' | 'cancel'.
    """
    options: List[Tuple[str, object]] = [(str(f), ("file", f)) for f in discover_files(exts)]
    options.append(("Enter a path manually...", ("manual", None)))
    if none_label:
        options.append((none_label, ("none", None)))
    sel = choose(prompt, options)
    if sel is None:
        return ("cancel", None)
    kind, val = sel  # type: ignore[misc]
    if kind == "manual":
        raw = _input("Path: ").strip().strip('"')
        return ("file", Path(raw)) if raw else ("cancel", None)
    return (kind, val)


def ensure_key(s: Session) -> Optional[str]:
    """The API key for the session's provider: cache -> env var -> hidden prompt."""
    if s.api_key:
        return s.api_key
    env = os.environ.get(s.preset.env_var)
    if env:
        s.api_key = env
        return env
    try:
        key = getpass.getpass(f"  API key for {s.provider} ({s.preset.env_var}, hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not key:
        print(c("  No API key entered - cancelled.", "warn"))
        return None
    s.api_key = key
    return key


def pause() -> None:
    _input("\n  " + c("Press Enter to return to the menu...", "dim"))


def open_path(path: Path) -> None:
    """Open a file/folder in the OS default app (best-effort, cross-platform)."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:  # pragma: no cover
        print(c(f"  Could not open it automatically: {e}", "warn"))


def _offer_open(pdf: Path) -> None:
    if confirm("  Open the PDF now?", default=True):
        open_path(pdf)


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #
def _extract(s: Session, path: Path, key: str):
    """Parse + LLM-extract a CV file (cached per session), with a summary."""
    from cv_agent.pipeline import cv_from_file

    cache_key = (str(path.resolve()), s.provider, s.effective_model, s.x_tolerance)
    if cache_key in s.cv_cache:
        cv = s.cv_cache[cache_key]
        print(c(f"  Reusing the CV extracted earlier this session ({cv.name}).", "dim"))
        return cv

    with spinner(f"Reading and understanding {path.name}"):
        cv = cv_from_file(path, provider=s.provider, model=s.model, api_key=key,
                          pdf_x_tolerance=s.x_tolerance)
    s.cv_cache[cache_key] = cv
    print(f"  -> {c(cv.name, 'b')}: {len(cv.experience)} experience, "
          f"{len(cv.education)} education, {len(cv.sections)} other section(s).")
    return cv


def _read_jd(path: Path) -> str:
    from cv_agent.pipeline import parse_file

    return parse_file(path)


def _render(s: Session, cv, *, basename: Optional[str] = None) -> Path:
    from cv_agent.render import render_pdf

    with spinner("Rendering PDF (first run may download fonts)"):
        return render_pdf(cv, output_dir=s.output_dir, basename=basename)


def _get_keywords(s: Session, jd: Path, key: str):
    from cv_agent.ats import extract_job_keywords

    with spinner("Reading the job description"):
        kws = extract_job_keywords(_read_jd(jd), provider=s.provider, model=s.model, api_key=key)
    print(f"  -> {len(kws)} keywords "
          f"({sum(k.importance.startswith('req') for k in kws)} required).")
    return kws


def _load_cv_or_none(s: Session, prompt: str):
    """Pick a CV file, get the key, extract. Returns (cv, key) or (None, None)."""
    kind, path = pick_file(prompt, _CV_EXTS)
    if kind != "file":
        return None, None
    if not path.exists():
        print(c(f"  No such file: {path}", "err"))
        return None, None
    key = ensure_key(s)
    if not key:
        return None, None
    return _extract(s, path, key), key


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
_CREATE_GUIDE = (
    "  Tell me about yourself and I'll organize it into a CV. Include what you can:\n"
    "    - Your name (required), and contact: email, phone, city\n"
    "    - Each job: company, title, dates, and what you did / achieved\n"
    "    - Education: school, degree, years\n"
    "    - Skills / technologies you know\n"
    "    - Optional: projects, certificates, languages, LinkedIn/GitHub links\n"
    "  Write freely - first person is fine. I won't invent anything you don't mention."
)


def action_create(s: Session) -> None:
    """Build a CV from a free-form self-description (typed/pasted or a text file)."""
    from cv_agent.ats import ats_report
    from cv_agent.extract import build_cv

    how = choose("Create a CV from a description - how will you provide it?",
                 [("Type or paste it here", "paste"), ("Load it from a text file", "file")])
    if how is None:
        return
    if how == "paste":
        print("\n" + _CREATE_GUIDE + "\n")
        text = read_multiline()
    else:
        kind, path = pick_file("Choose a text file describing you:", {".txt", ".md"})
        if kind != "file":
            return
        if not path.exists():
            print(c(f"  No such file: {path}", "err"))
            return
        text = path.read_text(encoding="utf-8")
    if not text.strip():
        print(c("  Nothing to build from.", "warn"))
        return

    key = ensure_key(s)
    if not key:
        return
    with spinner("Building your CV from the description"):
        cv = build_cv(text, provider=s.provider, model=s.effective_model, api_key=key)
    print(f"  -> {c(cv.name, 'b')}: {len(cv.experience)} experience, "
          f"{len(cv.education)} education, {len(cv.sections)} section(s).")
    pdf = _render(s, cv)
    print(c(f"\n  Created -> {pdf}", "ok"))

    # Reuse the ATS report to flag what's thin, so they can enrich + rebuild.
    report = ats_report(cv, pdf_path=pdf)
    if report.recommendations:
        print(c("\n  Tips to strengthen it (add to your description and rebuild):", "warn"))
        for rec in report.recommendations:
            print(f"    - {rec}")
    _offer_open(pdf)


def action_convert(s: Session) -> None:
    cv, _ = _load_cv_or_none(s, "Choose a CV to convert to PDF:")
    if cv is None:
        return
    pdf = _render(s, cv)
    print(c(f"\n  Rendered -> {pdf}", "ok"))
    _offer_open(pdf)


def action_score(s: Session) -> None:
    from cv_agent.ats import ats_report

    cv, key = _load_cv_or_none(s, "Choose a CV to score:")
    if cv is None:
        return
    jkind, jd = pick_file("Add a job description for keyword scoring (optional):",
                          _JD_EXTS, none_label="No job description (format score only)")
    if jkind == "cancel":
        return

    pdf = _render(s, cv)
    keywords = _get_keywords(s, jd, key) if jkind == "file" else None
    report = ats_report(cv, pdf_path=pdf, keywords=keywords, x_tolerance=s.x_tolerance)
    print("\n" + report.summary_text())
    print(c(f"\n  PDF -> {pdf}", "ok"))
    _offer_open(pdf)

    # Seamless next step: improve this same CV/job without re-extracting anything.
    if keywords and confirm("\n  Improve this CV for this job now?"):
        _do_improve(s, cv, keywords, key, pdf, report)


def _run_gates(s: Session, cv, keywords, key):
    """The guarded improvement: rewrite + remove-fabricated gate + declare/attach.
    Returns the improved CV. All decisions are the user's; nothing is fabricated."""
    from cv_agent.ats import (add_declared_skills, apply_keyword_decisions, improve_cv,
                              keyword_coverage, newly_surfaced_keywords, weavable_entries,
                              weave_skills)

    with spinner("Rewriting toward the job (structure is locked)"):
        improved = improve_cv(cv, keywords, provider=s.provider, model=s.model, api_key=key)

    # 1. Remove-fabricated gate: confirm any keyword the rewrite introduced.
    surfaced = newly_surfaced_keywords(cv, improved, keywords)
    if surfaced:
        print(c(f"\n  The rewrite added {len(surfaced)} keyword(s) NOT in your CV.", "warn"))
        print("  Keep each ONLY if you genuinely have it.")
        rejected = [k.text for k in surfaced if not confirm(f"    Keep '{k.text}' ({k.importance})?")]
        if rejected:
            improved = apply_keyword_decisions(cv, improved, rejected)
            print(c(f"  Removed: {', '.join(rejected)}", "dim"))

    # 2. Declare-and-attach gap-closer: add genuine skills the CV still lacks.
    missing = keyword_coverage(improved, keywords).missing
    if missing:
        print(c(f"\n  {len(missing)} job keyword(s) still missing. Add ONLY the ones you genuinely have:", "warn"))
        confirmed = [k.text for k in missing if confirm(f"    Do you genuinely have '{k.text}'?")]
        if confirmed:
            improved = add_declared_skills(improved, confirmed)
            print(c(f"  Added to Skills: {', '.join(confirmed)}", "dim"))
            targets = weavable_entries(improved)
            if targets:
                print("\n  Optionally attach a skill to a specific role/project (its tech line):")
                for idx, (label, _) in enumerate(targets, 1):
                    print(f"      {c(str(idx), 'b')}) {label}")
                assignments = []
                for skill in confirmed:
                    raw = _input(f"    Attach '{skill}' to which number(s)? (comma-separated, Enter to skip): ")
                    picks = [int(p) for p in raw.replace(",", " ").split()
                             if p.isdigit() and 1 <= int(p) <= len(targets)]
                    for p in picks:
                        assignments.append((targets[p - 1][1], [skill]))
                if assignments:
                    improved = weave_skills(improved, assignments)
                    print(c(f"  Attached {len(assignments)} placement(s).", "dim"))
    return improved


def _do_improve(s: Session, cv, keywords, key, base_pdf: Path, before) -> None:
    """Run the gates, render the improved CV, and report before/after."""
    from cv_agent.ats import ats_report

    improved = _run_gates(s, cv, keywords, key)
    imp_pdf = _render(s, improved, basename=base_pdf.stem + "-improved")
    after = ats_report(improved, pdf_path=imp_pdf, keywords=keywords, x_tolerance=s.x_tolerance)
    print(c(f"\n  Improved CV -> {imp_pdf}", "ok"))
    print(f"  Keyword match: {before.keyword_score} -> {after.keyword_score}  "
          f"(overall {before.overall} -> {after.overall})")
    _offer_open(imp_pdf)


def action_improve(s: Session) -> None:
    from cv_agent.ats import ats_report

    cv, key = _load_cv_or_none(s, "Choose a CV to improve:")
    if cv is None:
        return
    jkind, jd = pick_file("Choose the job description (required for improvement):", _JD_EXTS)
    if jkind != "file":
        print(c("  Improvement needs a job description - cancelled.", "warn"))
        return

    base_pdf = _render(s, cv)
    keywords = _get_keywords(s, jd, key)
    before = ats_report(cv, pdf_path=base_pdf, keywords=keywords, x_tolerance=s.x_tolerance)
    _do_improve(s, cv, keywords, key, base_pdf, before)


def action_sample(s: Session) -> None:
    """Render + score the built-in fictional sample - fully offline, no API key."""
    from cv_agent.ats import ats_report

    try:
        from examples.sample_data import sample_cv
    except Exception:
        print(c("  Could not import the sample (run from the repo root).", "err"))
        return
    pdf = _render(s, sample_cv)
    report = ats_report(sample_cv, pdf_path=pdf, x_tolerance=s.x_tolerance)
    print("\n" + report.summary_text())
    print(c(f"\n  Sample PDF -> {pdf}", "ok"))
    _offer_open(pdf)


def action_settings(s: Session) -> None:
    while True:
        xtol = "default" if s.x_tolerance is None else s.x_tolerance
        opt = choose(
            f"Settings   (provider={s.provider}, model={s.model_label}, "
            f"output={s.output_dir}/, pdf-tol={xtol})",
            [("Change provider", "provider"),
             ("Change model", "model"),
             ("Change output folder", "output"),
             ("Change PDF word-gap tolerance (advanced)", "xtol")],
            back="Done",
        )
        if opt is None:
            return
        if opt == "provider":
            prov = choose("Choose a provider:",
                          [(f"{n:<11} - {p.cost}", n) for n, p in PRESETS.items()])
            if prov:
                s.provider = prov
                s.model = None       # reset to the new provider's default
                s.api_key = None      # a key for one provider is not valid for another
                print(c(f"  Provider set to {prov}. Model reset to its default.", "dim"))
        elif opt == "model":
            known = KNOWN_MODELS.get(resolve_provider_name(s.provider), [])
            opts: List[Tuple[str, object]] = [(m, m) for m in known]
            opts.append((f"Provider default ({s.preset.default_model})", "__default__"))
            opts.append(("Enter a custom model id...", "__custom__"))
            sel = choose("Choose a model:", opts)
            if sel == "__default__":
                s.model = None
            elif sel == "__custom__":
                raw = _input("Model id: ")
                if raw:
                    s.model = raw
            elif sel is not None:
                s.model = sel  # type: ignore[assignment]
        elif opt == "output":
            raw = _input(f"Output folder [{s.output_dir}]: ")
            if raw:
                s.output_dir = raw
        elif opt == "xtol":
            raw = _input("PDF word-gap tolerance - lower (e.g. 1.5) if words merge, "
                         "or 'default' [current shown above]: ").lower()
            if raw in ("default", "none", "reset"):
                s.x_tolerance = None
            elif raw:
                try:
                    s.x_tolerance = float(raw)
                except ValueError:
                    print(c("  Not a number - left unchanged.", "warn"))


# --------------------------------------------------------------------------- #
# Friendly error hints
# --------------------------------------------------------------------------- #
def _is_auth_error(e: Exception) -> bool:
    m = str(e).lower()
    return any(w in m for w in ("api key", "apikey", "authentication", "401",
                                "invalid x-api-key", "unauthorized"))


def _error_hint(e: Exception) -> str:
    m = str(e).lower()
    if "tectonic" in m:
        return "Install Tectonic and put it on PATH (see the README's Installation section)."
    if _is_auth_error(e):
        return "That API key looks wrong - I've cleared it; you'll be asked again next time."
    if any(w in m for w in ("connection", "timeout", "network", "getaddrinfo", "temporarily")):
        return "Looks like a network problem - check your internet connection and retry."
    if "rate limit" in m or "429" in m:
        return "You're being rate-limited - wait a moment and try again."
    if "no text extracted" in m or "image-only" in m:
        return "That PDF has no text layer (scanned/image-only) - the parsers can't OCR it."
    return ""


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def _banner() -> None:
    rule = "  " + "=" * 46
    print(c("\n" + rule, "h"))
    print(c("   CV-Agent  ·  terminal UI", "b"))
    print(c("   parse -> extract -> render  ·  score  ·  improve", "dim"))
    print(c(rule, "h"))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Turkish characters on Windows
    except Exception:
        pass

    s = Session()
    _banner()
    # (label, function, tag)
    actions = [
        ("Create a CV from a description", action_create, "needs API key"),
        ("Convert an existing CV to PDF", action_convert, "needs API key"),
        ("Score a CV for ATS", action_score, "needs API key"),
        ("Improve a CV for a job (guarded)", action_improve, "needs API key"),
        ("Render the built-in sample", action_sample, "offline, no key"),
        ("Settings (provider / model / output)", action_settings, ""),
    ]
    menu = [(label + (c(f"   ({tag})", "dim") if tag else ""), fn) for label, fn, tag in actions]

    while True:
        print(c(f"\n  Provider: {s.provider}    Model: {s.model_label}    Output: {s.output_dir}/", "dim"))
        action = choose("Main menu", menu, back="Quit")
        if action is None:
            print("  Bye.")
            return
        try:
            action(s)  # type: ignore[operator]
        except KeyboardInterrupt:
            print(c("\n  Cancelled.", "warn"))
        except SystemExit:
            raise
        except Exception as e:
            print(c(f"\n  Error: {e}", "err"))
            if _is_auth_error(e):
                s.api_key = None
            hint = _error_hint(e)
            if hint:
                print(c("  " + hint, "dim"))
        pause()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")

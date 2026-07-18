"""Input parsers: turn a source CV file into plain text for the LLM step.

Each format gets its own module so their heavy third-party deps stay decoupled
(import only what you use):

    DOCX  -> cv_agent.parsers.docx   (python-docx)
    PDF   -> cv_agent.parsers.pdf    (pdfplumber)
    text  -> cv_agent.parsers.text   (passthrough)  [to come]

Import the one you need directly, e.g.::

    from cv_agent.parsers.docx import extract_text
"""

"""
pdf_parser.py
─────────────
Reads the AKAR website consolidated PDF and splits it into named sections.

Section detection rule
──────────────────────
A line that matches the pattern:

    <SOME TEXT> ( https://... )

is treated as a new section header.
  • section_title  = text before "("
  • url            = content inside "( … )"

All subsequent lines belong to that section until the next header appears.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Matches:  Any text  ( http://... )  or  ( https://... )
_SECTION_HEADER_RE = re.compile(
    r"^(?P<title>.+?)\s*\(\s*(?P<url>https?://[^\s)]+)\s*\)\s*$",
    re.IGNORECASE,
)


@dataclass
class Section:
    section_title: str
    url: str
    full_text: str = ""
    lines: list[str] = field(default_factory=list, repr=False)

    def finalise(self) -> None:
        """Join accumulated lines into full_text."""
        self.full_text = "\n".join(self.lines).strip()


def parse_pdf_sections(pdf_path: str) -> list[Section]:
    """
    Parse *pdf_path* and return a list of Section objects, each containing:
      - section_title  (e.g. "HERO PAGE")
      - url            (e.g. "https://akar-strategic-consultants.netlify.app")
      - full_text      (all text under that header)
    """
    doc = fitz.open(pdf_path)
    logger.info("Opened PDF '%s' — %d pages", pdf_path, len(doc))

    sections: list[Section] = []
    current: Optional[Section] = None

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        lines = raw_text.splitlines()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            match = _SECTION_HEADER_RE.match(stripped)
            if match:
                # Finalise previous section
                if current is not None:
                    current.finalise()
                    sections.append(current)
                    logger.debug(
                        "Section '%s' — %d chars",
                        current.section_title,
                        len(current.full_text),
                    )

                title = match.group("title").strip()
                url   = match.group("url").strip()
                current = Section(section_title=title, url=url)
                logger.info("New section detected: '%s' → %s  (page %d)", title, url, page_num)
            else:
                if current is not None:
                    current.lines.append(stripped)
                # Lines before the first section header are silently discarded

    # Finalise the last section
    if current is not None:
        current.finalise()
        sections.append(current)

    doc.close()
    logger.info("Parsed %d sections from PDF", len(sections))
    return sections

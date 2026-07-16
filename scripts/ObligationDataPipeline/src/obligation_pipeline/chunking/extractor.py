"""PDF text extraction with page boundaries for chunking."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PAGE_BREAK = "\f"


@dataclass
class PageContent:
    """Text content for a single page."""
    page_number: int
    text: str


@dataclass
class ExtractedDocument:
    """Full extracted text and per-page boundaries for a PDF."""
    full_text: str
    pages: list[PageContent]
    num_pages: int

    def get_page_at_offset(self, offset: int) -> Optional[int]:
        """Return 1-based page number containing the given character offset."""
        if offset < 0 or not self.pages:
            return None
        cumul = 0
        for p in self.pages:
            end = cumul + len(p.text) + (1 if cumul + len(p.text) < len(self.full_text) else 0)
            if offset < end:
                return p.page_number
            cumul = end
        return self.pages[-1].page_number if self.pages else None


def extract_pdf_pages(
    path: Path,
    skip_pages: int = 0,
    page_break: str = PAGE_BREAK,
) -> Optional[ExtractedDocument]:
    """Extract text from a PDF with per-page boundaries."""
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")
        return None

    path = Path(path)
    if not path.exists() or not path.suffix.lower() == ".pdf":
        logger.warning("Not a PDF or file missing: %s", path)
        return None

    try:
        doc = fitz.open(path)
    except Exception as e:
        logger.warning("Failed to open PDF %s: %s", path, e)
        return None

    pages: list[PageContent] = []
    parts: list[str] = []

    try:
        for i in range(len(doc)):
            page_num = i + 1
            if page_num <= skip_pages:
                continue
            page = doc[i]
            text = page.get_text().strip()
            pages.append(PageContent(page_number=page_num, text=text))
            if parts:
                parts.append(page_break)
            parts.append(text)
        doc.close()
    except Exception as e:
        logger.warning("Error extracting pages from %s: %s", path, e)
        doc.close()
        return None

    full_text = "".join(parts)
    return ExtractedDocument(full_text=full_text, pages=pages, num_pages=len(pages))

# app/rag/extract.py
from pathlib import Path

from pypdf import PdfReader

from app.config import OCR_ENABLED, OCR_MIN_CHARS
from app.rag import ocr


class ExtractionError(Exception):
    """Raised when a file cannot be turned into usable text."""


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}


def extract_text(path: str | Path) -> tuple[str, dict]:
    """
    Turn a file into plain text.

    Returns the text plus a report:
        {"ocr_pages": int, "total_pages": int, "ocr_skipped": int,
         "page_spans": [{"page": 1, "start": 0, "end": 1240, "ocr": False}, ...]}

    page_spans records where each page begins and ends in the returned string,
    so the chunker can map a chunk back to the pages it came from.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)

    if suffix == ".txt":
        text = _extract_txt(path).strip()
        ocr_used = False
    elif suffix in IMAGE_EXTENSIONS:
        text = ocr.image_to_text(path).strip()
        ocr_used = True
    else:
        raise ExtractionError(f"Cannot extract text from '{suffix}' files.")

    if not text:
        raise ExtractionError(
            "No text could be extracted from this file, even with OCR. "
            "The image may be too low quality, or contain no text."
        )
    return text, {
        "ocr_pages": 1 if ocr_used else 0,
        "total_pages": 1,
        "ocr_skipped": 0,
        "page_spans": [{"page": 1, "start": 0, "end": len(text), "ocr": ocr_used}],
    }


def _extract_pdf(path: Path) -> tuple[str, dict]:
    """
    Read each page's text layer. Pages with no usable text are rendered to
    images and read with OCR. While joining the pages into one string, record
    each page's character range so page numbers survive chunking.
    """
    reader = PdfReader(path)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    ocr_flags = [False] * len(pages)

    needs_ocr = [i for i, text in enumerate(pages) if len(text) < OCR_MIN_CHARS]

    ocr_used, ocr_skipped = 0, 0
    if needs_ocr and OCR_ENABLED and ocr.is_available():
        ocr_text, ocr_skipped = ocr.ocr_pdf_pages(path, needs_ocr)
        for i, text in ocr_text.items():
            if text:
                pages[i] = text
                ocr_flags[i] = True
                ocr_used += 1

    separator = "\n\n"
    parts, spans, cursor = [], [], 0
    for i, page_text in enumerate(pages):
        if not page_text:
            continue                      # blank pages take up no characters
        if parts:
            cursor += len(separator)      # the join will insert this
        spans.append({
            "page": i + 1,                # humans count pages from 1
            "start": cursor,
            "end": cursor + len(page_text),
            "ocr": ocr_flags[i],
        })
        parts.append(page_text)
        cursor += len(page_text)

    text = separator.join(parts)
    if not text.strip():
        raise ExtractionError(
            "No text could be extracted from this file, even with OCR. "
            "If this is a scanned PDF, the scan may be too low quality."
        )

    return text, {
        "ocr_pages": ocr_used,
        "total_pages": len(pages),
        "ocr_skipped": ocr_skipped,
        "page_spans": spans,
    }

def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
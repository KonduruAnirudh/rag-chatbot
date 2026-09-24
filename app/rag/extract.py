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

    Returns the text plus a short report:
        {"ocr_pages": int, "total_pages": int, "ocr_skipped": int}
    so the caller can tell the user how much of the document was read by OCR.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text, report = _extract_pdf(path)
    elif suffix == ".txt":
        text = _extract_txt(path)
        report = {"ocr_pages": 0, "total_pages": 1, "ocr_skipped": 0}
    elif suffix in IMAGE_EXTENSIONS:
        text = ocr.image_to_text(path)
        report = {"ocr_pages": 1, "total_pages": 1, "ocr_skipped": 0}
    else:
        raise ExtractionError(f"Cannot extract text from '{suffix}' files.")

    text = text.strip()
    if not text:
        raise ExtractionError(
            "No text could be extracted from this file, even with OCR. "
            "The image may be too low quality, or contain no text."
        )
    return text, report


def _extract_pdf(path: Path) -> tuple[str, dict]:
    """
    Read each page's text layer. Pages with no usable text are rendered to
    images and read with OCR, so a scanned page inside an otherwise normal
    document still gets indexed.
    """
    reader = PdfReader(path)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]

    needs_ocr = [i for i, text in enumerate(pages) if len(text) < OCR_MIN_CHARS]

    ocr_used, ocr_skipped = 0, 0
    if needs_ocr and OCR_ENABLED and ocr.is_available():
        ocr_text, ocr_skipped = ocr.ocr_pdf_pages(path, needs_ocr)
        for i, text in ocr_text.items():
            if text:
                pages[i] = text
                ocr_used += 1

    return "\n\n".join(p for p in pages if p), {
        "ocr_pages": ocr_used,
        "total_pages": len(pages),
        "ocr_skipped": ocr_skipped,
    }


def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
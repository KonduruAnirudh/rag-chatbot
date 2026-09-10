# app/rag/extract.py
from pathlib import Path
from pypdf import PdfReader


class ExtractionError(Exception):
    """Raised when a file cannot be turned into usable text."""


def extract_text(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf(path)
    elif suffix == ".txt":
        text = _extract_txt(path)
    else:
        raise ExtractionError(f"Cannot extract text from '{suffix}' files.")

    text = text.strip()
    if not text:
        raise ExtractionError(
            "No text could be extracted. If this is a scanned PDF, "
            "it contains images rather than text and would require OCR."
        )
    return text


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
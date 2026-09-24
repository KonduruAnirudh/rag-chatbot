# app/rag/ocr.py
"""
Optical character recognition for pages and images with no text layer.

Two engines, chosen by OCR_ENGINE in config:

  "vision"    — sends the page image to the LLM. Understands layout, so tables
                keep their structure and diagrams are described as relationships.
                Costs tokens. Required for graph extraction from diagrams.

  "tesseract" — classic OCR, local and free. Reads characters only: a flowchart
                becomes a list of disconnected words. Needs a system install:
                    macOS:  brew install tesseract
                    Ubuntu: apt install tesseract-ocr
"""
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from openai import OpenAI

from app.config import (
    CHAT_MODEL, OCR_CONCURRENCY, OCR_ENGINE, OCR_MAX_PAGES,
    REASONING_EFFORT, TESSERACT_DPI, VISION_DPI, require_api_key,
)

# Optional: only needed when OCR_ENGINE is "tesseract"
try:
    import pytesseract
except ImportError:
    pytesseract = None


class OCRError(Exception):
    """Raised when OCR is unavailable or fails."""


# A hung request must fail rather than hold a worker thread forever.
VISION_TIMEOUT_SECONDS = 90

# Vision models read a page fine at this size. Larger images cost more tokens
# and take longer to upload without improving the transcription.
VISION_MAX_PIXELS = 1600

VISION_PROMPT = """You transcribe scanned document pages.

Rules:
- Transcribe ALL text exactly as written. Do not summarise, correct, or reword.
- Keep the reading order. For multi-column pages, finish one column before the next.
- Render tables as markdown tables, preserving rows and columns.
- For a diagram, flowchart or architecture picture, transcribe its labels AND
  describe the connections in words, on their own line, like:
  [Diagram: Client sends requests to Load Balancer. Load Balancer forwards to
  API Server. API Server reads from Database.]
- If part of the page is unreadable, write [unreadable] rather than guessing.
- Output only the page content. No preamble, no commentary."""

_client = None


def _openai() -> OpenAI:
    """Created on first use, so importing this module doesn't require an API key."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=require_api_key())
    return _client


# --- public API ----------------------------------------------------

def is_available() -> bool:
    """True if the configured engine can actually run."""
    if OCR_ENGINE == "vision":
        try:
            require_api_key()
            return True
        except Exception:
            return False
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def image_to_text(path: str | Path) -> str:
    """Read text from a standalone image file."""
    _require_engine()
    try:
        with Image.open(path) as image:
            return _read(image.convert("RGB"))
    except OCRError:
        raise
    except Exception as e:
        print(f"[OCR ERROR] {type(e).__name__}: {e}")
        raise OCRError("Could not read text from this image.")


def ocr_pdf_pages(path: str | Path, page_numbers: list[int]) -> tuple[dict[int, str], int]:
    """
    OCR the given pages of a PDF.
    Returns {page_number: text} and how many pages the cap skipped.
    """
    if not page_numbers:
        return {}, 0

    _require_engine()

    selected = page_numbers[:OCR_MAX_PAGES]
    skipped = len(page_numbers) - len(selected)

    pdf = pdfium.PdfDocument(str(path))
    try:
        dpi = VISION_DPI if OCR_ENGINE == "vision" else TESSERACT_DPI
        images = {n: pdf[n].render(scale=dpi / 72).to_pil() for n in selected}
    finally:
        pdf.close()   # release the native handle before the slow part

    # Pages are independent, so they can be read in parallel. Concurrency is
    # capped to stay well inside API rate limits.
    with ThreadPoolExecutor(max_workers=OCR_CONCURRENCY) as pool:
        texts = list(pool.map(_read_safely, images.values()))

    return dict(zip(images.keys(), texts)), skipped


# --- engines -------------------------------------------------------

def _read(image: Image.Image) -> str:
    if OCR_ENGINE == "vision":
        return _read_with_vision(image)
    return _read_with_tesseract(image)


def _read_safely(image: Image.Image) -> str:
    """One unreadable page must not fail the whole document."""
    try:
        return _read(image)
    except Exception as e:
        print(f"[OCR ERROR] page failed: {type(e).__name__}: {e}")
        return ""


def _shrink(image: Image.Image, max_side: int = VISION_MAX_PIXELS) -> Image.Image:
    """Scale an image down so its longest side is at most max_side pixels."""
    if max(image.size) <= max_side:
        return image
    ratio = max_side / max(image.size)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    return image.resize(new_size, Image.LANCZOS)


def _read_with_vision(image: Image.Image) -> str:
    image = _shrink(image)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()

    response = _openai().responses.create(
        model=CHAT_MODEL,
        instructions=VISION_PROMPT,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": "Transcribe this page."},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
        ]}],
        reasoning={"effort": REASONING_EFFORT},
        temperature=0.0,   # transcription, not writing: no variation wanted
        timeout=VISION_TIMEOUT_SECONDS,
    )
    return response.output_text.strip()


def _read_with_tesseract(image: Image.Image) -> str:
    return pytesseract.image_to_string(image).strip()


def _require_engine() -> None:
    if is_available():
        return
    if OCR_ENGINE == "vision":
        raise OCRError("OCR is not available: no API key is configured.")
    raise OCRError(
        "OCR is not available: the Tesseract binary is not installed. "
        "Install it with `brew install tesseract` (macOS) or "
        "`apt install tesseract-ocr` (Linux)."
    )
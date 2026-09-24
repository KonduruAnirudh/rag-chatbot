# eval/ocr_check.py
"""
How accurate is Tesseract on our documents?

Renders real pages to images (simulating a scan), OCRs them, and compares
the result against pypdf's extraction of the same page, which we treat as
ground truth.

    python -m eval.ocr_check data/test_documents/rfc8935.pdf
"""
import re
import sys

import pypdfium2 as pdfium
import pytesseract
from pypdf import PdfReader

DPI = 300          # classic OCR needs a high-resolution image
PAGES = 3          # sample size


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def overlap(truth: list[str], guess: list[str]) -> float:
    """Share of the true words that OCR also produced (order ignored)."""
    if not truth:
        return 0.0
    remaining = list(guess)
    hits = 0
    for word in truth:
        if word in remaining:
            remaining.remove(word)
            hits += 1
    return hits / len(truth)


def main(path: str):
    reader = PdfReader(path)
    pdf = pdfium.PdfDocument(path)
    pages = min(PAGES, len(pdf))

    print(f"{path} — {len(pdf)} pages, checking the first {pages} at {DPI} DPI\n")

    for i in range(pages):
        truth_text = reader.pages[i].extract_text() or ""
        if len(truth_text.strip()) < 100:
            print(f"page {i}: skipped (no text layer to compare against)")
            continue

        image = pdf[i].render(scale=DPI / 72).to_pil()
        ocr_text = pytesseract.image_to_string(image)

        truth, guess = words(truth_text), words(ocr_text)
        print(f"page {i}: pypdf {len(truth):4} words | ocr {len(guess):4} words | "
              f"recovered {overlap(truth, guess):.0%}")

    # Show one page side by side so you can judge the errors yourself
    image = pdf[0].render(scale=DPI / 72).to_pil()
    print("\n--- pypdf, first 300 chars ---")
    print((reader.pages[0].extract_text() or "")[:300])
    print("\n--- tesseract, first 300 chars ---")
    print(pytesseract.image_to_string(image)[:300])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/test_documents/rfc8935.pdf")
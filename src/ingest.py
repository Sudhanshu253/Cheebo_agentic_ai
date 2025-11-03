# src/ingest.py
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io, os, re
from typing import Dict

OCR_DPI = 200
MIN_TEXT_LEN = 50         # if page text length < MIN_TEXT_LEN -> use OCR

def _clean_extracted_text(text: str) -> str:
    """
    Light cleaning:
    - normalize newlines
    - remove repeated header/footer lines (heuristic)
    - remove multiple hyphens used as separators
    - trim long runs of whitespace
    """
    if not text:
        return ""
    # Normalize unicode spaces and newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove common page header/footer patterns: lines with pdf page numbers or short repeating lines
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned_lines = []
    last = None
    for ln in lines:
        # skip empty lines beyond single spacing
        if ln == "" and (not cleaned_lines or cleaned_lines[-1] == ""):
            continue
        # skip lines that are just page numbers or very short repeated tokens like "IIT KANPUR" etc.
        if re.fullmatch(r"\d{1,4}", ln) or (len(ln) < 6 and ln.isupper()):
            continue
        # remove repeated short headers (heuristic)
        if ln == last and len(ln) < 30:
            continue
        # remove long sequences of hyphens
        if re.fullmatch(r"[-=]{3,}", ln):
            continue
        cleaned_lines.append(ln)
        last = ln
    txt = "\n".join(cleaned_lines)
    # collapse multiple blank lines
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    # normalize multiple spaces
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()

def pdf_page_to_image(page, dpi=OCR_DPI):
    """Return a PIL image from a PyMuPDF page for OCR."""
    pix = page.get_pixmap(dpi=dpi)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def pdf_to_text(path: str, ocr_enabled: bool = True, debug_write: bool = False) -> str:
    """
    Extract text from PDF using PyMuPDF, falling back to OCR for pages with little text.
    Returns the cleaned full-text string.
    If debug_write=True, writes <filename>.txt in data/raw_notes/debug/ for inspection.
    """
    doc = fitz.open(path)
    texts = []
    for i, page in enumerate(doc):
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        # If extracted text is short and OCR enabled, do OCR
        if ocr_enabled and (not text or len(text.strip()) < MIN_TEXT_LEN):
            try:
                img = pdf_page_to_image(page)
                # pytesseract returns '\n' terminated lines; specify lang if needed
                ocr_text = pytesseract.image_to_string(img)
                # prefer OCR when it's longer than the extracted text
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
            except Exception:
                # fallback: keep whatever text we have
                pass
        texts.append(text)

    full = "\n\n".join(texts)
    cleaned = _clean_extracted_text(full)
    if debug_write:
        os.makedirs(os.path.join("data","raw_notes","debug"), exist_ok=True)
        fn = os.path.basename(path)
        with open(os.path.join("data","raw_notes","debug", fn + ".txt"), "w", encoding="utf-8") as f:
            f.write(cleaned)
    return cleaned


def load_all_notes(folder: str = "data/raw_notes", debug_write: bool = False) -> Dict[str, str]:
    """
    Loads all PDF files under `folder`. Returns dict of {filename: full_text}.
    If debug_write=True, also writes extracted .txt files to data/raw_notes/debug/.
    """
    docs = {}
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(".pdf"):
            path = os.path.join(folder, fn)
            try:
                docs[fn] = pdf_to_text(path, ocr_enabled=True, debug_write=debug_write)
            except Exception as e:
                docs[fn] = ""
    return docs


if __name__ == "__main__":
    docs = load_all_notes(debug_write=True)
    for k, v in docs.items():
        print(k, len(v), "chars")

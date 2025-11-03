# src/chunker.py
import re
from typing import List, Tuple

SENTENCE_END = re.compile(r'(?<=[\.\?\!])\s+')

def contains_latex(text: str) -> bool:
    latex_patterns = [
        r"\$.*?\$",         # inline math
        r"\\\(.*?\\\)",     # \( ... \)
        r"\\\[.*?\\\]",     # \[ ... \]
        r"\\begin\{equation\}.*?\\end\{equation\}",  # equation blocks
        r"\\begin\{align\}.*?\\end\{align\}",
    ]
    return any(re.search(pat, text, flags=re.DOTALL) for pat in latex_patterns)


def is_heading(line: str) -> bool:
    """
    Heuristic to detect headings: short line, title-case or ends with ':', or all-caps
    """
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if s.isupper() and len(s.split()) < 8:
        return True
    if s.istitle() and len(s.split()) < 6:
        return True
    if s.endswith(":") and len(s.split()) < 8:
        return True
    return False


def normalize_paragraphs(text: str) -> List[str]:
    """
    Turn raw page text into paragraphs:
    - merge lines that are likely word-wrapped (line breaks inside sentences)
    - preserve blank-line separated paragraphs
    """
    # First, separate by blank lines
    blocks = re.split(r'\n\s*\n', text)
    paras = []
    for b in blocks:
        # remove excessive newlines within block
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        # join lines that look like wrapped lines (no punctuation at end)
        joined = []
        buf = ""
        for ln in lines:
            if buf:
                # if previous line ended with punctuation, start new sentence
                if re.search(r'[\.!?:"\)\]]\s*$', buf):
                    joined.append(buf.strip())
                    buf = ln
                else:
                    # likely wrapped, concatenate with space
                    buf = buf + " " + ln
            else:
                buf = ln
        if buf:
            joined.append(buf.strip())
        # each element in joined is a paragraph candidate
        for j in joined:
            paras.append(j.strip())
    return paras


def split_into_chunks(text: str, max_chars: int = 1500, overlap_chars: int = 200) -> List[str]:
    """
    Main chunker:
    - normalize paragraphs
    - keep LaTeX paragraphs intact
    - try to keep headings attached to following paragraph
    - use sentence-aware overlap (not raw character overlap)
    """
    paragraphs = normalize_paragraphs(text)
    chunks = []
    cur = ""

    def flush_current():
        nonlocal cur
        if cur.strip():
            # cleanup multiple newlines
            c = re.sub(r'\n{3,}','\n\n', cur.strip())
            chunks.append(c)
            cur = ""

    for i, p in enumerate(paragraphs):
        # attach heading to next paragraph
        if is_heading(p):
            # if current chunk non-empty and small enough, append heading to it
            if not cur or len(cur) + len(p) + 2 <= max_chars:
                cur = cur + ("\n\n" + p if cur else p)
                continue
            else:
                flush_current()
                cur = p
                continue

        # if paragraph has LaTeX, never split inside — treat as a block
        if contains_latex(p):
            # if adding would overflow, flush current first
            if cur and len(cur) + len(p) + 2 > max_chars:
                flush_current()
            # add latex paragraph (as its own block if large)
            if len(p) > max_chars:
                # split by sentences but keep latex spans intact (do not split inside $...$)
                flush_current()
                chunks.append(p)
                cur = ""
            else:
                cur += ("\n\n" + p) if cur else p
                # if cur now too large, flush
                if len(cur) > max_chars:
                    flush_current()
            continue

        # normal paragraph: try to append
        if not cur:
            cur = p
        elif len(cur) + len(p) + 2 <= max_chars:
            cur += "\n\n" + p
        else:
            # need to flush: create overlap by taking last few sentences from cur
            # sentence-aware overlap
            sent_parts = SENTENCE_END.split(cur)
            overlap_text = ""
            rem = ""
            # accumulate sentences from the end until reach overlap_chars
            for s in reversed(sent_parts):
                if not s.strip():
                    continue
                candidate = (s + " " + overlap_text).strip()
                if len(candidate) > overlap_chars:
                    break
                overlap_text = candidate
            flush_current()
            # start new chunk with overlap + current paragraph
            if overlap_text:
                cur = overlap_text + "\n\n" + p
            else:
                cur = p

    # final flush
    if cur.strip():
        chunks.append(re.sub(r'\n{3,}', '\n\n', cur.strip()))

    # final cleanup: trim leading/trailing whitespace & filter tiny chunks
    cleaned = [c.strip() for c in chunks if len(c.strip()) > 30]
    return cleaned


if __name__ == "__main__":
    # quick smoke test
    sample = """Transmission Line Models

Short lines neglect capacitance.

The nominal \\pi model uses Z = R + jX and shunt admittance Y = jB/2.

Equation example:
\\[
A = 1 + \\frac{ZY}{2}
\\]

Surge Impedance Loading (SIL) is ...
"""
    ch = split_into_chunks(sample, max_chars=300)
    print("Chunks:", len(ch))
    for i,c in enumerate(ch):
        print("----",i+1, c[:200].replace("\n"," "))

# scripts/raw_to_readable.py
import json
import os
import re
import sys
import textwrap
from pathlib import Path

WRAP_WIDTH = 90

def clean_repeated_filenames(text: str) -> str:
    """
    Replace repeated occurrences of filenames like 'Something.pdf Something.pdf ...'
    with just a single filename.
    """
    # collapse runs like "X.pdf X.pdf X.pdf" -> "X.pdf"
    text = re.sub(r'(\b[\w\-\.\(\)]+\.pdf\b)(?:\s+\1)+', r'\1', text)
    return text

def collapse_repeated_lines(text: str) -> str:
    """
    Remove consecutive duplicate lines (often caused by bad OCR or model echo).
    """
    lines = text.splitlines()
    out_lines = []
    prev = None
    for ln in lines:
        ln_stripped = ln.strip()
        if ln_stripped == prev:
            # skip exact duplicate
            continue
        out_lines.append(ln)
        prev = ln_stripped
    return "\n".join(out_lines)

def normalize_whitespace(text: str) -> str:
    # Convert Windows newlines, remove leading/trailing whitespace on each line
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # remove repeated blank lines (more than 2 -> 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # trim spaces at line ends
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    return text.strip()

def wrap_paragraphs(text: str, width: int = WRAP_WIDTH) -> str:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    wrapped = []
    for p in paragraphs:
        wrapped.append(textwrap.fill(p, width=width))
    return "\n\n".join(wrapped)

def raw_to_readable(raw: str) -> str:
    t = raw
    t = clean_repeated_filenames(t)
    t = collapse_repeated_lines(t)
    t = normalize_whitespace(t)
    t = wrap_paragraphs(t)
    return t

def json_to_readable(infile: Path):
    data = json.load(open(infile, 'r', encoding='utf-8'))
    # Prefer structured fields if present
    if isinstance(data, dict) and any(k in data for k in ("topic","summary","key_points")):
        # Convert structured JSON to markdown if available
        topic = data.get("topic", infile.stem)
        md_lines = [f"# {topic}\n"]
        if data.get("summary"):
            md_lines.append("## Summary\n")
            md_lines.append(data["summary"].strip() + "\n")
        if data.get("key_points"):
            md_lines.append("## Key points\n")
            for kp in data["key_points"]:
                md_lines.append(f"- {kp}")
            md_lines.append("")
        if data.get("formulas"):
            md_lines.append("## Formulas\n")
            for f in data["formulas"]:
                latex = f.get("latex","")
                name = f.get("name","")
                meaning = f.get("meaning","")
                units = f.get("units","")
                md_lines.append(f"- **{name}** `{latex}` — {meaning} [{units}]")
            md_lines.append("")
        if data.get("important_questions"):
            md_lines.append("## Important questions\n")
            for q in data["important_questions"]:
                md_lines.append(f"- Q: {q.get('q')}")
                md_lines.append(f"  - Why important: {q.get('why_important')}")
                md_lines.append(f"  - Difficulty: {q.get('difficulty')}\n")
        if data.get("solved_examples"):
            md_lines.append("## Solved examples\n")
            for ex in data["solved_examples"]:
                md_lines.append(f"### {ex.get('question')}\n")
                for i, s in enumerate(ex.get("solution_steps", []), 1):
                    md_lines.append(f"{i}. {s}")
                md_lines.append(f"**Final answer:** {ex.get('final_answer')}\n")
        md = "\n".join(md_lines).strip()
        txt = re.sub(r'\n{2,}', '\n\n', md)
    else:
        raw_text = data.get("raw_output") or data.get("output") or json.dumps(data, ensure_ascii=False)
        txt = raw_to_readable(raw_text)

        # create a simple markdown wrapper
        md = f"# {infile.stem}\n\n" + txt

    # write outputs
    out_txt = infile.with_name(infile.stem + "_readable.txt")
    out_md = infile.with_name(infile.stem + ".md")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    return out_txt, out_md

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/raw_to_readable.py <path-to-output-json>  OR  python scripts/raw_to_readable.py outputs/*.json")
        sys.exit(1)

    files = []
    # expand globs
    for arg in sys.argv[1:]:
        files.extend(sorted(Path(".").glob(arg)))

    if not files:
        print("No files found for:", sys.argv[1:])
        sys.exit(1)

    for f in files:
        print("Processing:", f)
        out_txt, out_md = json_to_readable(f)
        print("Saved text:", out_txt)
        print("Saved markdown:", out_md)
        print()

if __name__ == "__main__":
    main()

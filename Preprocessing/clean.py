#!/usr/bin/env python3
# clean_texts.py
"""
Cleans NCERT_cleaned/<Grade>/<Subject>/*.txt -> NCERT_cleaned_cleaned/<Grade>/<Subject>/*.txt
Heuristics:
 - remove lines that look like repeated headers/footers (short uppercase lines, page numbers)
 - fix broken words like "Y ou" -> "You"
 - collapse excessive whitespace
 - remove common OCR artifacts
"""
from pathlib import Path
import re, argparse

SRC = Path("NCERT_cleaned")
OUT = Path("NCERT_cleaned_cleaned")
OUT.mkdir(parents=True, exist_ok=True)

UPPER_SHORT_RE = re.compile(r'^[A-Z][A-Z0-9 \-]{2,80}$')  # likely header/footer line
PAGE_RE = re.compile(r'^\s*page\s*\d+\s*$', re.I)
CHAPTER_CONT_RE = re.compile(r'^\s*\d+(\.\d+)*\s*')  # numeric headings

def fix_broken_words(text):
    # merge single-letter splits like "Y ou", "ac count"
    # pattern: letter + space + letter/word but avoid breaking normal spaced words
    text = re.sub(r'(?<=\b[A-Za-z])\s+(?=[A-Za-z]{1,3}\b)', '', text)
    # also remove stray spaces inside words caused by OCR: sequences of single letters split by spaces
    text = re.sub(r'(?<=\S)\s+(?=\S)', ' ', text)
    return text

def remove_headers_footers(lines):
    out=[]
    last_seen = []
    for i,ln in enumerate(lines):
        s = ln.strip()
        # drop obvious page number lines
        if PAGE_RE.match(s):
            continue
        # drop short all-caps lines that repeat frequently
        if UPPER_SHORT_RE.match(s) and len(s.split()) <= 6:
            # but keep if looks like a sentence (has vowel-lowercase)
            # check previous/next lines to detect repetition
            last_seen.append(s)
            continue
        # drop short lines that look like "Accountancy - Partnership Accounts" repeated
        out.append(ln)
    return out

def clean_file(in_path, out_path):
    raw = in_path.read_text(encoding="utf-8", errors="ignore")
    # normalize newlines
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    # remove weird control chars
    raw = ''.join(ch for ch in raw if ch == '\n' or ord(ch) >= 32)
    # split lines and remove repeated headers/footers heuristically
    lines = raw.split('\n')
    lines = [l for l in lines if l.strip() != '']
    lines = remove_headers_footers(lines)
    text = "\n".join(lines)
    # fix broken words and common OCR artifacts
    text = fix_broken_words(text)
    # common OCR: "ﬁ" ligature etc.
    text = text.replace("\ufb01", "fi").replace("\ufb02","ff").replace("ﬁ","fi")
    # remove multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # collapse multi-newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grade", type=str, help="Grade folder (e.g., Grade_12)")
    parser.add_argument("--subject", type=str, help="Subject folder")
    parser.add_argument("--limit", type=int, help="Limit chapters for quick test")
    args = parser.parse_args()
    processed = 0
    for g in sorted([d for d in SRC.iterdir() if d.is_dir()]):
        if args.grade and g.name != args.grade: continue
        for s in sorted([d for d in g.iterdir() if d.is_dir()]):
            if args.subject and s.name != args.subject: continue
            for f in sorted(s.glob("*.txt")):
                if args.limit and processed >= args.limit:
                    break
                out_p = OUT / g.name / s.name / f.name
                try:
                    clean_file(f, out_p)
                    processed += 1
                except Exception as e:
                    print("Error cleaning", f, e)
            if args.limit and processed >= args.limit: break
        if args.limit and processed >= args.limit: break
    print("Cleaned files:", processed)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# paragraph_restore.py
"""
Reads NCERT_cleaned_cleaned/<grade>/<subject>/*.txt and writes 
NCERT_paragraphs/<grade>/<subject>/<chapter>.para.txt
Each output file is paragraph-separated (double newline), with headings preserved.
"""
from pathlib import Path
import re, argparse

SRC = Path("NCERT_cleaned_cleaned")
OUT = Path("NCERT_paragraphs")
OUT.mkdir(parents=True, exist_ok=True)

# Patterns to detect headings or list starts
HEADING_RE = re.compile(r'^(?:Chapter|CHAPTER|1\.|[0-9]+\.)\s*', re.I)
NUMBERED_LINE_RE = re.compile(r'^\s*\d+[\.\)]\s+')
SENT_END_RE = re.compile(r'(?<=[\.\?\!])\s+')

def restore_paragraphs(text):
    # split on double newline first
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if paras:
        # further split very long paragraphs by sentence boundaries
        out = []
        for p in paras:
            if len(p.split()) > 800:
                sents = re.split(SENT_END_RE, p)
                buf = []
                bufw = 0
                for s in sents:
                    w = len(s.split())
                    if bufw + w > 200 and buf:
                        out.append(" ".join(buf).strip())
                        buf=[s.strip()]; bufw = w
                    else:
                        buf.append(s.strip()); bufw += w
                if buf: out.append(" ".join(buf).strip())
            else:
                out.append(p)
        return out
    # fallback: split on headings and numbered lines
    lines = text.splitlines()
    paras=[]; cur=[]
    for ln in lines:
        ln = ln.strip()
        if not ln: 
            if cur:
                paras.append(" ".join(cur)); cur=[]
            continue
        if HEADING_RE.match(ln) or NUMBERED_LINE_RE.match(ln) or (ln.isupper() and len(ln.split())<8):
            if cur:
                paras.append(" ".join(cur)); cur=[]
            paras.append(ln)
        else:
            cur.append(ln)
    if cur: paras.append(" ".join(cur))
    return paras

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grade", type=str)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    processed=0
    for g in sorted([d for d in SRC.iterdir() if d.is_dir()]):
        if args.grade and g.name != args.grade: continue
        for s in sorted([d for d in g.iterdir() if d.is_dir()]):
            if args.subject and s.name != args.subject: continue
            for f in sorted(s.glob("*.txt")):
                if args.limit and processed >= args.limit: break
                try:
                    raw = f.read_text(encoding="utf-8", errors="ignore")
                    paras = restore_paragraphs(raw)
                    outp = OUT / g.name / s.name
                    outp.mkdir(parents=True, exist_ok=True)
                    (outp / f.name).write_text("\n\n".join(paras), encoding="utf-8")
                    processed += 1
                except Exception as e:
                    print("Error restoring", f, e)
            if args.limit and processed >= args.limit: break
        if args.limit and processed >= args.limit: break
    print("Paragraph files created:", processed)

if __name__ == "__main__":
    main()

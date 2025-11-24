#!/usr/bin/env python3
"""
clean_ncert_text.py

Cleans and normalizes text extracted from NCERT PDFs:
- Removes repeated headers/footers and page numbers
- Fixes hyphenation and joins broken lines into paragraphs
- Normalizes unicode and punctuation
- Produces NCERT_cleaned/... and a CSV report

Usage: python clean_ncert_text.py
"""

import re
import unicodedata
import csv
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

# -------- CONFIG ----------
IN_ROOT = Path("NCERT_text")        # input from Task B
OUT_ROOT = Path("NCERT_cleaned")    # cleaned output
REPORT_CSV = Path("cleaning_report.csv")

# Heuristic thresholds
HEADER_FOOTER_MIN_PAGES = 3         # require at least this many pages to consider header/footer frequency
HEADER_FREQ_THRESHOLD = 0.30        # if a candidate (first/last line) appears in >=30% pages, treat as header/footer
SHORT_LINE_MAX = 3                  # lines with <= this many chars considered noise (page numbers etc.)
# ---------------------------

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# --- utility normalization functions ---
def unicode_normalize(s: str) -> str:
    # Normalize to NFKC
    return unicodedata.normalize("NFKC", s)

def normalize_punctuation(s: str) -> str:
    # Replace common fancy quotes and dashes with ASCII equivalents
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u00A0": " ", "\u200B": "",
        "\u2026": "...",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # collapse multiple dots
    s = re.sub(r'\.{3,}', '...', s)
    return s

def remove_noise_lines(lines):
    """Remove lines that are very short or purely digits (page numbers)"""
    out = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        # line with only digits (page numbers) or roman numerals
        if re.fullmatch(r'[\dIVXLCMivxlcm]+', stripped):
            continue
        # tiny artifacts like "*", "-" or single char
        if len(stripped) <= SHORT_LINE_MAX and re.fullmatch(r'[^A-Za-z\u0900-\u097F0-9]+', stripped):
            continue
        # if line is just 1-3 chars and not alnum, skip
        if len(stripped) <= SHORT_LINE_MAX and not re.search(r'[A-Za-z\u0900-\u097F0-9]', stripped):
            continue
        out.append(ln)
    return out

# --- page-splitting heuristic ---
def split_to_pages(text: str):
    """
    Try to split chapter text into pages:
    We use double-newline blocks as approximate page separators.
    If extracted text contains formfeeds '\f' use that as page delimiter.
    """
    if '\f' in text:
        pages = text.split('\f')
        return [p.strip() for p in pages if p.strip()]
    # split by two or more newlines as page boundaries
    pages = re.split(r'\n{2,}', text)
    return [p.strip() for p in pages if p.strip()]

# --- header/footer detection ---
def detect_repeating_headers_footers(pages):
    """
    For each page take its first and last non-empty lines as candidate header/footer.
    Count frequency across pages and return sets to remove.
    """
    first_lines = []
    last_lines = []
    for p in pages:
        lines = [ln for ln in p.splitlines() if ln.strip()]
        if not lines:
            continue
        first_lines.append(lines[0].strip())
        last_lines.append(lines[-1].strip())

    header_to_remove = set()
    footer_to_remove = set()
    n_pages = max(1, len(pages))
    if n_pages >= HEADER_FOOTER_MIN_PAGES:
        # count frequency
        fcount = Counter(first_lines)
        lcount = Counter(last_lines)
        for line, cnt in fcount.items():
            if cnt / n_pages >= HEADER_FREQ_THRESHOLD and len(line) > 2:
                header_to_remove.add(line)
        for line, cnt in lcount.items():
            if cnt / n_pages >= HEADER_FREQ_THRESHOLD and len(line) > 2:
                footer_to_remove.add(line)
    return header_to_remove, footer_to_remove

# --- hyphenation fix & paragraph join ---
def fix_hyphens_and_join(lines):
    """
    Join lines into paragraphs:
    - If a line ends with a hyphen '-' (word split), remove hyphen and join directly.
    - If previous line ends with sentence punctuation (.!?;:) or is blank, preserve break.
    - Else join with space.
    """
    out_lines = []
    prev = ""
    for ln in lines:
        s = ln.rstrip()
        if not s:
            # empty line -> paragraph break
            if prev:
                out_lines.append(prev.strip())
                prev = ""
            continue

        # if prev ended with hyphenated split
        if prev.endswith('-'):
            prev = prev[:-1] + s.lstrip()  # join broken word
            continue

        # if prev is empty, start new
        if not prev:
            prev = s
            continue

        # if prev ends with sentence end -> start new paragraph
        if re.search(r'[\.!?;:]\s*$', prev):
            out_lines.append(prev.strip())
            prev = s
            continue

        # otherwise join with space
        prev = prev + " " + s.lstrip()

    if prev:
        out_lines.append(prev.strip())
    return out_lines

# --- main cleaning for a single chapter text ---
def clean_chapter_text(raw_text: str):
    # unicode normalize
    t = unicode_normalize(raw_text)
    t = normalize_punctuation(t)

    # split into pages for header/footer detection
    pages = split_to_pages(t)
    # detect candidate headers/footers
    headers, footers = detect_repeating_headers_footers(pages)

    cleaned_pages = []
    removed_header_footer_count = 0
    for p in pages:
        # remove any header/footer lines that match exactly
        lines = p.splitlines()
        # trim lines
        lines = [ln.rstrip() for ln in lines]
        # remove header (first line) if present in headers set
        if lines and lines[0].strip() in headers:
            removed_header_footer_count += 1
            lines = lines[1:]
        # remove footer (last line) if present in footers
        if lines and lines[-1].strip() in footers:
            removed_header_footer_count += 1
            lines = lines[:-1]
        # remove noise lines (page numbers etc.)
        lines = remove_noise_lines(lines)
        # now fix hyphens and join into paragraphs
        paras = fix_hyphens_and_join(lines)
        cleaned_pages.extend(paras)

    # join cleaned paragraphs with double newline
    cleaned_text = "\n\n".join([p.strip() for p in cleaned_pages if p.strip()])

    # final cleanups
    # collapse multiple blank lines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    # collapse multiple spaces
    cleaned_text = re.sub(r'[ \t]{2,}', ' ', cleaned_text)
    # strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip()

    return cleaned_text, len(pages), removed_header_footer_count

# --- walkthrough of all chapters ---
def process_all(in_root: Path, out_root: Path, report_csv: Path):
    rows = []
    total = 0
    failed = 0
    # iterate file structure Grade/Subject/Chapter.txt
    for grade_dir in sorted(in_root.iterdir()):
        if not grade_dir.is_dir():
            continue
        for subj_dir in sorted(grade_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            out_subj_dir = out_root / grade_dir.name / subj_dir.name
            out_subj_dir.mkdir(parents=True, exist_ok=True)
            for chap_file in sorted(subj_dir.glob("*.txt")):
                total += 1
                try:
                    raw = chap_file.read_text(encoding="utf-8", errors="ignore")
                    cleaned_text, n_pages, removed_count = clean_chapter_text(raw)
                    out_path = out_subj_dir / chap_file.name
                    if cleaned_text:
                        out_path.write_text(cleaned_text, encoding="utf-8")
                    else:
                        out_path.write_text("", encoding="utf-8")
                    rows.append({
                        "grade": grade_dir.name,
                        "subject": subj_dir.name,
                        "chapter_file": str(chap_file.relative_to(in_root)),
                        "pages_detected": n_pages,
                        "removed_header_footer_count": removed_count,
                        "orig_chars": len(raw),
                        "clean_chars": len(cleaned_text),
                        "cleaned_path": str(out_path.relative_to(out_root))
                    })
                except Exception as e:
                    failed += 1
                    rows.append({
                        "grade": grade_dir.name,
                        "subject": subj_dir.name,
                        "chapter_file": str(chap_file.relative_to(in_root)),
                        "error": str(e)
                    })
                    print(f"✖ Failed cleaning {chap_file}: {e}")

    # write CSV report
    fieldnames = ["grade","subject","chapter_file","pages_detected","removed_header_footer_count",
                  "orig_chars","clean_chars","cleaned_path","error"]
    with report_csv.open("w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nDone: processed {total} files, failed {failed}. Report -> {report_csv}")

if __name__ == "__main__":
    process_all(IN_ROOT, OUT_ROOT, REPORT_CSV)

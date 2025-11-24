#!/usr/bin/env python3
"""
rechunk_paragraphs_improved.py

Sliding-window word-based rechunker.

Outputs:
 - NCERT_passages_rechunk/passages.jsonl  (one JSON per passage: id, grade, subject, chapter, text, token_count, source)
 - NCERT_passages_rechunk/passage_index.csv

Usage examples:
  # Rechunk only Grade_6 (safe test)
  python rechunk_paragraphs_improved.py --grade Grade_6 --limit 10

  # Rechunk all content (careful)
  python rechunk_paragraphs_improved.py
"""
import argparse, json, csv, re, math
from pathlib import Path
from tqdm import tqdm

SRC_ROOT = Path("NCERT_cleaned")
OUT_ROOT = Path("NCERT_passages_rechunk")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_ROOT / "passages.jsonl"
OUT_CSV = OUT_ROOT / "passage_index.csv"

# default chunking params (recommended)
TARGET_WORDS = 250   # aim ~250 words per chunk
MIN_WORDS = 120
MAX_WORDS = 400
STRIDE_WORDS = 100   # overlap

def safe_read_text(p: Path):
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return p.read_text(encoding="latin-1", errors="ignore")

def normalize_whitespace(t: str):
    # collapse excessive whitespace but keep single newlines to help paragraph split
    t = t.replace('\r', '\n')
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'[ \t]+', ' ', t)
    return t.strip()

def paragraph_split(text: str):
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if paras:
        return paras
    # fallback: split by sentence-ish boundaries
    sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', text) if s.strip()]
    # group sentences into paragraph-like units
    grouped=[]
    cur=[]; curw=0
    for s in sents:
        w = len(s.split())
        if cur and curw + w > 120:
            grouped.append(" ".join(cur))
            cur=[s]; curw=w
        else:
            cur.append(s); curw += w
    if cur: grouped.append(" ".join(cur))
    return grouped

def sliding_word_chunks_from_text(text: str, target=TARGET_WORDS, min_w=MIN_WORDS, max_w=MAX_WORDS, stride=STRIDE_WORDS):
    words = text.split()
    n = len(words)
    if n == 0:
        return []
    if n <= max_w:
        return [" ".join(words)]
    chunks = []
    i = 0
    while i < n:
        j = i + target
        if j >= n:
            # make last chunk include trailing words but ensure >= min_w by shifting left
            start = max(0, n - target)
            chunks.append(" ".join(words[start:n]))
            break
        chunks.append(" ".join(words[i:j]))
        i += (target - stride)
    # post-process: merge any very short chunk with previous
    final=[]
    for c in chunks:
        if final and len(c.split()) < min_w:
            final[-1] = final[-1] + " " + c
        else:
            final.append(c)
    return final

def chunks_from_paragraphs(paras, **kw):
    # join small consecutive paragraphs until we have at least MIN_WORDS, then apply sliding on that block
    blocks=[]
    cur=[]; curw=0
    for p in paras:
        w = len(p.split())
        if curw + w > kw.get("max_block_words", 2*kw.get("target", TARGET_WORDS)):
            blocks.append(" ".join(cur))
            cur=[p]; curw=w
        else:
            cur.append(p); curw += w
    if cur:
        blocks.append(" ".join(cur))
    # now create word-chunks from each block
    out=[]
    for b in blocks:
        out.extend(sliding_word_chunks_from_text(b, target=kw["target"], min_w=kw["min_w"], max_w=kw["max_w"], stride=kw["stride"]))
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grade", type=str, help="only process this grade folder (e.g., Grade_6)")
    parser.add_argument("--subject", type=str, help="only this subject")
    parser.add_argument("--limit", type=int, help="limit number of chapter files processed (for quick test)")
    parser.add_argument("--target", type=int, default=TARGET_WORDS)
    parser.add_argument("--min_words", type=int, default=MIN_WORDS)
    parser.add_argument("--max_words", type=int, default=MAX_WORDS)
    parser.add_argument("--stride", type=int, default=STRIDE_WORDS)
    args = parser.parse_args()

    target = args.target; min_w = args.min_words; max_w = args.max_words; stride = args.stride
    processed = 0; written = 0

    # clear previous outputs safely
    if OUT_JSONL.exists(): OUT_JSONL.unlink()
    if OUT_CSV.exists(): OUT_CSV.unlink()

    with OUT_JSONL.open("a", encoding="utf-8") as jf, OUT_CSV.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["id","grade","subject","chapter","passage_index","token_count","source"])
        writer.writeheader()
        grade_dirs = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])
        for g in grade_dirs:
            if args.grade and g.name != args.grade:
                continue
            subj_dirs = sorted([d for d in g.iterdir() if d.is_dir()])
            for s in subj_dirs:
                if args.subject and s.name != args.subject:
                    continue
                chap_files = sorted([f for f in s.glob("*.txt") if f.is_file()])
                for chap_file in chap_files:
                    if args.limit and processed >= args.limit:
                        break
                    processed += 1
                    try:
                        raw = safe_read_text(chap_file)
                        text = normalize_whitespace(raw)
                        paras = paragraph_split(text)
                        chunks = chunks_from_paragraphs(paras, target=target, min_w=min_w, max_w=max_w, stride=stride, max_block_words=max_w*2)
                        # if no chunks (weird), fallback to single
                        if not chunks:
                            chunks = [text]
                        for i, ch in enumerate(chunks):
                            pid = f"{g.name}___{s.name}___{chap_file.stem}___p{i:04d}"
                            token_count = len(re.findall(r"\S+", ch))
                            rec = {"id": pid, "grade": g.name, "subject": s.name, "chapter": chap_file.stem, "passage_index": i, "token_count": token_count, "source": str(chap_file), "text": ch}
                            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            writer.writerow({"id": rec["id"], "grade": rec["grade"], "subject": rec["subject"], "chapter": rec["chapter"], "passage_index": rec["passage_index"], "token_count": rec["token_count"], "source": rec["source"]})
                            written += 1
                    except Exception as e:
                        print("Error processing", chap_file, e)
                    if args.limit and processed >= args.limit:
                        break
                if args.limit and processed >= args.limit:
                    break
            if args.limit and processed >= args.limit:
                break

    print(f"Processed chapters: {processed}, Passages written: {written}")
    print("Wrote JSONL:", OUT_JSONL)
    print("Wrote CSV:", OUT_CSV)

if __name__ == "__main__":
    main()

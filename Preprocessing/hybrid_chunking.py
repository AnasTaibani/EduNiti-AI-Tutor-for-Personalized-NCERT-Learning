#!/usr/bin/env python3
# chunk_passages_hybrid.py
"""
Reads NCERT_paragraphs/<grade>/<subject>/*.para.txt and writes NCERT_passages_hybrid/passages.jsonl
Hybrid chunking: keep paragraphs, but if a paragraph > MAX_WORDS -> split by sentences. Then merge consecutive paragraphs until target words.
"""
from pathlib import Path
import re, json, csv, argparse
from tqdm import tqdm

PARA_DIR = Path("NCERT_paragraphs")
OUT_DIR = Path("NCERT_passages_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "passages.jsonl"
OUT_CSV = OUT_DIR / "passage_index.csv"

# defaults tuned for NCERT
TARGET_WORDS = 250
MIN_WORDS = 120
MAX_WORDS = 400
STRIDE_WORDS = 100

SENT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def sent_split(text):
    sents = re.split(SENT_RE, text)
    return [s.strip() for s in sents if s.strip()]

def chunk_paragraph(p):
    words = p.split()
    n = len(words)
    if n <= MAX_WORDS:
        return [" ".join(words)]
    # sliding windows over words if paragraph too long
    chunks=[]
    i=0
    while i < n:
        end = min(i+TARGET_WORDS, n)
        chunks.append(" ".join(words[i:end]))
        if end==n: break
        i += (TARGET_WORDS - STRIDE_WORDS)
    return chunks

def merge_to_target(blocks, target=TARGET_WORDS, min_w=MIN_WORDS, stride=STRIDE_WORDS):
    out=[]
    cur=[]
    curw=0
    for b in blocks:
        w = len(b.split())
        if curw + w > target and cur:
            out.append(" ".join(cur))
            # start new buffer; include overlap: keep last `stride` words from previous
            cur = [b]
            curw = w
        else:
            cur.append(b); curw += w
    if cur:
        out.append(" ".join(cur))
    # cleanup: merge smalls
    final=[]
    for c in out:
        if final and len(c.split()) < min_w:
            final[-1] = final[-1] + " " + c
        else:
            final.append(c)
    return final

def process_file(fpath):
    raw = fpath.read_text(encoding="utf-8", errors="ignore")
    paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
    blocks=[]
    for p in paras:
        # if paragraph tiny, keep; if too big, split by sentences and then into word windows
        if len(p.split()) > MAX_WORDS:
            sents = sent_split(p)
            # group sentences into chunks roughly target size
            buf=[]; bufw=0
            for s in sents:
                sw = len(s.split())
                if buf and bufw + sw > TARGET_WORDS:
                    blocks.append(" ".join(buf)); buf=[s]; bufw=sw
                else:
                    buf.append(s); bufw += sw
            if buf: blocks.append(" ".join(buf))
        else:
            blocks.append(p)
    # now merge blocks into final chunks
    final = merge_to_target(blocks, target=TARGET_WORDS, min_w=MIN_WORDS, stride=STRIDE_WORDS)
    return final

def main():
    global TARGET_WORDS, MIN_WORDS, MAX_WORDS, STRIDE_WORDS
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--grade", type=str)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--limit", type=int, help="limit number of chapter files processed")
    parser.add_argument("--target", type=int, default=TARGET_WORDS)
    parser.add_argument("--min_words", type=int, default=MIN_WORDS)
    parser.add_argument("--max_words", type=int, default=MAX_WORDS)
    parser.add_argument("--stride", type=int, default=STRIDE_WORDS)
    args = parser.parse_args()

    # override defaults
    
    TARGET_WORDS, MIN_WORDS, MAX_WORDS, STRIDE_WORDS = args.target, args.min_words, args.max_words, args.stride

    processed=0; written=0
    if OUT_JSONL.exists(): OUT_JSONL.unlink()
    if OUT_CSV.exists(): OUT_CSV.unlink()

    with OUT_JSONL.open("a", encoding="utf-8") as jf, OUT_CSV.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["id","grade","subject","chapter","passage_index","token_count","source"])
        writer.writeheader()
        grade_dirs = sorted([d for d in PARA_DIR.iterdir() if d.is_dir()])
        for g in grade_dirs:
            if args.grade and g.name != args.grade: continue
            for s in sorted([d for d in g.iterdir() if d.is_dir()]):
                if args.subject and s.name != args.subject: continue
                for f in sorted(s.glob("*.txt")):
                    if args.limit and processed >= args.limit: break
                    processed += 1
                    try:
                        chunks = process_file(f)
                        for i,ch in enumerate(chunks):
                            pid = f"{g.name}___{s.name}___{f.stem}___p{i:04d}"
                            token_count = len(re.findall(r"\S+", ch))
                            rec = {"id": pid, "grade": g.name, "subject": s.name, "chapter": f.stem, "passage_index": i, "token_count": token_count, "source": str(f), "text": ch}
                            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            writer.writerow({"id": rec["id"], "grade": rec["grade"], "subject": rec["subject"], "chapter": rec["chapter"], "passage_index": rec["passage_index"], "token_count": rec["token_count"], "source": rec["source"]})
                            written += 1
                    except Exception as e:
                        print("Error chunking", f, e)
                    if args.limit and processed >= args.limit: break
                if args.limit and processed >= args.limit: break
            if args.limit and processed >= args.limit: break
    print("Processed chapters:", processed, "Passages written:", written)

if __name__ == "__main__":
    main()

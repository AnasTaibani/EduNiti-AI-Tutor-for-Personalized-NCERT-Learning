#!/usr/bin/env python3
"""
build_gold_from_bm25_queries.py

Builds gold_eval_set.csv by:
  - loading NCERT passages from passages.jsonl
  - building BM25 index on the fly
  - for each query in train_queries.txt, selecting top-1 passage id

Inputs:
  - Preprocessing/NCERT_passages_hybrid/passages.jsonl
  - Retrieval/eval/train_queries.txt

Output:
  - Retrieval/eval/gold_eval_set.csv   (columns: query, expected_id)
"""

import json
import csv
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi
import re


# --------- CONFIG: adjust if your paths differ ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PASSAGES_JSONL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "passages.jsonl"
QUERIES_PATH   = PROJECT_ROOT / "Retrieval" / "eval" / "train_queries.txt"
OUT_CSV        = PROJECT_ROOT / "Retrieval" / "eval" / "gold_eval_set.csv"
# --------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase + word characters."""
    return re.findall(r"\w+", (text or "").lower())


def load_passages(path: Path) -> List[Dict]:
    """Load passages from JSONL; each line is a JSON object with at least 'id' and 'text'."""
    if not path.exists():
        raise FileNotFoundError(f"Passages file not found at: {path}")
    passages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Expecting at least 'id' and 'text'
            pid = obj.get("id")
            txt = obj.get("text", "")
            if not pid or not txt.strip():
                continue
            passages.append({
                "id": pid,
                "text": txt,
                "grade": obj.get("grade"),
                "subject": obj.get("subject"),
                "chapter": obj.get("chapter"),
            })
    return passages


def load_queries(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found at: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def main():
    print(f"➡️ Loading passages from: {PASSAGES_JSONL}")
    passages = load_passages(PASSAGES_JSONL)
    print(f"   Loaded {len(passages)} passages")

    if not passages:
        print("❌ No passages loaded. Check your passages.jsonl.")
        return

    print("➡️ Tokenizing passages for BM25...")
    corpus_tokens = [tokenize(p["text"]) for p in passages]

    print("➡️ Building BM25Okapi index...")
    bm25 = BM25Okapi(corpus_tokens)
    print("   BM25 index ready.")

    print(f"➡️ Loading queries from: {QUERIES_PATH}")
    queries = load_queries(QUERIES_PATH)
    print(f"   Total queries: {len(queries)}")

    out_rows = []

    for q in queries:
        q_tokens = tokenize(q)
        if not q_tokens:
            continue
        scores = bm25.get_scores(q_tokens)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_passage = passages[best_idx]
        pid = best_passage["id"]

        out_rows.append({
            "query": q,
            "expected_id": pid
        })

        # nice debug print
        print(f"Query: {q} -> {pid} (grade={best_passage.get('grade')}, subject={best_passage.get('subject')})")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "expected_id"])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print("✅ Wrote gold eval set to:", OUT_CSV)
    print("   Total pairs:", len(out_rows))


if __name__ == "__main__":
    main()

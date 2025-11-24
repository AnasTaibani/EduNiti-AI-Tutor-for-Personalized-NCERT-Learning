#!/usr/bin/env python3
"""
bm25_retrieval_demo.py

Simple BM25-only retrieval over NCERT passages.

Uses:
  - Preprocessing/NCERT_passages_hybrid/bm25.pkl
      Expected format (from your pipeline): {
          "bm25": BM25Okapi instance,
          "tokenized_corpus": List[List[str]],
          "ids": List[str]   # passage IDs
      }

  - Preprocessing/NCERT_passages_hybrid/passages.jsonl
      One JSON per line:
      {
        "id": "...",
        "grade": "...",
        "subject": "...",
        "chapter": "...",
        "text": "...",
        ...
      }

Usage:
  python Retrieval/bm25/bm25_retrieval_demo.py --query "What is photosynthesis?"
  python Retrieval/bm25/bm25_retrieval_demo.py --topk 10 --query "Causes of French Revolution"
"""

import argparse
import json
import pickle
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi


# ---- PATHS (adjust if needed) ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BM25_PKL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "bm25.pkl"
PASSAGES_JSONL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "passages.jsonl"


# ---- Tokenization (no NLTK) ----
def simple_tokenize(text: str) -> List[str]:
    """
    Simple regex-based tokenizer: lowercase + split on word chars.
    Works for English + most Latin text; harmless for others.
    """
    if not text:
        return []
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


# ---- Loading BM25 + passages ----
def load_bm25(
    bm25_path: Path = BM25_PKL,
    passages_path: Path = PASSAGES_JSONL,
) -> Tuple[BM25Okapi, List[List[str]], List[str], Dict[str, dict]]:
    """
    Load BM25 object, corpus tokens, ids list, and id -> passage meta mapping.
    """
    print(f"Loading BM25 from: {bm25_path}")
    with bm25_path.open("rb") as f:
        obj = pickle.load(f)

    # Expect dict format from your pipeline
    if not isinstance(obj, dict):
        raise ValueError("bm25.pkl is not a dict. Expected keys: bm25, tokenized_corpus, ids")

    missing = {k for k in ["bm25", "tokenized_corpus", "ids"] if k not in obj}
    if missing:
        raise ValueError(f"bm25.pkl missing keys: {missing}")

    bm25: BM25Okapi = obj["bm25"]
    corpus_tokens: List[List[str]] = obj["tokenized_corpus"]
    ids: List[str] = obj["ids"]

    # Build id -> passage metadata from passages.jsonl
    print(f"Loading passages from: {passages_path}")
    id2rec: Dict[str, dict] = {}
    id_set = set(ids)

    with passages_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("id")
            if pid in id_set:
                id2rec[pid] = rec

    print(f"BM25 corpus size: {len(ids)} passages (id2rec mapped: {len(id2rec)})")
    return bm25, corpus_tokens, ids, id2rec


# ---- BM25 search ----
def search_bm25(
    query: str,
    bm25: BM25Okapi,
    corpus_tokens: List[List[str]],
    ids: List[str],
    id2rec: Dict[str, dict],
    topk: int = 5,
):
    q_tokens = simple_tokenize(query)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)
    # get topk indices
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:topk]

    results = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        pid = ids[idx] if idx < len(ids) else None
        rec = id2rec.get(pid, {})
        results.append({
            "rank": rank,
            "id": pid,
            "score": float(score),
            "grade": rec.get("grade"),
            "subject": rec.get("subject"),
            "chapter": rec.get("chapter"),
            "text": rec.get("text", ""),
        })
    return results


def print_results(query: str, results: List[dict]):
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    if not results:
        print("No results found.")
        return

    for r in results:
        print(
            f"[{r['rank']}] {r['id']}  score={r['score']:.3f}  "
            f"(grade={r.get('grade')}, subject={r.get('subject')}, chapter={r.get('chapter')})"
        )
        txt = (r.get("text") or "").replace("\n", " ")
        snippet = txt[:300] + ("..." if len(txt) > 300 else "")
        wrapped = textwrap.fill(snippet, width=80, subsequent_indent="    ")
        print("    " + wrapped)
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Natural language query")
    parser.add_argument("--topk", type=int, default=5, help="Number of results to show")
    parser.add_argument("--bm25-path", type=str, default=str(BM25_PKL))
    parser.add_argument("--passages-path", type=str, default=str(PASSAGES_JSONL))
    args = parser.parse_args()

    bm25, corpus_tokens, ids, id2rec = load_bm25(
        Path(args.bm25_path),
        Path(args.passages_path),
    )

    results = search_bm25(args.query, bm25, corpus_tokens, ids, id2rec, topk=args.topk)
    print_results(args.query, results)


if __name__ == "__main__":
    main()

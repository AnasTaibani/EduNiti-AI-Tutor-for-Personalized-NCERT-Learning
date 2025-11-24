#!/usr/bin/env python3
"""
bm25_cross_encoder_rerank_demo.py

Task E — Cross-encoder reranker integration (precision boosting)

Pipeline:
  1) Load BM25 index + passages from Preprocessing/NCERT_passages_hybrid
  2) For a given query, get top-N candidates from BM25
  3) Re-rank those candidates using a cross-encoder model
  4) Print final ranked results

Usage:
  python Retrieval/rerank/bm25_cross_encoder_rerank_demo.py --query "What is photosynthesis?"
  python Retrieval/rerank/bm25_cross_encoder_rerank_demo.py --query "पर्यावरण संरक्षण के उपाय क्या हैं" --topk-bm25 50 --topk-final 5
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ---------- Paths (adapted to your repo layout) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROC_ROOT = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid"
BM25_PKL = PREPROC_ROOT / "bm25.pkl"
PASSAGES_JSONL = PREPROC_ROOT / "passages.jsonl"

# A small, relatively fast cross-encoder.
# This is English-focused; for more Hindi/Sanskrit support we can switch to a multilingual one.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------- Helpers ----------

def simple_tokenize(text: str) -> List[str]:
    """Simple word tokenizer (no NLTK to avoid punkt downloads)."""
    return re.findall(r"\w+", (text or "").lower())


def load_passages(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load passages.jsonl into a dict: id -> passage_meta."""
    id2rec: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("id")
            if pid:
                id2rec[pid] = rec
    return id2rec


def load_bm25(bm25_path: Path, passages_path: Path):
    """
    Load BM25 index produced in preprocessing.

    Expected pickle format (from your current pipeline):
      obj = {
        "bm25": BM25Okapi(...),
        "tokenized_corpus": List[List[str]],
        "ids": List[str]
      }
    """
    print(f"Loading BM25 from: {bm25_path}")
    with bm25_path.open("rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise ValueError("Unexpected BM25 pickle format; expected dict with keys: 'bm25', 'tokenized_corpus', 'ids'.")

    bm25 = obj.get("bm25")
    corpus_tokens = obj.get("tokenized_corpus")
    ids = obj.get("ids")

    if not isinstance(bm25, BM25Okapi):
        raise ValueError("bm25 entry in pickle is not a BM25Okapi object.")
    if not isinstance(corpus_tokens, list) or not isinstance(ids, list):
        raise ValueError("tokenized_corpus or ids are missing/invalid in BM25 pickle.")

    # Load passage metadata / text
    print(f"  Loading passages from: {passages_path}")
    id2rec = load_passages(passages_path)

    print(f"  BM25 corpus size: {len(corpus_tokens)} passages")
    print(f"  Passages in id2rec: {len(id2rec)}")

    return bm25, corpus_tokens, ids, id2rec


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    corpus_tokens: List[List[str]],
    ids: List[str],
    id2rec: Dict[str, Dict[str, Any]],
    topk: int = 20
) -> List[Dict[str, Any]]:
    """BM25 search: returns topk candidate passages with BM25 scores."""
    q_tokens = simple_tokenize(query)
    scores = bm25.get_scores(q_tokens)  # shape: [num_docs]
    scores = np.asarray(scores)

    topk = min(topk, len(scores))
    top_idx = np.argpartition(-scores, topk - 1)[:topk]
    # sort these topk by score desc
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        pid = ids[idx] if idx < len(ids) else None
        rec = id2rec.get(pid, {})
        text = rec.get("text", "")
        results.append({
            "rank": rank,
            "id": pid,
            "score": float(scores[idx]),
            "grade": rec.get("grade"),
            "subject": rec.get("subject"),
            "chapter": rec.get("chapter"),
            "text": text
        })
    return results


def rerank_with_cross_encoder(
    query: str,
    bm25_results: List[Dict[str, Any]],
    model: CrossEncoder,
    topk_final: int = 5,
    batch_size: int = 8
) -> List[Dict[str, Any]]:
    """
    Given BM25 candidates, re-rank them using a cross-encoder.

    model.predict takes list of (query, passage_text) tuples and returns scores.
    """
    if not bm25_results:
        return []

    # Prepare (query, passage) pairs
    pairs = []
    for res in bm25_results:
        text = res.get("text") or ""
        # If passages are very long, truncate to keep things fast & within max_length
        # The cross-encoder model has its own tokenizer; we just trim raw chars a bit.
        pairs.append((query, text[:1000]))

    # Predict relevance scores
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    scores = scores.tolist()

    # Attach scores and sort
    for res, s in zip(bm25_results, scores):
        res["ce_score"] = float(s)

    reranked = sorted(bm25_results, key=lambda r: r["ce_score"], reverse=True)
    return reranked[:topk_final]


def print_results(title: str, results: List[Dict[str, Any]], show_text: bool = True):
    print(f"\n{title}")
    if not results:
        print("  (no results)")
        return
    for r in results:
        rid = r.get("id")
        score = r.get("score") or r.get("ce_score")
        grade = r.get("grade")
        subj = r.get("subject")
        chap = r.get("chapter")
        print(f"  [{r['rank']}] {rid}  score={score:.3f}  (grade={grade}, subject={subj}, chapter={chap})")
        if show_text:
            text = r.get("text") or ""
            snippet = text[:160].replace("\n", " ")
            print(f"      {snippet}")
            if len(text) > 160:
                print("      ...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="User query to test retrieval.")
    parser.add_argument("--topk-bm25", type=int, default=20, help="Number of BM25 candidates before reranking.")
    parser.add_argument("--topk-final", type=int, default=5, help="Final results after reranking.")
    parser.add_argument("--model", type=str, default=CROSS_ENCODER_MODEL, help="Cross-encoder model name.")
    args = parser.parse_args()

    # 1) Load BM25 and corpus
    bm25, corpus_tokens, ids, id2rec = load_bm25(BM25_PKL, PASSAGES_JSONL)

    # 2) BM25 search
    print("\n" + "=" * 60)
    print(f"Query: {args.query}")
    print("=" * 60)

    bm25_candidates = bm25_search(
        args.query,
        bm25,
        corpus_tokens,
        ids,
        id2rec,
        topk=args.topk_bm25,
    )
    print_results("Top BM25 candidates (before rerank):", bm25_candidates[:5])

    # 3) Load cross-encoder
    print(f"\nLoading cross-encoder model: {args.model}")
    reranker = CrossEncoder(args.model, max_length=256)  # CPU by default

    # 4) Rerank
    reranked = rerank_with_cross_encoder(
        args.query,
        bm25_candidates,
        model=reranker,
        topk_final=args.topk_final,
    )

    print_results("Top results AFTER cross-encoder rerank:", reranked, show_text=True)


if __name__ == "__main__":
    main()

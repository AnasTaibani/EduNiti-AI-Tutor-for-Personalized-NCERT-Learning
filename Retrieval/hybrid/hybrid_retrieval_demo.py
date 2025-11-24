#!/usr/bin/env python3
"""
hybrid_retrieval_demo.py

Task D (part 2) — Hybrid retrieval: BM25 + DPR (FAISS).

Uses:
  - BM25 index over passages (lexical relevance)
  - DPR embeddings + FAISS index (semantic relevance)
Combines them into a hybrid score for better retrieval.

Inputs:
  - Preprocessing/NCERT_passages_hybrid/passages.jsonl
  - Preprocessing/NCERT_passages_hybrid/bm25.pkl
  - Retrieval/dpr/passages_dpr_embs.npy
  - Retrieval/dpr/passages_dpr_meta.json
  - Retrieval/dpr/indexes/dpr_faiss_flat.index  (or ivf / hnsw)

Usage (from project root):
  cd D:\eduniti-majorProject

  # Single query
  python Retrieval\\hybrid\\hybrid_retrieval_demo.py --query "What is photosynthesis?"

  # Interactive
  python Retrieval\\hybrid\\hybrid_retrieval_demo.py --interactive
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss


# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PASSAGES_JSONL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "passages.jsonl"
BM25_PKL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "bm25.pkl"

DPR_EMB_PATH = PROJECT_ROOT / "Retrieval" / "dpr" / "passages_dpr_embs.npy"
DPR_META_PATH = PROJECT_ROOT / "Retrieval" / "dpr" / "passages_dpr_meta.json"

# pick which FAISS index to use (flat / ivf / hnsw)
DPR_FAISS_FLAT   = PROJECT_ROOT / "Retrieval" / "dpr" / "indexes" / "dpr_faiss_flat.index"
DPR_FAISS_IVF    = PROJECT_ROOT / "Retrieval" / "dpr" / "indexes" / "dpr_faiss_ivf.index"
DPR_FAISS_HNSW   = PROJECT_ROOT / "Retrieval" / "dpr" / "indexes" / "dpr_faiss_hnsw.index"

DPR_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

TOPK_BM25 = 20
TOPK_DPR = 20
TOPK_HYBRID = 10
ALPHA_BM25 = 0.5  # weight for BM25 vs DPR in hybrid score


# ---------- HELPERS ----------

def simple_tokenize(text: str):
    text = (text or "").lower()
    return re.findall(r"\w+", text)


def load_passages_map(jsonl_path: Path):
    """
    Build a dict id -> record (for showing snippets).
    """
    id2rec = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = obj.get("id")
            if pid:
                id2rec[pid] = obj
    return id2rec


def load_bm25(bm25_path: Path):
    with bm25_path.open("rb") as f:
        data = pickle.load(f)
    ids = data["ids"]
    tokenized_corpus = data["tokenized_corpus"]
    bm25 = data["bm25"]
    return ids, tokenized_corpus, bm25


def load_dpr():
    embs = np.load(DPR_EMB_PATH)
    with DPR_META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    # meta[i]["idx"] should equal i, but we map explicitly anyway
    id_by_idx = {m["idx"]: m["id"] for m in meta}
    return embs, meta, id_by_idx


def load_faiss_index(index_path: Path):
    index = faiss.read_index(str(index_path))
    return index


def build_query_encoder():
    model = SentenceTransformer(DPR_MODEL_NAME)
    return model


def normalize_scores(scores: np.ndarray):
    """
    Min-max normalize to [0, 1]. If constant, return zeros.
    """
    s_min = float(scores.min())
    s_max = float(scores.max())
    if s_max - s_min < 1e-8:
        return np.zeros_like(scores, dtype="float32")
    return ((scores - s_min) / (s_max - s_min)).astype("float32")


def bm25_search(query: str, ids, bm25: BM25Okapi, topk: int):
    q_tokens = simple_tokenize(query)
    scores = np.array(bm25.get_scores(q_tokens), dtype="float32")
    # sort descending
    top_idx = np.argsort(-scores)[:topk]
    results = []
    for rank, idx in enumerate(top_idx, start=1):
        results.append({
            "rank": rank,
            "id": ids[idx],
            "score": float(scores[idx]),
            "idx": int(idx),
        })
    return results, scores


def dpr_search(query: str, encoder, faiss_index, topk: int):
    q_emb = encoder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    # if index is IP with normalized vectors, distances are similarities
    D, I = faiss_index.search(q_emb, topk)
    D = D[0]
    I = I[0]
    results = []
    for rank, (dist, idx) in enumerate(zip(D, I), start=1):
        if idx < 0:
            continue
        results.append({
            "rank": rank,
            "idx": int(idx),
            "score": float(dist),
        })
    return results, D


def hybrid_search(query: str,
                  ids,
                  bm25: BM25Okapi,
                  encoder,
                  faiss_index,
                  id_by_idx,
                  alpha_bm25: float = ALPHA_BM25,
                  topk_bm25: int = TOPK_BM25,
                  topk_dpr: int = TOPK_DPR,
                  topk_hybrid: int = TOPK_HYBRID):
    # BM25 and DPR search
    bm25_results, bm25_scores = bm25_search(query, ids, bm25, topk_bm25)
    dpr_results, dpr_scores_full = dpr_search(query, encoder, faiss_index, topk_dpr)

    # Collect indices used
    bm25_indices = [r["idx"] for r in bm25_results]
    dpr_indices = [r["idx"] for r in dpr_results]

    # Normalize scores on subsets to combine
    bm25_sub_scores = np.array([bm25_scores[i] for i in bm25_indices], dtype="float32")
    dpr_sub_scores = np.array([r["score"] for r in dpr_results], dtype="float32")

    bm25_norm = normalize_scores(bm25_sub_scores)
    dpr_norm = normalize_scores(dpr_sub_scores)

    # Map idx -> normalized scores
    bm25_norm_map = {idx: float(s) for idx, s in zip(bm25_indices, bm25_norm)}
    dpr_norm_map = {idx: float(s) for idx, s in zip(dpr_indices, dpr_norm)}

    # Union of candidate indices
    candidate_indices = sorted(set(bm25_indices) | set(dpr_indices))

    hybrid_results = []
    for idx in candidate_indices:
        bm25_s = bm25_norm_map.get(idx, 0.0)
        dpr_s = dpr_norm_map.get(idx, 0.0)
        hybrid_s = alpha_bm25 * bm25_s + (1.0 - alpha_bm25) * dpr_s
        pid = id_by_idx.get(idx, ids[idx] if 0 <= idx < len(ids) else None)
        hybrid_results.append({
            "idx": int(idx),
            "id": pid,
            "bm25_norm": bm25_s,
            "dpr_norm": dpr_s,
            "hybrid_score": hybrid_s,
        })

    # Sort by hybrid_score descending
    hybrid_results.sort(key=lambda r: -r["hybrid_score"])

    # Add ranks and cut to topk_hybrid
    for rank, r in enumerate(hybrid_results[:topk_hybrid], start=1):
        r["rank"] = rank

    return bm25_results, dpr_results, hybrid_results[:topk_hybrid]


def print_results(query, bm25_res, dpr_res, hybrid_res, id2rec, id_by_idx, max_snip_len=200):
    print("\n============================================================")
    print(f"Query: {query}")
    print("============================================================\n")

    print("Top BM25 results:")
    for r in bm25_res[:5]:
        obj = id2rec.get(r["id"], {})
        txt = obj.get("text", "")
        snip = txt[:max_snip_len].replace("\n", " ")
        print(f"  [{r['rank']}] {r['id']}  score={r['score']:.3f}")
        print(f"      {snip}")
    print()

    print("Top DPR results:")
    for r in dpr_res[:5]:
        pid = id_by_idx.get(r["idx"])
        obj = id2rec.get(pid, {})
        txt = obj.get("text", "")
        snip = txt[:max_snip_len].replace("\n", " ")
        print(f"  [{r['rank']}] {pid}  score={r['score']:.3f}")
        print(f"      {snip}")
    print()

    print("Top HYBRID results (BM25 + DPR):")
    for r in hybrid_res:
        obj = id2rec.get(r["id"], {})
        txt = obj.get("text", "")
        snip = txt[:max_snip_len].replace("\n", " ")
        print(f"  [{r['rank']}] {r['id']}  hybrid={r['hybrid_score']:.3f}  "
              f"(bm25={r['bm25_norm']:.3f}, dpr={r['dpr_norm']:.3f})")
        print(f"      {snip}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single query to run.")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode.")
    parser.add_argument("--alpha", type=float, default=ALPHA_BM25, help="BM25 weight in hybrid score (0-1).")
    parser.add_argument("--index-type", type=str, choices=["flat", "ivf", "hnsw"], default="flat",
                        help="Which FAISS DPR index to use.")
    args = parser.parse_args()

    # Adjust FAISS index path based on index-type
    index_file = {
        "flat": DPR_FAISS_FLAT,
        "ivf": DPR_FAISS_IVF,
        "hnsw": DPR_FAISS_HNSW,
    }[args.index_type]

    print("Loading passages map...")
    id2rec = load_passages_map(PASSAGES_JSONL)

    print("Loading BM25 index...")
    ids, tokenized_corpus, bm25 = load_bm25(BM25_PKL)
    print("BM25 loaded with", len(ids), "documents.")

    print("Loading DPR embeddings meta...")
    embs, meta, id_by_idx = load_dpr()
    print("DPR meta entries:", len(meta))

    print("Loading FAISS index:", index_file)
    faiss_index = load_faiss_index(index_file)

    print("Loading DPR query encoder:", DPR_MODEL_NAME)
    encoder = build_query_encoder()

    def run_one(q):
        bm25_res, dpr_res, hybrid_res = hybrid_search(
            q,
            ids,
            bm25,
            encoder,
            faiss_index,
            id_by_idx,
            alpha_bm25=args.alpha,
        )
        print_results(q, bm25_res, dpr_res, hybrid_res, id2rec, id_by_idx)

    if args.interactive:
        while True:
            q = input("\nEnter query (or 'exit'): ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break
            run_one(q)
    else:
        if not args.query:
            print("No query provided. Use --query or --interactive.")
            return
        run_one(args.query)


if __name__ == "__main__":
    main()

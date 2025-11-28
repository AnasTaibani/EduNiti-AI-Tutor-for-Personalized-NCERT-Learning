#!/usr/bin/env python3
"""
hybrid_retrieval_ft_demo.py

Demo: Hybrid retrieval (BM25 + fine-tuned DPR) over NCERT passages.

Uses:
  - BM25 index over NCERT_passages_hybrid (pickle)
  - Fine-tuned DPR model (SentenceTransformers)
  - Fine-tuned DPR FAISS HNSW index

Outputs:
  - Top BM25 hits
  - Top DPR hits
  - Top HYBRID hits (combined score)
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# BM25 built on chunks from NCERT_passages_hybrid
BM25_PKL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "bm25.pkl"
PASSAGES_JSONL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "passages.jsonl"

# Fine-tuned DPR model + FAISS index + meta
DPR_MODEL_DIR = PROJECT_ROOT / "Retrieval" / "dpr" / "models" / "ncert-dpr-v1"
DPR_INDEX_PATH = PROJECT_ROOT / "Retrieval" / "dpr" / "indexes_ft" / "dpr_ft_faiss_hnsw.index"
DPR_META_PATH = PROJECT_ROOT / "Retrieval" / "dpr" / "passages_dpr_meta.json"


# ---------- simple tokeniser ----------
def simple_tokenize(text: str):
    return re.findall(r"\w+", (text or "").lower())


# ---------- flexible BM25 loader ----------
def load_bm25(path: Path):
    """
    Try to support multiple file formats for bm25.pkl.

    For your current project, the important one is:
      {"bm25": ..., "tokenized_corpus": ..., "ids": [...]}

    We then reconstruct `records` by reading passages.jsonl and
    matching by id.
    """
    print(f"🔹 Loading BM25 from: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)

    from rank_bm25 import BM25Okapi  # ensure installed in your env

    bm25 = None
    corpus_tokens = None
    records = None

    # --- Case 1: tuple formats (older style, just in case) ---
    if isinstance(obj, tuple):
        if len(obj) == 3:
            bm25, corpus_tokens, records = obj
            print("   → Detected tuple format: (bm25, corpus_tokens, records)")
        elif len(obj) == 2:
            bm25, records = obj
            print("   → Detected tuple format: (bm25, records); rebuilding corpus_tokens from records['text']")
            corpus_tokens = []
            for r in records:
                txt = r.get("text", "") or ""
                corpus_tokens.append(simple_tokenize(txt))
        else:
            print(f"   ⚠️ Unexpected tuple length: {len(obj)}; treating as records list and rebuilding.")
            records = list(obj)
            corpus_tokens = []
            for r in records:
                txt = r.get("text", "") or ""
                corpus_tokens.append(simple_tokenize(txt))
            bm25 = BM25Okapi(corpus_tokens)

    # --- Case 2: dict formats (your current one) ---
    elif isinstance(obj, dict):
        keys = set(obj.keys())
        print(f"   → Detected dict format with keys: {keys}")

        # 🔴 Your current format: {'bm25', 'tokenized_corpus', 'ids'}
        if {"bm25", "tokenized_corpus", "ids"} <= keys:
            print("   → Dict with bm25 + tokenized_corpus + ids; rebuilding records from passages.jsonl")
            bm25 = obj["bm25"]
            corpus_tokens = obj["tokenized_corpus"]
            ids = obj["ids"]

            if not PASSAGES_JSONL.exists():
                raise FileNotFoundError(f"Expected passages JSONL at {PASSAGES_JSONL}, but it does not exist.")

            # Load all passage records into a dict by id
            id2record = {}
            with PASSAGES_JSONL.open("r", encoding="utf-8") as pf:
                for line in pf:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pid = rec.get("id")
                    if pid:
                        id2record[pid] = rec

            # Build records list in the same order as ids / corpus_tokens
            records = []
            missing = 0
            for pid in ids:
                rec = id2record.get(pid)
                if rec is None:
                    # If somehow not in JSONL, create a stub
                    missing += 1
                    rec = {"id": pid, "text": "", "grade": None, "subject": None, "chapter": None}
                records.append(rec)

            if missing > 0:
                print(f"   ⚠️ Warning: {missing} ids not found in {PASSAGES_JSONL}, filled with empty stubs.")

        # Other possible dict layouts
        elif {"bm25", "corpus_tokens", "records"} <= keys:
            bm25 = obj["bm25"]
            corpus_tokens = obj["corpus_tokens"]
            records = obj["records"]
        elif "records" in obj and "bm25" in obj:
            bm25 = obj["bm25"]
            records = obj["records"]
            print("   → Dict with bm25 + records; rebuilding corpus_tokens from records['text']")
            corpus_tokens = []
            for r in records:
                txt = r.get("text", "") or ""
                corpus_tokens.append(simple_tokenize(txt))
        elif "records" in obj:
            records = obj["records"]
            print("   → Dict with only records; rebuilding BM25 from records['text']")
            corpus_tokens = []
            for r in records:
                txt = r.get("text", "") or ""
                corpus_tokens.append(simple_tokenize(txt))
            bm25 = BM25Okapi(corpus_tokens)
        else:
            print("   ⚠️ Dict format not recognised; treating values as records list and rebuilding.")
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    records = v
                    break
            if records is None:
                raise ValueError("Could not locate records list in BM25 dict object.")
            corpus_tokens = []
            for r in records:
                txt = r.get("text", "") or ""
                corpus_tokens.append(simple_tokenize(txt))
            bm25 = BM25Okapi(corpus_tokens)

    # --- Case 3: list -> treat as records and rebuild ---
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        print("   → Detected list[dict]; treating as records and rebuilding BM25")
        records = obj
        corpus_tokens = []
        for r in records:
            txt = r.get("text", "") or ""
            corpus_tokens.append(simple_tokenize(txt))
        bm25 = BM25Okapi(corpus_tokens)
    else:
        raise ValueError(f"Unsupported bm25.pkl object type: {type(obj)}")

    if bm25 is None or corpus_tokens is None or records is None:
        raise ValueError("Failed to reconstruct BM25, corpus_tokens or records from pickle.")

    print(f"   → BM25 corpus size: {len(corpus_tokens)} passages")
    id2rec = {r["id"]: r for r in records}
    return bm25, corpus_tokens, records, id2rec


def bm25_search(query: str, bm25, corpus_tokens, records, topk: int = 5):
    q_tokens = simple_tokenize(query)
    scores = bm25.get_scores(q_tokens)  # numpy array
    idxs = np.argsort(scores)[::-1][:topk]

    results = []
    for rank, i in enumerate(idxs, start=1):
        rec = records[i]
        score = float(scores[i])
        results.append({
            "rank": rank,
            "id": rec["id"],
            "grade": rec.get("grade"),
            "subject": rec.get("subject"),
            "chapter": rec.get("chapter"),
            "score": score,
            "text": rec.get("text", "")[:400]
        })
    return results


# ---------- load DPR (fine-tuned) ----------
def load_ft_dpr(model_dir: Path, index_path: Path, meta_path: Path):
    print(f"🔹 Loading fine-tuned DPR model from: {model_dir}")
    model = SentenceTransformer(str(model_dir))

    print(f"🔹 Loading DPR FAISS HNSW index from: {index_path}")
    index = faiss.read_index(str(index_path))

    print(f"🔹 Loading DPR passage meta from: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    id2meta = {m["id"]: m for m in meta}
    print(f"   → Loaded meta for {len(meta)} passages.")

    return model, index, id2meta


def dpr_search(query: str, model, index, id2meta, id2rec_bm25, topk: int = 5):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, topk)
    D = D[0]
    I = I[0]

    results = []
    meta_keys = list(id2meta.keys())
    for rank, (dist, idx) in enumerate(zip(D, I), start=1):
        if idx < 0 or idx >= len(meta_keys):
            continue
        pid = meta_keys[idx]
        m = id2meta[pid]
        rec = id2rec_bm25.get(pid, {})
        text = rec.get("text", "")[:400]

        results.append({
            "rank": rank,
            "id": pid,
            "grade": m.get("grade"),
            "subject": m.get("subject"),
            "chapter": m.get("chapter"),
            "score": float(dist),
            "text": text
        })
    return results


# ---------- hybrid scoring ----------
def hybrid_merge(bm25_res, dpr_res, alpha: float = 0.5, topk: int = 10):
    bm25_scores = {r["id"]: r["score"] for r in bm25_res}
    dpr_scores = {r["id"]: r["score"] for r in dpr_res}

    def norm_scores(score_dict):
        if not score_dict:
            return {}
        vals = np.array(list(score_dict.values()), dtype="float32")
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax <= vmin:
            return {k: 1.0 for k in score_dict}
        return {k: (v - vmin) / (vmax - vmin) for k, v in score_dict.items()}

    bm25_norm = norm_scores(bm25_scores)
    dpr_norm = norm_scores(dpr_scores)

    all_ids = set(bm25_scores.keys()) | set(dpr_scores.keys())
    merged = []
    for pid in all_ids:
        b = bm25_norm.get(pid, 0.0)
        d = dpr_norm.get(pid, 0.0)
        h = alpha * b + (1 - alpha) * d
        merged.append((pid, h, b, d))

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:topk]


# ---------- pretty printing ----------
def print_results(query, bm25_res, dpr_res, hybrid_res, id2rec_bm25, id2meta, alpha):
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    print("\nTop BM25 results:")
    for r in bm25_res:
        print(f"  [{r['rank']}] {r['id']}  score={r['score']:.3f}")
        print(f"      {r['text']}\n")

    print("Top DPR (fine-tuned) results:")
    for r in dpr_res:
        print(f"  [{r['rank']}] {r['id']}  score={r['score']:.3f}")
        print(f"      {r['text']}\n")

    print(f"Top HYBRID results (alpha={alpha:.2f} BM25, {1-alpha:.2f} DPR):")
    for rank, (pid, h, b, d) in enumerate(hybrid_res, start=1):
        meta = id2meta.get(pid, {})
        rec = id2rec_bm25.get(pid, {})
        text = rec.get("text", "")[:400]

        print(f"  [{rank}] {pid}  hybrid={h:.3f}  (bm25={b:.3f}, dpr={d:.3f})")
        print(f"      grade={meta.get('grade')}, subject={meta.get('subject')}, chapter={meta.get('chapter')}")
        print(f"      {text}\n")


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single query to test. If not provided, runs an interactive loop.")
    parser.add_argument("--k_bm25", type=int, default=5)
    parser.add_argument("--k_dpr", type=int, default=5)
    parser.add_argument("--k_hybrid", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 in hybrid fusion (0..1)")
    args = parser.parse_args()

    # Load BM25 (robust)
    bm25, corpus_tokens, records, id2rec = load_bm25(BM25_PKL)

    # Load DPR fine-tuned components
    dpr_model, dpr_index, id2meta = load_ft_dpr(DPR_MODEL_DIR, DPR_INDEX_PATH, DPR_META_PATH)

    def run_one(q: str):
        bm25_res = bm25_search(q, bm25, corpus_tokens, records, topk=args.k_bm25)
        dpr_res = dpr_search(q, dpr_model, dpr_index, id2meta, id2rec, topk=args.k_dpr)
        hybrid = hybrid_merge(bm25_res, dpr_res, alpha=args.alpha, topk=args.k_hybrid)
        print_results(q, bm25_res, dpr_res, hybrid, id2rec, id2meta, args.alpha)

    if args.query:
        run_one(args.query)
    else:
        print("Enter queries (blank line to exit):")
        while True:
            try:
                q = input("\nQuery> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            run_one(q)


if __name__ == "__main__":
    main()

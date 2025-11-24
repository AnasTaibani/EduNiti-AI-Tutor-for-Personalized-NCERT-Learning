#!/usr/bin/env python3
"""
eval_bm25_gold.py

Task H — Retrieval evaluation, metrics, and acceptance testing (>90% relevance)

Evaluates your BM25 passage index against a curated gold evaluation set.

Inputs:
  - Preprocessing/NCERT_passages_hybrid/bm25.pkl
      (dict with keys: 'bm25', 'tokenized_corpus', 'ids')
  - Preprocessing/NCERT_passages_hybrid/passages.jsonl
      (one JSON per passage with id, grade, subject, chapter, text, ...)
  - Retrieval/eval/gold_eval_set.csv
      Columns:
        - query           (required)
        - expected_id     (required for scoring; skip row if missing)
        - grade           (optional filter)
        - subject         (optional filter)
        - chapter         (optional filter)

Outputs (to Retrieval/eval/results_bm25/):
  - per_query_results.csv      : one row per query@K
  - summary.json               : global metrics
  - summary.txt                : human-readable metrics
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi


# --- PATHS (adjust if your structure changes) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BM25_PKL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "bm25.pkl"
PASSAGES_JSONL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "passages.jsonl"
GOLD_CSV_DEFAULT = PROJECT_ROOT / "Retrieval" / "eval" / "gold_eval_set.csv"
OUT_DIR = PROJECT_ROOT / "Retrieval" / "eval" / "results_bm25"


# --- LOADING HELPERS --------------------------------------------------------


def load_bm25(bm25_path: Path):
    """Load BM25 from your existing pickle format."""
    import pickle

    print(f"🔹 Loading BM25 from: {bm25_path}")
    with bm25_path.open("rb") as f:
        obj = pickle.load(f)

    # Expect dict with keys: bm25, tokenized_corpus, ids
    if isinstance(obj, dict) and {"bm25", "tokenized_corpus", "ids"} <= set(obj.keys()):
        bm25 = obj["bm25"]
        corpus_tokens = obj["tokenized_corpus"]
        ids = obj["ids"]
        print(f"   → Detected dict format: {{'bm25','tokenized_corpus','ids'}}")
        print(f"   → Corpus size: {len(corpus_tokens)} passages")
        return bm25, corpus_tokens, ids

    raise ValueError("Unexpected BM25 pickle format; expected dict with keys 'bm25','tokenized_corpus','ids'.")


def load_passage_meta(passages_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load passages.jsonl and build id -> metadata dict
    (including grade, subject, chapter, text).
    """
    print(f"🔹 Loading passage metadata from: {passages_path}")
    id2meta: Dict[str, Dict[str, Any]] = {}
    with passages_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("id")
            if not pid:
                continue
            id2meta[pid] = obj
    print(f"   → Loaded meta for {len(id2meta)} passages.")
    return id2meta


def load_gold_eval(gold_csv: Path) -> List[Dict[str, str]]:
    """
    Load gold evaluation CSV. Expected at least 'query' and 'expected_id'.
    Optional filters: grade, subject, chapter.
    """
    print(f"🔹 Loading gold eval set from: {gold_csv}")
    rows = []
    with gold_csv.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            q = (r.get("query") or "").strip()
            exp = (r.get("expected_id") or "").strip()
            if not q or not exp:
                # skip rows without expected_id (not labeled yet)
                continue
            rows.append({
                "query": q,
                "expected_id": exp,
                "grade": (r.get("grade") or "").strip(),
                "subject": (r.get("subject") or "").strip(),
                "chapter": (r.get("chapter") or "").strip(),
            })
    print(f"   → Loaded {len(rows)} labeled queries.")
    return rows


# --- RETRIEVAL / METRICS ----------------------------------------------------


def bm25_scores_for_query(
    query: str,
    bm25: BM25Okapi,
    corpus_tokens: List[List[str]],
) -> np.ndarray:
    """
    Compute BM25 scores for a query over the full corpus.
    NOTE: We rely on the same tokenizer used when building BM25.
          The pickled BM25 object handles tokenization internally,
          so we pass raw query string to .get_scores().
    """
    scores = bm25.get_scores(query)
    return np.array(scores, dtype=float)


def apply_filter_mask(
    ids: List[str],
    id2meta: Dict[str, Dict[str, Any]],
    grade: str,
    subject: str,
    chapter: str,
) -> np.ndarray:
    """
    Build a boolean mask over ids based on optional filters.
    Empty filter strings mean 'no filter' for that field.
    """
    n = len(ids)
    mask = np.ones(n, dtype=bool)

    def match(val: str, target: str) -> bool:
        if not target:
            return True
        return (val or "").strip() == target

    if grade or subject or chapter:
        for i, pid in enumerate(ids):
            meta = id2meta.get(pid, {})
            g = meta.get("grade", "")
            s = meta.get("subject", "")
            c = meta.get("chapter", "")
            if not (match(g, grade) and match(s, subject) and match(c, chapter)):
                mask[i] = False

    return mask


def evaluate_bm25(
    gold_rows: List[Dict[str, str]],
    bm25: BM25Okapi,
    corpus_tokens: List[List[str]],
    ids: List[str],
    id2meta: Dict[str, Dict[str, Any]],
    ks: Tuple[int, ...] = (1, 3, 5, 10),
):
    """
    For each gold query, run BM25 and compute:
      - hit rank of expected_id (or -1 if not in rankings)
      - hit@k for each k
      - MRR (mean reciprocal rank)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_query_rows = []
    total = len(gold_rows)

    # For summary metrics
    hit_counts = {k: 0 for k in ks}
    rr_sum = 0.0

    print("🔹 Running BM25 evaluation on gold queries...")
    for row in tqdm(gold_rows):
        q = row["query"]
        expected_id = row["expected_id"]
        f_grade = row["grade"]
        f_subject = row["subject"]
        f_chapter = row["chapter"]

        scores = bm25_scores_for_query(q, bm25, corpus_tokens)

        # apply filter mask if any filters present
        mask = apply_filter_mask(ids, id2meta, f_grade, f_subject, f_chapter)
        if mask.sum() == 0:
            # no documents match filter; treat as miss
            hit_rank = -1
            hit_scores = {k: 0 for k in ks}
            best_id = ""
            best_score = float("nan")
        else:
            # mask scores: set invalid ones very low
            masked_scores = scores.copy()
            masked_scores[~mask] = -1e9

            # sort indices by score descending
            sorted_idx = np.argsort(-masked_scores)

            # find rank of expected_id (1-based)
            hit_rank = -1
            for rank, idx in enumerate(sorted_idx, start=1):
                if ids[idx] == expected_id:
                    hit_rank = rank
                    break

            # compute hit@k
            hit_scores = {}
            for k in ks:
                if hit_rank != -1 and hit_rank <= k:
                    hit_scores[k] = 1
                else:
                    hit_scores[k] = 0

            # best doc info
            best_idx = sorted_idx[0]
            best_id = ids[best_idx]
            best_score = float(masked_scores[best_idx])

        # RR
        rr = 1.0 / hit_rank if hit_rank > 0 else 0.0
        rr_sum += rr

        for k in ks:
            hit_counts[k] += hit_scores[k]

        per_query_rows.append({
            "query": q,
            "expected_id": expected_id,
            "grade_filter": f_grade,
            "subject_filter": f_subject,
            "chapter_filter": f_chapter,
            "hit_rank": hit_rank,
            "rr": rr,
            "hit@1": hit_scores[1],
            "hit@3": hit_scores[3],
            "hit@5": hit_scores[5],
            "hit@10": hit_scores[10],
            "best_id": best_id,
            "best_score": best_score,
        })

    # --- summary metrics ---
    summary = {}
    for k in ks:
        summary[f"recall@{k}"] = hit_counts[k] / total if total else 0.0
    summary["MRR"] = rr_sum / total if total else 0.0
    summary["num_queries"] = total

    return per_query_rows, summary


# --- MAIN -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        type=str,
        default=str(GOLD_CSV_DEFAULT),
        help="Path to gold_eval_set.csv (with query, expected_id, optional filters).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Primary K for acceptance threshold check (default: 5).",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)

    if not BM25_PKL.exists():
        print("ERROR: BM25 pickle not found at:", BM25_PKL)
        return
    if not PASSAGES_JSONL.exists():
        print("ERROR: passages.jsonl not found at:", PASSAGES_JSONL)
        return
    if not gold_path.exists():
        print("ERROR: gold eval CSV not found at:", gold_path)
        return

    bm25, corpus_tokens, ids = load_bm25(BM25_PKL)
    id2meta = load_passage_meta(PASSAGES_JSONL)
    gold_rows = load_gold_eval(gold_path)

    per_query_rows, summary = evaluate_bm25(
        gold_rows,
        bm25,
        corpus_tokens,
        ids,
        id2meta,
        ks=(1, 3, 5, 10),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write per-query CSV
    per_query_csv = OUT_DIR / "per_query_results_bm25.csv"
    with per_query_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "expected_id",
                "grade_filter",
                "subject_filter",
                "chapter_filter",
                "hit_rank",
                "rr",
                "hit@1",
                "hit@3",
                "hit@5",
                "hit@10",
                "best_id",
                "best_score",
            ],
        )
        writer.writeheader()
        for r in per_query_rows:
            writer.writerow(r)

    # Write JSON summary
    summary_json = OUT_DIR / "summary_bm25.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Human-readable summary
    summary_txt = OUT_DIR / "summary_bm25.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("=== BM25 Evaluation Summary (Gold Set) ===\n")
        f.write(f"Num queries: {summary['num_queries']}\n")
        f.write(f"Recall@1  : {summary['recall@1']:.3f}\n")
        f.write(f"Recall@3  : {summary['recall@3']:.3f}\n")
        f.write(f"Recall@5  : {summary['recall@5']:.3f}\n")
        f.write(f"Recall@10 : {summary['recall@10']:.3f}\n")
        f.write(f"MRR       : {summary['MRR']:.3f}\n")
        # acceptance check
        acc_k = args.k
        r_at_k = summary.get(f"recall@{acc_k}", 0.0)
        passed = r_at_k >= 0.90
        f.write(f"\nAcceptance criterion: Recall@{acc_k} >= 0.90\n")
        f.write(f"Result: Recall@{acc_k} = {r_at_k:.3f} → {'PASS ✅' if passed else 'FAIL ❌'}\n")

    print("\n✅ Evaluation complete.")
    print("Per-query results saved to:", per_query_csv)
    print("Summary JSON saved to    :", summary_json)
    print("Summary TXT saved to     :", summary_txt)
    acc_k = args.k
    r_at_k = summary.get(f"recall@{acc_k}", 0.0)
    print(f"\n➡️ Acceptance check (Recall@{acc_k} >= 0.90): {r_at_k:.3f} → {'PASS ✅' if r_at_k >= 0.90 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()

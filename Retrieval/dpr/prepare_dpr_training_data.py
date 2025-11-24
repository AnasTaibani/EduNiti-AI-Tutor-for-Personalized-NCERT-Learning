#!/usr/bin/env python3
"""
prepare_dpr_training_data.py

Use your curated gold_eval_set.csv + passages.jsonl
to build training pairs for DPR fine-tuning.

Inputs:
  - Preprocessing/NCERT_passages_hybrid/passages.jsonl
      (one JSON object per line, with "id" and "text")
  - Retrieval/eval/gold_eval_set.csv
      columns: query, expected_id  (others are ignored)

Output:
  - Retrieval/dpr/train_pairs.jsonl
      one JSON per line: { "query": ..., "passage_id": ..., "passage_text": ... }
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PASSAGES_JSONL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "passages.jsonl"
GOLD_EVAL_CSV = PROJECT_ROOT / "Retrieval" / "eval" / "gold_eval_set.csv"
OUT_TRAIN_JSONL = PROJECT_ROOT / "Retrieval" / "dpr" / "train_pairs.jsonl"


def load_passages(jsonl_path: Path):
    """
    Build a dict: id -> text from passages.jsonl
    """
    id2text = {}
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
            text = obj.get("text")
            if pid and text:
                id2text[pid] = text
    return id2text


def load_gold_pairs(csv_path: Path):
    """
    Read gold_eval_set.csv with at least columns:
      - query
      - expected_id
    Returns list of (query, expected_id)
    """
    pairs = []
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            q = (row.get("query") or "").strip()
            eid = (row.get("expected_id") or "").strip()
            if not q or not eid:
                continue
            pairs.append((q, eid))
    return pairs


def main():
    if not PASSAGES_JSONL.exists():
        print("ERROR: passages.jsonl not found at", PASSAGES_JSONL)
        return
    if not GOLD_EVAL_CSV.exists():
        print("ERROR: gold_eval_set.csv not found at", GOLD_EVAL_CSV)
        return

    print("Loading passages from:", PASSAGES_JSONL)
    id2text = load_passages(PASSAGES_JSONL)
    print("Total passages loaded:", len(id2text))

    print("Loading gold query–passage pairs from:", GOLD_EVAL_CSV)
    qpairs = load_gold_pairs(GOLD_EVAL_CSV)
    print("Total gold pairs:", len(qpairs))

    if not qpairs:
        print("No valid rows with query+expected_id. Fill gold_eval_set.csv first.")
        return

    # Build train_pairs.jsonl
    missing = 0
    count = 0
    with OUT_TRAIN_JSONL.open("w", encoding="utf-8") as out_f:
        for query, pid in qpairs:
            text = id2text.get(pid)
            if not text:
                missing += 1
                continue
            rec = {
                "query": query,
                "passage_id": pid,
                "passage_text": text,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ Wrote {count} training pairs to {OUT_TRAIN_JSONL}")
    if missing > 0:
        print(f"⚠️ {missing} pairs skipped (expected_id not found in passages.jsonl)")


if __name__ == "__main__":
    main()

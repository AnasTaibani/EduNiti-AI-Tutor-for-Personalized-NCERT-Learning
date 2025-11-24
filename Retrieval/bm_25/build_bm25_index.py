#!/usr/bin/env python3
"""
build_bm25_index.py

Task D (part 1) — BM25 index build over NCERT passages.

Inputs:
  - Preprocessing/NCERT_passages_hybrid/passages.jsonl
    Each line: {"id", "text", "grade", "subject", "chapter", ...}

Outputs:
  - Preprocessing/NCERT_passages_hybrid/bm25.pkl
    Contains:
      {
        "ids": [passage_id_0, passage_id_1, ...],
        "tokenized_corpus": [...],
        "bm25": BM25Okapi instance
      }

Usage (from project root):
  cd D:\eduniti-majorProject
  python Retrieval\\bm25\\build_bm25_index.py
"""

import json
import pickle
from pathlib import Path
import re
from tqdm import tqdm

from rank_bm25 import BM25Okapi

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PASSAGES_JSONL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "passages.jsonl"
BM25_OUT = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "bm25.pkl"


def simple_tokenize(text: str):
    """
    Simple tokenizer: lowercase + word chars.
    This mirrors what we used in preprocessing / BM25 earlier.
    """
    text = (text or "").lower()
    return re.findall(r"\w+", text)


def load_passages(jsonl_path: Path):
    if not jsonl_path.exists():
        raise FileNotFoundError(f"passages.jsonl not found at: {jsonl_path.resolve()}")

    ids = []
    tokenized_corpus = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading passages"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            pid = obj.get("id")
            text = obj.get("text", "")

            if not pid or not text.strip():
                continue

            ids.append(pid)
            tokenized_corpus.append(simple_tokenize(text))

    return ids, tokenized_corpus


def main():
    print("Building BM25 index for NCERT passages...")
    print("Input:", PASSAGES_JSONL)

    ids, tokenized_corpus = load_passages(PASSAGES_JSONL)
    print(f"Loaded {len(ids)} passages to index with BM25.")

    if not ids:
        print("No passages loaded, aborting BM25 build.")
        return

    print("Fitting BM25Okapi...")
    bm25 = BM25Okapi(tokenized_corpus)

    data = {
        "ids": ids,
        "tokenized_corpus": tokenized_corpus,
        "bm25": bm25,
    }

    with BM25_OUT.open("wb") as f:
        pickle.dump(data, f)

    print(f"BM25 index saved to: {BM25_OUT}")
    print("Task D (BM25 build) complete.")


if __name__ == "__main__":
    main()

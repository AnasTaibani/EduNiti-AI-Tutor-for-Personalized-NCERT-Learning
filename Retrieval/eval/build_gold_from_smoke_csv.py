#!/usr/bin/env python3
"""
build_gold_from_smoke_csv.py

Builds a clean gold_eval_set.csv from a smoke-test-based CSV.

Assumptions:
- Input CSV has at least: query + some passage ID columns.
- Optional rank columns: 'hybrid_rank', 'rank', 'bm25_rank', 'vector_rank'.
- Passage id may be in one of: 'expected_id', 'doc_id', 'id', 'passage_id'.

Logic:
- Group rows by `query`.
- If any row for that query already has `expected_id`, use the first non-empty one.
- Otherwise:
    - Pick the row with the best (lowest) rank using priority:
        hybrid_rank > rank > bm25_rank > vector_rank.
    - Use its passage id as expected_id.
- Output CSV: query, expected_id
"""

import csv
from pathlib import Path
from collections import defaultdict

# 👉 Adjust these paths to match your project layout
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "Retrieval" / "eval" / "gold_eval_filled_from_smoke.csv"
OUTPUT_CSV = PROJECT_ROOT / "Retrieval" / "eval" / "gold_eval_set.csv"

# If your input file has a different name, change INPUT_CSV above.


def parse_int(value, default=9999):
    try:
        if value is None:
            return default
        value = str(value).strip()
        if not value:
            return default
        return int(value)
    except Exception:
        return default


def choose_best_row(rows):
    """
    Given all rows for a single query, pick the 'best' row based on rank.
    Priority order of rank columns:
      - hybrid_rank
      - rank
      - bm25_rank
      - vector_rank
    """
    def score_row(r):
        for col in ["hybrid_rank", "rank", "bm25_rank", "vector_rank"]:
            if col in r:
                s = parse_int(r.get(col))
                if s != 9999:
                    return s
        # fallback: keep them at the end
        return 9999

    best = min(rows, key=score_row)
    return best


def get_passage_id(row):
    """
    Try different column names to find a passage id.
    Adjust if your CSV uses other names.
    """
    for col in ["expected_id", "doc_id", "id", "passage_id"]:
        val = row.get(col)
        if val and str(val).strip():
            return str(val).strip()
    return None


def main():
    if not INPUT_CSV.exists():
        print("❌ Input CSV not found at:", INPUT_CSV)
        return

    print("➡️ Reading smoke-based CSV from:", INPUT_CSV)

    rows_by_query = defaultdict(list)

    with INPUT_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        print("Detected columns:", fieldnames)
        for row in reader:
            q = (row.get("query") or "").strip()
            if not q:
                continue
            rows_by_query[q].append(row)

    print("Total unique queries:", len(rows_by_query))

    out_rows = []
    missing_id_queries = 0

    for q, rows in rows_by_query.items():
        # 1) If any row already has expected_id, prefer that
        explicit_expected = None
        for r in rows:
            ev = (r.get("expected_id") or "").strip()
            if ev:
                explicit_expected = ev
                break

        if explicit_expected:
            expected_id = explicit_expected
        else:
            # 2) Otherwise, pick best-ranked row
            best = choose_best_row(rows)
            expected_id = get_passage_id(best)

        if not expected_id:
            missing_id_queries += 1
            # still write the query with empty expected_id to review manually
            out_rows.append({"query": q, "expected_id": ""})
        else:
            out_rows.append({"query": q, "expected_id": expected_id})

    # Write output gold eval CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "expected_id"])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print("✅ Wrote gold eval set to:", OUTPUT_CSV)
    print("   Total queries:", len(out_rows))
    if missing_id_queries:
        print(f"⚠️ {missing_id_queries} queries had no detectable passage id; "
              f"their expected_id is empty and should be filled manually.")


if __name__ == "__main__":
    main()

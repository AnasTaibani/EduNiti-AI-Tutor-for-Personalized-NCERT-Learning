#!/usr/bin/env python3
"""
fill_gold_eval_from_smoke.py

Use smoke test results to auto-fill `query` and `expected_id` columns
in the gold evaluation CSV.

- Reads:
    Retrieval/eval/gold_eval_template.csv
    Preprocessing/NCERT_processed/smoke_test_results/smoke_results.json   (adjustable)

- For each smoke-test entry:
    - query           = entry["query"]
    - expected_id     = entry["bm25"][0]["id"]  (BM25 rank-1 document)
    - grade,subject,chapter parsed from expected_id
    - Finds first row in template with matching (grade, subject, chapter)
      where query=="" and expected_id=="" and fills them.
    - If no such row exists, appends a new one.

- Writes:
    Retrieval/eval/gold_eval_filled_from_smoke.csv
"""

import json
import pandas as pd
from pathlib import Path

# --------- CONFIG (adjust paths here if different) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # points to eduniti-majorProject

TEMPLATE_CSV = Path("gold_eval_template.csv")
OUT_CSV      = Path("gold_eval_filled_from_smoke.csv")

# default location where smoke_results.json was written earlier
SMOKE_JSON   = Path ("D:/eduniti-majorProject/Preprocessing/NCERT_processed/smoke_test_results/smoke_results.json")

# If you want to use vector rank-1 instead of BM25 rank-1, set this to True
USE_VECTOR_TOP1 = False
# -----------------------------------------------------------


def parse_id_parts(passage_id: str):
    """
    Expected id format: Grade_7___Science___Chapter_10___p0019
    Returns (grade, subject, chapter).
    """
    parts = passage_id.split("___")
    if len(parts) < 3:
        return None, None, None
    grade = parts[0]     # e.g., Grade_7
    subject = parts[1]   # e.g., Science
    chapter = parts[2]   # e.g., Chapter_10
    return grade, subject, chapter


def load_smoke_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Smoke results JSON not found at: {path.resolve()}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def main():
    print("➡️ Filling gold eval template from smoke test results...")

    if not TEMPLATE_CSV.exists():
        raise FileNotFoundError(f"Template CSV not found at: {TEMPLATE_CSV.resolve()}")

    df = pd.read_csv(TEMPLATE_CSV, encoding="utf-8")

    # Ensure expected columns exist
    for col in ["query", "expected_id", "grade", "subject", "chapter",
                "suggested_passage_id", "source_file", "notes"]:
        if col not in df.columns:
            # Create if missing (minimal case)
            df[col] = ""

    smoke_data = load_smoke_results(SMOKE_JSON)

    filled_count = 0
    appended_count = 0

    for entry in smoke_data:
        q = entry.get("query", "").strip()
        if not q:
            continue

        bm25_list = entry.get("bm25") or []
        vec_list = entry.get("vector") or []

        if USE_VECTOR_TOP1:
            if not vec_list:
                # no vector results; skip
                continue
            top = vec_list[0]
        else:
            if not bm25_list:
                # no bm25 results; skip
                continue
            top = bm25_list[0]

        expected_id = top.get("id")
        if not expected_id:
            continue

        grade, subject, chapter = parse_id_parts(expected_id)
        if not grade:
            continue

        # Try to find a row to fill: same (grade, subject, chapter) and empty query/expected_id
        mask = (
            (df["grade"] == grade) &
            (df["subject"] == subject) &
            (df["chapter"] == chapter) &
            (df["query"].isna() | (df["query"] == "")) &
            (df["expected_id"].isna() | (df["expected_id"] == ""))
        )

        if mask.any():
            idx = df[mask].index[0]
            df.at[idx, "query"] = q
            df.at[idx, "expected_id"] = expected_id
            filled_count += 1
        else:
            # Append a new row if no slot exists
            new_row = {
                "query": q,
                "expected_id": expected_id,
                "grade": grade,
                "subject": subject,
                "chapter": chapter,
                "suggested_passage_id": expected_id,
                "source_file": top.get("path", ""),
                "notes": "auto-added from smoke results",
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            appended_count += 1

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"✅ Done. Filled {filled_count} existing rows, appended {appended_count} new rows.")
    print(f"   Output written to: {OUT_CSV}")


if __name__ == "__main__":
    main()

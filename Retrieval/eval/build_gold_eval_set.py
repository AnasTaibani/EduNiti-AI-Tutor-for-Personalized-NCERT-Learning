#!/usr/bin/env python3
"""
build_gold_eval_set.py

Task A — Syllabus mapping & gold evaluation set creation.

This script:
  - Loads passage-level metadata from:
        Preprocessing/NCERT_processed/passages_meta_lang.csv
  - Groups by (grade, subject, chapter)
  - For each chapter, creates a few blank rows for you to fill queries + expected passage IDs.
  - Suggests a canonical passage_id per chapter (first chunk) to help mapping.

Output:
  Retrieval/eval/gold_eval_template.csv

Usage (from project root):
  cd D:\eduniti-majorProject
  python Retrieval\eval\build_gold_eval_set.py
"""

import csv
from pathlib import Path
import pandas as pd

# ---------- CONFIG ----------
# relative to project root (eduniti-majorProject)
META_CSV = Path("D:/eduniti-majorProject/Preprocessing/NCERT_processed/passages_meta_lang.csv")

OUT_DIR = Path("Retrieval/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "gold_eval_template.csv"

# how many blank query rows per chapter you want to pre-create
N_QUERIES_PER_CHAPTER = 3


def load_meta(meta_csv: Path) -> pd.DataFrame:
    """Load metadata CSV and normalize a 'source_file' column."""
    if not meta_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found at: {meta_csv.resolve()}")

    df = pd.read_csv(meta_csv, encoding="utf-8")

    # required columns
    required_cols = {"id", "grade", "subject", "chapter"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing required columns: {missing}")

    # normalize a 'source_file' column from whatever exists
    if "source" in df.columns:
        df["source_file"] = df["source"]
    elif "source_path" in df.columns:
        df["source_file"] = df["source_path"]
    elif "path" in df.columns:
        df["source_file"] = df["path"]
    else:
        df["source_file"] = ""

    return df   # <-- IMPORTANT: we must return df


def build_template(df: pd.DataFrame, n_queries_per_chapter: int = N_QUERIES_PER_CHAPTER):
    """
    Group by (grade, subject, chapter), and for each chapter:
      - pick the first passage as suggested_passage_id
      - emit n_queries_per_chapter blank rows for manual query filling
    """
    rows = []

    # sort so template is nicely ordered
    df_sorted = df.sort_values(["grade", "subject", "chapter", "id"])

    grouped = df_sorted.groupby(["grade", "subject", "chapter"])

    for (grade, subject, chapter), gdf in grouped:
        # choose first passage in that chapter as "suggested"
        first_row = gdf.iloc[0]
        suggested_passage_id = first_row["id"]
        source_file = first_row.get("source_file", "")

        for _ in range(n_queries_per_chapter):
            rows.append({
                "query": "",                       # you will fill
                "expected_id": "",                 # you will fill
                "grade": grade,
                "subject": subject,
                "chapter": chapter,
                "suggested_passage_id": suggested_passage_id,
                "source_file": source_file,
                "notes": ""                        # optional comments
            })

    return rows


def write_template(rows, out_csv: Path):
    if not rows:
        print("No rows to write – check metadata input.")
        return

    fieldnames = [
        "query",
        "expected_id",
        "grade",
        "subject",
        "chapter",
        "suggested_passage_id",
        "source_file",
        "notes",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ Wrote gold evaluation template to: {out_csv}")
    print("Next steps:")
    print("  1) Open this CSV in Excel / Google Sheets.")
    print("  2) For selected rows, fill in:")
    print("       - 'query' (a realistic student-style question)")
    print("       - 'expected_id' (the passage id that best answers it)")
    print("  3) Save your curated file as 'gold_eval_set.csv' in the same folder.")


def main():
    print("➡️ Building gold evaluation template (Task A — syllabus mapping & gold set)...")
    df = load_meta(META_CSV)
    rows = build_template(df, n_queries_per_chapter=N_QUERIES_PER_CHAPTER)
    write_template(rows, OUT_CSV)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
eval_week3_rag.py

Week 3 – Task 6: Internal Testing With 50 Multigrade Queries

This script:
  - Runs 50 predefined NCERT-style queries across multiple grades & subjects
  - Uses the RAG pipeline via RAGResponseGenerator (BM25 + reranker + LLM)
  - Stores outputs as:
      - Model/eval/week3_rag_results.json  (full raw results)
      - Model/eval/week3_rag_results.csv   (flat table for manual evaluation)

Usage (from project root):
  retrieval_env> python Model/eval_week3_rag.py

Prerequisites:
  - FastAPI BM25 server running (uvicorn Retrieval.api.main:app --reload)
  - OPENAI_API_KEY set
  - retrieval_env activated
"""

import csv
import json
import sys
from pathlib import Path
from datetime import datetime

# ----------------------------------------------------------------------
# Import generator (RAGResponseGenerator)
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "Model"

if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from generator import RAGResponseGenerator  # type: ignore


# ----------------------------------------------------------------------
# 50 multi-grade test queries
# ----------------------------------------------------------------------

TEST_QUERIES = [
    # ---- Science (6–8) ----
    {"id": 1, "query": "What is photosynthesis?", "grade": "Grade_7", "subject": "Science", "language": "English"},
    {"id": 2, "query": "Define germination.", "grade": "Grade_7", "subject": "Science", "language": "English"},
    {"id": 3, "query": "What is friction?", "grade": "Grade_8", "subject": "Science", "language": "English"},
    {"id": 4, "query": "What are biodegradable materials?", "grade": "Grade_8", "subject": "Science", "language": "English"},
    {"id": 5, "query": "What is a constellation?", "grade": "Grade_8", "subject": "Science", "language": "English"},

    # ---- Mathematics (6–10) ----
    {"id": 6, "query": "What is an integer?", "grade": "Grade_6", "subject": "Mathematics", "language": "English"},
    {"id": 7, "query": "Explain Pythagoras theorem.", "grade": "Grade_10", "subject": "Mathematics", "language": "English"},
    {"id": 8, "query": "What is an algebraic expression?", "grade": "Grade_7", "subject": "Mathematics", "language": "English"},
    {"id": 9, "query": "What is a linear equation in one variable?", "grade": "Grade_8", "subject": "Mathematics", "language": "English"},
    {"id": 10, "query": "What is the area of a trapezium?", "grade": "Grade_8", "subject": "Mathematics", "language": "English"},

    # ---- Social Science (History / Civics) ----
    {"id": 11, "query": "What were the main causes of the French Revolution?", "grade": "Grade_9", "subject": "Social_Science", "language": "English"},
    {"id": 12, "query": "What is democracy?", "grade": "Grade_9", "subject": "Social_Science", "language": "English"},
    {"id": 13, "query": "What are fundamental rights in the Indian Constitution?", "grade": "Grade_8", "subject": "Social_Science", "language": "English"},
    {"id": 14, "query": "What is globalization?", "grade": "Grade_10", "subject": "Social_Science", "language": "English"},
    {"id": 15, "query": "What is the role of Parliament in India?", "grade": "Grade_8", "subject": "Social_Science", "language": "English"},

    # ---- English ----
    {"id": 16, "query": "What is a metaphor?", "grade": "Grade_8", "subject": "English", "language": "English"},
    {"id": 17, "query": "What is the central idea of the poem 'The Road Not Taken'?", "grade": "Grade_9", "subject": "English", "language": "English"},
    {"id": 18, "query": "What is a noun clause?", "grade": "Grade_9", "subject": "English", "language": "English"},
    {"id": 19, "query": "Summarise the chapter 'A Letter to God'.", "grade": "Grade_10", "subject": "English", "language": "English"},
    {"id": 20, "query": "What is a formal letter?", "grade": "Grade_8", "subject": "English", "language": "English"},

    # ---- Hindi ----
    {"id": 21, "query": "लोकतंत्र क्या है?", "grade": "Grade_8", "subject": "Hindi", "language": "Hindi"},
    {"id": 22, "query": "‘सच्चाई’ पाठ का सारांश लिखिए।", "grade": "Grade_7", "subject": "Hindi", "language": "Hindi"},
    {"id": 23, "query": "कर्ता और कर्म क्या होते हैं?", "grade": "Grade_8", "subject": "Hindi", "language": "Hindi"},
    {"id": 24, "query": "लोक कथाएँ क्या होती हैं?", "grade": "Grade_6", "subject": "Hindi", "language": "Hindi"},
    {"id": 25, "query": "‘दादी माँ’ पाठ का मुख्य विचार बताइए।", "grade": "Grade_6", "subject": "Hindi", "language": "Hindi"},

    # ---- Sanskrit ----
    {"id": 26, "query": "पर्यावरण संरक्षण क्या है?", "grade": "Grade_10", "subject": "Sanskrit", "language": "Sanskrit"},
    {"id": 27, "query": "नदी वर्णन के मुख्य बिंदु लिखिए।", "grade": "Grade_9", "subject": "Sanskrit", "language": "Sanskrit"},
    {"id": 28, "query": "संस्कृत में उपसर्ग क्या है?", "grade": "Grade_8", "subject": "Sanskrit", "language": "Sanskrit"},
    {"id": 29, "query": "धातु किसे कहते हैं?", "grade": "Grade_9", "subject": "Sanskrit", "language": "Sanskrit"},
    {"id": 30, "query": "संयुक्त वाक्य का उदाहरण दीजिए।", "grade": "Grade_10", "subject": "Sanskrit", "language": "Sanskrit"},

    # ---- Economics (9–12) ----
    {"id": 31, "query": "What is GDP?", "grade": "Grade_12", "subject": "Economics", "language": "English"},
    {"id": 32, "query": "What is inflation?", "grade": "Grade_12", "subject": "Economics", "language": "English"},
    {"id": 33, "query": "What is human capital?", "grade": "Grade_11", "subject": "Economics", "language": "English"},
    {"id": 34, "query": "What is poverty line?", "grade": "Grade_9", "subject": "Economics", "language": "English"},
    {"id": 35, "query": "What is credit in economics?", "grade": "Grade_10", "subject": "Economics", "language": "English"},

    # ---- Accountancy (11–12) ----
    {"id": 36, "query": "What is a ledger in accountancy?", "grade": "Grade_11", "subject": "Accountancy", "language": "English"},
    {"id": 37, "query": "Explain the double entry system of accounting.", "grade": "Grade_11", "subject": "Accountancy", "language": "English"},
    {"id": 38, "query": "What is a journal entry?", "grade": "Grade_11", "subject": "Accountancy", "language": "English"},
    {"id": 39, "query": "What is a trial balance?", "grade": "Grade_11", "subject": "Accountancy", "language": "English"},
    {"id": 40, "query": "What is depreciation?", "grade": "Grade_11", "subject": "Accountancy", "language": "English"},

    # ---- Physics (11–12) ----
    {"id": 41, "query": "What is velocity?", "grade": "Grade_11", "subject": "Physics", "language": "English"},
    {"id": 42, "query": "State Newton's second law of motion.", "grade": "Grade_11", "subject": "Physics", "language": "English"},
    {"id": 43, "query": "What is Ohm's law?", "grade": "Grade_12", "subject": "Physics", "language": "English"},
    {"id": 44, "query": "Define momentum.", "grade": "Grade_11", "subject": "Physics", "language": "English"},
    {"id": 45, "query": "What is pressure?", "grade": "Grade_8", "subject": "Science", "language": "English"},

    # ---- Environment / Geography ----
    {"id": 46, "query": "What is greenhouse effect?", "grade": "Grade_9", "subject": "Science", "language": "English"},
    {"id": 47, "query": "What is the water cycle?", "grade": "Grade_6", "subject": "Science", "language": "English"},
    {"id": 48, "query": "What are natural resources?", "grade": "Grade_8", "subject": "Geography", "language": "English"},
    {"id": 49, "query": "What is agriculture?", "grade": "Grade_8", "subject": "Geography", "language": "English"},
    {"id": 50, "query": "What is weathering of rocks?", "grade": "Grade_9", "subject": "Geography", "language": "English"},
]


# ----------------------------------------------------------------------
# Main evaluation runner
# ----------------------------------------------------------------------

def run_evaluation():
    eval_dir = MODEL_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    json_path = eval_dir / "week3_rag_results.json"
    csv_path = eval_dir / "week3_rag_results.csv"

    print(f"[EVAL] Project root: {PROJECT_ROOT}")
    print(f"[EVAL] Saving JSON to: {json_path}")
    print(f"[EVAL] Saving CSV  to: {csv_path}")
    print()

    # Initialise generator (this will also bring up RAGPipeline, reranker, etc.)
    gen = RAGResponseGenerator()

    results = []
    total = len(TEST_QUERIES)

    for idx, case in enumerate(TEST_QUERIES, start=1):
        qid = case["id"]
        query = case["query"]
        grade = case["grade"]
        subject = case["subject"]
        language = case.get("language", "English")

        print(f"[EVAL] ({idx}/{total}) Q{qid}: {query}")
        try:
            rag_result = gen.generate_response(
                question=query,
                grade=grade,
                subject=subject,
                chapter=None,
                language=language,
                top_k=5,
                bm25_candidates=20,
            )
        except Exception as e:
            print(f"  -> ERROR for Q{qid}: {e}")
            # Record failure placeholder
            results.append({
                "id": qid,
                "query": query,
                "grade": grade,
                "subject": subject,
                "language": language,
                "error": str(e),
                "answer": "",
                "citations": [],
                "confidence": 0.0,
                "num_hits": 0,
                "used_reranker": False,
                "first_passage_id": "",
            })
            continue

        answer = rag_result.get("answer", "")
        citations = rag_result.get("citations", [])
        confidence = rag_result.get("confidence", 0.0)
        top_passages = rag_result.get("top_passages", [])
        meta = rag_result.get("meta", {})

        first_passage_id = top_passages[0]["id"] if top_passages else ""

        print(f"  -> hits={meta.get('num_hits')}, conf={confidence:.3f}, citations={citations}")

        # Store full record
        results.append({
            "id": qid,
            "query": query,
            "grade": grade,
            "subject": subject,
            "language": language,
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "num_hits": meta.get("num_hits"),
            "bm25_candidates": meta.get("bm25_candidates"),
            "used_reranker": meta.get("used_reranker"),
            "top_passages": top_passages,
            "first_passage_id": first_passage_id,
            "meta": meta,
        })

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "num_queries": len(results),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Write CSV (flat, for manual marking)
    # ------------------------------------------------------------------
    fieldnames = [
        "id",
        "query",
        "grade",
        "subject",
        "language",
        "confidence",
        "num_hits",
        "bm25_candidates",
        "used_reranker",
        "first_passage_id",
        "citations",
        "answer_preview",
        # columns for manual marking:
        "is_correct",
        "is_ncert_aligned",
        "citations_ok",
        "teacher_score",
        "notes",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            citations_str = ";".join(r.get("citations") or [])
            answer_preview = (r.get("answer") or "").replace("\n", " ")
            if len(answer_preview) > 200:
                answer_preview = answer_preview[:200] + "..."

            writer.writerow({
                "id": r["id"],
                "query": r["query"],
                "grade": r["grade"],
                "subject": r["subject"],
                "language": r.get("language", ""),
                "confidence": f"{r.get('confidence', 0.0):.3f}",
                "num_hits": r.get("num_hits", 0),
                "bm25_candidates": r.get("bm25_candidates", ""),
                "used_reranker": r.get("used_reranker", False),
                "first_passage_id": r.get("first_passage_id", ""),
                "citations": citations_str,
                "answer_preview": answer_preview,
                # manual marking placeholders:
                "is_correct": "",
                "is_ncert_aligned": "",
                "citations_ok": "",
                "teacher_score": "",
                "notes": "",
            })

    print("\n[EVAL] Done.")
    print(f"[EVAL] JSON: {json_path}")
    print(f"[EVAL] CSV : {csv_path}")
    print("You can now open the CSV in Excel/Sheets and fill correctness columns.")


if __name__ == "__main__":
    run_evaluation()

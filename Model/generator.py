#!/usr/bin/env python3
"""
generator.py

High-level RAG response generator for EduNiti.

- Uses RAGPipeline (BM25 + optional BGE reranker + LLMEngine)
- Returns a clean JSON response with:
    {
      "answer": "Final explanation...",
      "citations": ["Grade_7___Science___Chapter_10___p0012", ...],
      "top_passages": [...],
      "confidence": 0.92,
      "meta": { ... }
    }
"""

import argparse
import json
from typing import Any, Dict, List, Optional

from rag_pipeline import RAGPipeline
from reranker import BGEReranker


def _compute_confidence(hits: List[Dict[str, Any]]) -> float:
    """
    Heuristic confidence score based on BM25 scores of the final hits.

    - If no hits      → 0.0
    - Otherwise       → normalize average of top-3 scores into (0.3–0.98) range.

    This is a simple placeholder; you can later replace it with a
    proper calibration based on evaluation metrics.
    """
    if not hits:
        return 0.0

    # Collect numeric scores only
    scores: List[float] = []
    for h in hits:
        s = h.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))

    if not scores:
        return 0.5  # unknown scores → mid confidence

    scores_sorted = sorted(scores, reverse=True)
    top_scores = scores_sorted[:3]
    avg_score = sum(top_scores) / len(top_scores)

    # Simple squash: assume typical BM25 scores in ~[5, 25]
    # Map avg_score to 0.3–0.98 range.
    min_s, max_s = 5.0, 25.0
    norm = (avg_score - min_s) / (max_s - min_s)
    norm = max(0.0, min(1.0, norm))  # clamp 0–1
    conf = 0.3 + norm * (0.98 - 0.3)
    return round(conf, 3)


class RAGResponseGenerator:
    """
    High-level wrapper over RAGPipeline.

    Responsibilities:
      - Call RAGPipeline (BM25 + optional reranker + LLM)
      - Extract top passages and citations
      - Compute a confidence score
      - Return a clean JSON-like dict for the frontend / API
    """

    def __init__(
        self,
        use_reranker: bool = True,
        llm_model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> None:
        self.pipeline = RAGPipeline(
            use_reranker=use_reranker,
            llm_model_name=llm_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_response(
        self,
        question: str,
        grade: Optional[str] = None,
        subject: Optional[str] = None,
        chapter: Optional[str] = None,
        language: Optional[str] = None,
        top_k: int = 5,
        bm25_candidates: int = 20,
    ) -> Dict[str, Any]:
        """
        Main public method.

        Returns:
          {
            "answer": <str>,
            "citations": [<passage_id>, ...],
            "top_passages": [...],     # final passages given to the LLM
            "confidence": <float>,
            "meta": {
               "question": ...,
               "grade": ...,
               "subject": ...,
               "chapter": ...,
               "top_k": ...,
               "bm25_candidates": ...,
               "used_reranker": <bool>
            }
          }
        """
        rag_result = self.pipeline.answer_with_rag(
            question=question,
            grade=grade,
            subject=subject,
            chapter=chapter,
            language=language,
            top_k=top_k,
            bm25_candidates=bm25_candidates,
        )

        retrieval = rag_result.get("retrieval", {})
        hits = retrieval.get("hits", []) or []
        llm_answer = rag_result.get("llm_answer", {})
        answer_text = llm_answer.get("answer", "")

        # Citations are simply the IDs of the top passages
        citations = [h.get("id") for h in hits if h.get("id")]

        confidence = _compute_confidence(hits)

        if len(rag_result["retrieval"]["hits"]) == 0:
            return {
                "answer": (
                    "I could not find any relevant NCERT content for your question. "
                    "Please check the grade, subject, or rephrase your question."
                 ),
                "citations": [],
                "top_passages": [],
                "confidence": 0.0,
                "meta": rag_result["llm_answer"]["meta"],
    }

        return {
            "answer": answer_text,
            "citations": citations,
            "top_passages": hits,
            "confidence": confidence,
            "meta": {
                "question": question,
                "grade": grade,
                "subject": subject,
                "chapter": chapter,
                "top_k": retrieval.get("top_k"),
                "num_hits": retrieval.get("num_hits"),
                "bm25_candidates": retrieval.get("bm25_candidates"),
                "used_reranker": retrieval.get("used_reranker"),
            },
        }


# ----------------------------------------------------------------------
# CLI test
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test EduNiti RAG response generator.")
    parser.add_argument("--query", type=str, required=False, default="What is photosynthesis?")
    parser.add_argument("--grade", type=str, required=False, default="Grade_7")
    parser.add_argument("--subject", type=str, required=False, default="Science")
    parser.add_argument("--chapter", type=str, required=False, default=None)
    parser.add_argument("--top_k", type=int, required=False, default=5)
    parser.add_argument("--bm25_candidates", type=int, default=20)
    parser.add_argument("--language", type=str, required=False, default="English")
    parser.add_argument("--no_rerank", action="store_true", help="Disable BGE reranker and use BM25 only")
    args = parser.parse_args()

    gen = RAGResponseGenerator(use_reranker=not args.no_rerank)

    result = gen.generate_response(
        question=args.query,
        grade=args.grade,
        subject=args.subject,
        chapter=args.chapter,
        language=args.language,
        top_k=args.top_k,
        bm25_candidates=args.bm25_candidates,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

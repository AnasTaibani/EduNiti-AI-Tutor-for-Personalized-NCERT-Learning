#!/usr/bin/env python3
"""
rag_pipeline.py

RAG (Retrieval-Augmented Generation) pipeline for EduNiti.

- Calls the BM25 FastAPI endpoint to retrieve top-k NCERT passages
- Optionally reranks them using a BGE Cross-Encoder reranker
- Feeds those passages into LLMEngine (OpenRouter / GPT-like)
- Returns a grounded, syllabus-aligned answer
"""

import os
import argparse
from typing import List, Dict, Any, Optional

import requests

from llm_engine import LLMEngine  # assumes this is in the same Model/ package
from reranker import BGEReranker

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

API_BASE_URL = os.getenv("EDUNITI_API_BASE_URL", "http://127.0.0.1:8000")
# Your FastAPI route that does BM25 retrieval
BM25_ENDPOINT = "/retrieve"


# ----------------------------------------------------------------------
# Retrieval helper: call FastAPI BM25 endpoint
# ----------------------------------------------------------------------

def retrieve_bm25(
    query: str,
    grade: Optional[str] = None,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    top_k: int = 5,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Call the BM25 FastAPI endpoint and return the 'hits' list.

    Request body:
      {
        "query": "...",
        "top_k": 5,
        "grade": "Grade_7",     # optional
        "subject": "Science",   # optional
        "chapter": "Chapter_10" # optional
      }

    Response:
      {
        "query": "...",
        "top_k": 5,
        "hits": [
          {
            "id": "...",
            "score": ...,
            "grade": "...",
            "subject": "...",
            "chapter": "...",
            "text": "...",
            "source": "..."
          },
          ...
        ]
      }
    """
    url = API_BASE_URL.rstrip("/") + BM25_ENDPOINT
    print(f"[RAG] Calling BM25 API: {url}")

    payload: Dict[str, Any] = {
        "query": query,
        "top_k": top_k,
    }
    if grade:
        payload["grade"] = grade
    if subject:
        payload["subject"] = subject
    if chapter:
        payload["chapter"] = chapter

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("hits", [])


# ----------------------------------------------------------------------
# RAG pipeline class
# ----------------------------------------------------------------------

class RAGPipeline:
    def __init__(
        self,
        api_base_url: Optional[str] = None,
        llm_model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 700,
        use_reranker: bool = True,
    ) -> None:
        self.api_base_url = api_base_url or API_BASE_URL
        self.use_reranker = use_reranker

        from llm_engine import LLMEngine
        self.llm_engine = LLMEngine(
            model_name=llm_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # ✅ actually initialise reranker here
        self.reranker = None
        if self.use_reranker:
            try:
                print("[RAG] Initialising BGE reranker...")
                self.reranker = BGEReranker()
                print("[RAG] Reranker initialised.")
            except Exception as e:
                print(f"[RAG] Failed to initialise reranker: {e}")
                self.reranker = None
                self.use_reranker = False

    
    def answer_with_rag(
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
        End-to-end RAG call.

        1) Retrieve BM25 hits (bm25_k, e.g. 20)
        2) Optionally rerank them with BGE reranker → final top_k
        3) Ask LLM to answer using those passages

        Returns:
          {
            "question": ...,
            "retrieval": {
               "top_k": ...,
               "num_hits": ...,
               "hits": [...]      # final hits (reranked or BM25)
            },
            "llm_answer": {
               "answer": "...",
               "raw_model_output": "...",
               "meta": { ... }
            }
          }
        """

        # 1) BM25 retrieval
        hits_bm25 = retrieve_bm25(
            query=question,
            grade=grade,
            subject=subject,
            chapter=chapter,
            top_k=bm25_candidates,
        )

        used_reranker = False

        if self.use_reranker and self.reranker and hits_bm25:
            print(f"[RAG] Reranking {len(hits_bm25)} BM25 hits with BGE...")
            hits_final = self.reranker.rerank(
                query=question,
                hits=hits_bm25,
                top_k=top_k,
            )
            used_reranker = True
        else:
            hits_final = hits_bm25[:top_k]

        # 3) LLM answer
        llm_resp = self.llm_engine.answer_question(
            question=question,
            passages=hits_final,
            grade=grade,
            subject=subject,
            language=language,
        )

        return {
            "question": question,
            "retrieval": {
                "top_k": top_k,
                "num_hits": len(hits_final),
                "hits": hits_final,
                "bm25_candidates": len(hits_bm25),
                "used_reranker": self.use_reranker and self.reranker is not None,
            },
            "llm_answer": llm_resp,
        }


# ----------------------------------------------------------------------
# CLI test
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test EduNiti RAG pipeline (BM25 + optional BGE reranker + LLM).")
    parser.add_argument("--query", type=str, required=False, default="What is photosynthesis?")
    parser.add_argument("--grade", type=str, required=False, default="Grade_7")
    parser.add_argument("--subject", type=str, required=False, default="Science")
    parser.add_argument("--chapter", type=str, required=False, default=None)
    parser.add_argument("--top_k", type=int, required=False, default=5)
    parser.add_argument("--bm25_k", type=int, required=False, default=20, help="Number of BM25 candidates before reranking")
    parser.add_argument("--language", type=str, required=False, default="English")
    parser.add_argument("--no_rerank", action="store_true", help="Disable BGE reranker and use BM25 only")
    args = parser.parse_args()

    print(f"[RAG] Using API base URL: {API_BASE_URL}")
    print(f"[RAG] Question: {args.query}")
    print(f"[RAG] Grade: {args.grade}, Subject: {args.subject}, Chapter: {args.chapter}")
    print(f"[RAG] top_k (final): {args.top_k}, bm25_k (candidates): {args.bm25_k}")
    print(f"[RAG] Reranker enabled: {not args.no_rerank}")
    print("-" * 60)

    pipeline = RAGPipeline(use_reranker=not args.no_rerank)

    result = pipeline.answer_with_rag(
        question=args.query,
        grade=args.grade,
        subject=args.subject,
        chapter=args.chapter,
        language=args.language,
        top_k=args.top_k,
        bm25_k=args.bm25_k,
    )

    print("\n=== RETRIEVED PASSAGES (FINAL) ===\n")
    for i, hit in enumerate(result["retrieval"]["hits"], start=1):
        score = hit.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
        print(f"[{i}] {hit.get('id')}  (score={score_str})")
        print(f"    Grade={hit.get('grade')}, Subject={hit.get('subject')}, Chapter={hit.get('chapter')}")
        text = (hit.get("text") or "").strip()
        if len(text) > 400:
            text = text[:400] + "..."
        print("    " + text)
        print()

    print("\n=== MODEL ANSWER ===\n")
    print(result["llm_answer"]["answer"])


if __name__ == "__main__":
    main()

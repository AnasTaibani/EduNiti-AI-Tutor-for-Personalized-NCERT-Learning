#!/usr/bin/env python3
"""
llm_engine.py

LLM engine for EduNiti RAG pipeline using OpenAI gpt-4o-mini via LangChain.

- Uses BM25-retrieved NCERT passages as context
- Generates grounded answers with light citations
- Designed to be called from your FastAPI layer or any Python code

Requirements:
  pip install langchain langchain-openai openai

Environment:
  Set your OpenAI API key (one of):
    - OPENAI_API_KEY env var
"""

import os
from typing import List, Dict, Optional, Any

from langchain_openai import ChatOpenAI


# ----------------------------------------------------------------------
# LLM Engine
# ----------------------------------------------------------------------
class LLMEngine:
    """
    Wrapper around gpt-4o-mini for NCERT-grounded answering.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 700,
        max_chars_per_passage: int = 900,
    ) -> None:
        """
        Initialize the ChatOpenAI client.

        The API key is read from the OPENAI_API_KEY env var.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Create an API key at https://platform.openai.com/api-keys "
                "and set it in your environment."
            )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.max_chars_per_passage = max_chars_per_passage

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _truncate_text(self, text: str) -> str:
        """Hard truncate a passage to keep prompt size under control."""
        if not text:
            return ""
        if len(text) <= self.max_chars_per_passage:
            return text
        return text[: self.max_chars_per_passage].rstrip() + " …"

    def _build_context_block(self, passages: List[Dict[str, Any]]) -> str:
        """
        Turn retrieval results into a formatted context string.

        Each passage dict is expected to have at least:
          - id
          - text
        Optionally:
          - grade
          - subject
          - chapter
        """
        if not passages:
            return "No NCERT passages were retrieved."

        lines: List[str] = []
        for i, p in enumerate(passages, start=1):
            pid = p.get("id", f"passage_{i}")
            grade = p.get("grade", "Unknown_Grade")
            subject = p.get("subject", "Unknown_Subject")
            chapter = p.get("chapter", "Unknown_Chapter")
            text = self._truncate_text(p.get("text", "").strip())

            lines.append(
                f"[Source {i}] ID: {pid}\n"
                f"  Grade: {grade}, Subject: {subject}, Chapter: {chapter}\n"
                f"  Text:\n{text}\n"
            )

        return "\n".join(lines)

    def _build_prompt(
        self,
        question: str,
        passages: List[Dict[str, Any]],
        grade: Optional[str] = None,
        subject: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Build a single string prompt for gpt-4o-mini.

        Adds:
        - Follow-up questions section at the end.
        """

        context_block = self._build_context_block(passages)

        grade_info = grade or "Not specified"
        subject_info = subject or "Not specified"
        lang_instruction = ""

        if language:
            lang_instruction = (
                f"Write the final answer primarily in {language}. "
                "You may include short key terms in English if needed.\n"
            )

        prompt = f"""
You are an **NCERT curriculum tutor** for school students in India.
Your job is to answer questions **only using the NCERT passages provided below**.

Student profile:
- Target Grade: {grade_info}
- Subject: {subject_info}

Rules:
1. Use ONLY the information from the "NCERT Sources" section below.
2. If the answer is not clearly supported by the sources, say:
   "Based on the given NCERT content, I cannot fully answer this."
3. Keep the explanation age-appropriate for the target grade.
4. Be clear, step-by-step, and avoid unnecessary jargon.
5. At the end of your answer, write:
   **Sources: [Source 1], [Source 3]**
6. After that, ALWAYS add:
   **Follow-up Questions:** Provide 2–3 simple, syllabus-aligned questions that help the student revise or think deeper, based ONLY on provided sources.

STRICT NCERT GROUNDING RULES:
1. You MUST answer only using the passages provided in `NCERT Sources`.
2. If the question cannot be answered from the provided sources, respond:
   "Based on the given NCERT content, I cannot answer this fully."
3. Never invent facts, examples, formulas, definitions, or explanations.
4. Never mix information from outside NCERT.
5. If sources contain partial information, give only that partial information.
6. Keep answers aligned to the student’s grade level.
7. End with simple citations like: Sources: [Source 1].
8. If the query is unsafe, inappropriate, or not academic → politely decline.
{lang_instruction}
----------------------
NCERT Sources:
----------------------
{context_block}
----------------------
Student Question:
{question}

Now, provide:
1. A clear explanation for the student.
2. Any definitions, formulas, or key ideas relevant to the question.
3. A "Sources" line showing which [Source X] numbers you used.
4. A "Follow-up Questions" section (2–3 questions) based strictly on the NCERT passages.
"""

        return prompt.strip()


    def _build_sources_meta(self, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build a structured list of citation metadata corresponding to [Source i].
        This is useful for the frontend to show proper NCERT references.
        """
        sources: List[Dict[str, Any]] = []
        for i, p in enumerate(passages, start=1):
            sources.append(
                {
                    "source_index": i,  # matches [Source i] in the prompt
                    "id": p.get("id"),
                    "grade": p.get("grade"),
                    "subject": p.get("subject"),
                    "chapter": p.get("chapter"),
                    "source_path": p.get("source") or p.get("path"),
                }
            )
        return sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer_question(
        self,
        question: str,
        passages: List[Dict[str, Any]],
        grade: Optional[str] = None,
        subject: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main method called from the RAG pipeline.

        Args:
            question: student question text.
            passages: list of retrieved passage dicts from BM25.
                      Expected keys: id, text, grade, subject, chapter, source/path
            grade: e.g., "Grade_7".
            subject: e.g., "Science".
            language: optional (e.g., "English", "Hindi").

        Returns:
            {
              "answer": <str>,
              "raw_model_output": <str>,
              "meta": {
                  "grade": ...,
                  "subject": ...,
                  "num_passages": ...,
                  "sources": [ {source_index, id, grade, subject, chapter, source_path}, ... ]
              }
            }
        """
        prompt = self._build_prompt(
            question=question,
            passages=passages,
            grade=grade,
            subject=subject,
            language=language,
        )

        resp = self.llm.invoke(prompt)
        answer_text = resp.content if hasattr(resp, "content") else str(resp)

        sources_meta = self._build_sources_meta(passages)

        return {
            "answer": answer_text,
            "raw_model_output": answer_text,
            "meta": {
                "grade": grade,
                "subject": subject,
                "num_passages": len(passages),
                "sources": sources_meta,
            },
        }


# ----------------------------------------------------------------------
# Simple CLI test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Quick local test (run: python Model/llm_engine.py)

    This uses a dummy passage just to verify that the LLM call works.
    In real usage, you'll pass in BM25-retrieved passages from your API.
    """
    engine = LLMEngine()

    sample_question = "What is photosynthesis?"
    sample_passages = [
        {
            "id": "Grade_7___Science___Chapter_10___p0019",
            "grade": "Grade_7",
            "subject": "Science",
            "chapter": "Chapter_10",
            "source": "NCERT_paragraphs/Grade_7/Science/Chapter_10.txt",
            "text": (
                "This process of synthesis of food is known as photosynthesis. "
                "Leaves are the food factories of a plant. Tiny pores on the surface of leaves, "
                "called stomata, help in the exchange of oxygen and carbon dioxide during "
                "photosynthesis and respiration."
            ),
        }
    ]

    result = engine.answer_question(
        question=sample_question,
        passages=sample_passages,
        grade="Grade_7",
        subject="Science",
        language="English",
    )

    print("\n=== MODEL ANSWER ===\n")
    print(result["answer"])
    print("\n=== META ===")
    print(result["meta"])

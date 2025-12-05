#!/usr/bin/env python3
"""
Retrieval API (BM25-only, with grade/subject filters)

Endpoints:
  POST /retrieve
    body: {
      "query": "...",
      "top_k": 5,
      "grade": "Grade_10",        # optional
      "subject": "Science",       # optional
      "chapter": "Chapter_1"      # optional
    }

Uses:
  - BM25 pickle: Preprocessing/NCERT_passages_hybrid/bm25.pkl
      expected keys: {"bm25", "tokenized_corpus", "ids"}
  - Passages JSONL: Preprocessing/NCERT_passages_hybrid/passages.jsonl
      each line: {
        "id": "...",
        "grade": "...",
        "subject": "...",
        "chapter": "...",
        "text": "..."
        ...
      }
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from quiz import router as quiz_router
# near other includes
from student import router as student_router





import json
import re

from fastapi import FastAPI
from pydantic import BaseModel

from rank_bm25 import BM25Okapi

# ---------- Paths (adjust if your layout differs) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BM25_PKL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "bm25.pkl"
PASSAGES_JSONL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "passages.jsonl"

# ---------- FastAPI app ----------
app = FastAPI(title="EduNiti NCERT Retrieval API (BM25)")
app.include_router(quiz_router, prefix="/quiz", tags=["quiz"])
app.include_router(student_router, prefix="/student", tags=["student"])

bm25: BM25Okapi | None = None
corpus_tokens: List[List[str]] = []
ids: List[str] = []
id2meta: Dict[str, Dict[str, Any]] = {}


# ---------- Simple tokenizer (no NLTK needed) ----------
def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    # split on word characters (a–z, digits, underscore) – works OK for English queries
    return re.findall(r"\w+", text)


# ---------- Load BM25 ----------
def load_bm25(pkl_path: Path):
    import pickle

    print(f"[INIT] Loading BM25 from {pkl_path}")
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)

    # Your bm25.pkl is a dict with: {"bm25", "tokenized_corpus", "ids"}
    if isinstance(obj, dict) and {"bm25", "tokenized_corpus", "ids"} <= set(obj.keys()):
        bm25_obj = obj["bm25"]
        corpus = obj["tokenized_corpus"]
        ids_list = obj["ids"]
        print(f"[INIT] BM25 dict loaded: {len(ids_list)} documents")
        return bm25_obj, corpus, ids_list

    raise ValueError("Unsupported bm25.pkl format. Expected dict with keys: bm25, tokenized_corpus, ids")


# ---------- Load passages metadata ----------
def load_passages_meta(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    print(f"[INIT] Loading passages from {jsonl_path}")
    mapping: Dict[str, Dict[str, Any]] = {}
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("id")
            if not pid:
                continue
            mapping[pid] = obj
            count += 1
    print(f"[INIT] Loaded {count} passages into id→meta mapping")
    return mapping


# ---------- Request / Response models ----------
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    grade: Optional[str] = None     # e.g. "Grade_10"
    subject: Optional[str] = None   # e.g. "Science"
    chapter: Optional[str] = None   # e.g. "Chapter_1"


class PassageHit(BaseModel):
    id: str
    score: float
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    text: Optional[str] = None
    source: Optional[str] = None


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    hits: List[PassageHit]


# ---------- Core search (BM25 with filters) ----------
def search_bm25(
    query: str,
    top_k: int,
    grade: Optional[str] = None,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
) -> List[PassageHit]:
    assert bm25 is not None

    q_tokens = simple_tokenize(query)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)  # list/np.array length = len(corpus_tokens)

    scored: List[tuple[str, float]] = []
    for i, score in enumerate(scores):
        pid = ids[i]
        meta = id2meta.get(pid)
        if not meta:
            continue

        # Apply filters (exact string match)
        if grade and meta.get("grade") != grade:
            continue
        if subject and meta.get("subject") != subject:
            continue
        if chapter and meta.get("chapter") != chapter:
            continue

        scored.append((pid, float(score)))

    # Sort by BM25 score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    hits: List[PassageHit] = []
    for pid, score in scored[:top_k]:
        meta = id2meta.get(pid, {})
        hits.append(
            PassageHit(
                id=pid,
                score=score,
                grade=meta.get("grade"),
                subject=meta.get("subject"),
                chapter=meta.get("chapter"),
                text=meta.get("text"),
                source=meta.get("source"),
            )
        )
    return hits


# ---------- Startup ----------
@app.on_event("startup")
def startup_event():
    global bm25, corpus_tokens, ids, id2meta

    # Load BM25 index
    bm25_obj, corpus, ids_list = load_bm25(BM25_PKL)
    bm25 = bm25_obj
    corpus_tokens = corpus
    ids = ids_list
    print(f"[INIT] BM25 corpus size: {len(ids)} documents")

    # Load metadata
    global id2meta
    id2meta = load_passages_meta(PASSAGES_JSONL)
    print("[INIT] API ready.")


# ---------- Endpoints ----------
@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    hits = search_bm25(
        query=req.query,
        top_k=req.top_k,
        grade=req.grade,
        subject=req.subject,
        chapter=req.chapter,
    )
    return RetrieveResponse(query=req.query, top_k=req.top_k, hits=hits)


@app.get("/health")
def health():
    return {"status": "ok", "bm25_docs": len(ids)}

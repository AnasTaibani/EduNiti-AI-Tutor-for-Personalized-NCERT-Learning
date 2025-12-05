from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel
from datetime import datetime

from infra.mongo_client_ import _get_client as get_db_client, DEFAULT_DB, list_mastery_by_student as list_mastery

router = APIRouter()

class MasteryItem(BaseModel):
    concept_id: str
    p_mastery: float
    last_updated: Optional[datetime] = None
    meta: Optional[Dict[str, Any]] = None

class MasteryResponse(BaseModel):
    student_id: str
    mastery: List[MasteryItem]

class RecommendationItem(BaseModel):
    concept_id: str
    p_mastery: float
    # normalized to list of dicts for frontend
    example_questions: List[Dict[str, Any]] = []

class RecommendationResponse(BaseModel):
    student_id: str
    recommendations: List[RecommendationItem]


def _normalize_questions(raw):
    """
    Normalize example_questions field to a list of dicts with at least {"text": ...}.
    Accepts:
      - None -> []
      - list of strings -> [{"text": s}, ...]
      - list of dicts -> keep dicts
      - single string/dict -> convert to list
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"text": raw}]
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, (list, tuple)):
        out = []
        for item in raw:
            if isinstance(item, str):
                out.append({"text": item})
            elif isinstance(item, dict):
                out.append(item)
            else:
                out.append({"text": str(item)})
        return out
    return [{"text": str(raw)}]


@router.get("/{student_id}/mastery", response_model=MasteryResponse)
def get_mastery_for_student(
    student_id: str,
    grade: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    client = get_db_client()
    db = client[DEFAULT_DB]
    try:
        docs = list_mastery(student_id, db=db)
    finally:
        client.close()

    if grade or subject:
        def keep(d):
            meta = d.get("meta", {}) or {}
            if grade and meta.get("grade") != grade:
                return False
            if subject and meta.get("subject") != subject:
                return False
            return True
        docs = [d for d in docs if keep(d)]

    docs_sorted = sorted(docs, key=lambda x: x.get("p_mastery", 0.0))
    docs_limited = docs_sorted[:limit]
    return MasteryResponse(
        student_id=student_id,
        mastery=[MasteryItem(**{
            "concept_id": d["concept_id"],
            "p_mastery": float(d.get("p_mastery", 0.0)),
            "last_updated": d.get("last_updated"),
            "meta": d.get("meta"),
        }) for d in docs_limited]
    )


@router.get("/{student_id}/recommendation", response_model=RecommendationResponse)
def recommend_for_student(
    student_id: str,
    n: int = Query(3, ge=1, le=50),
    grade: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
):
    client = get_db_client()
    db = client[DEFAULT_DB]
    try:
        mastery_docs = list_mastery(student_id, db=db)
        mastery_map = {d["concept_id"]: float(d.get("p_mastery", 0.05)) for d in mastery_docs}

        concept_query = {}
        if grade:
            concept_query["grade"] = grade
        if subject:
            concept_query["subject"] = subject

        concept_cursor = db["concepts"].find(concept_query, {"_id": 0, "concept_id": 1, "example_questions": 1})
        concepts = list(concept_cursor)

        DEFAULT_PRIOR = 0.05
        recs = []
        if not concepts:
            items = sorted(mastery_map.items(), key=lambda kv: kv[1])[:n]
            for k, v in items:
                recs.append({"concept_id": k, "p_mastery": v, "example_questions": []})
        else:
            scored = []
            for c in concepts:
                cid = c.get("concept_id")
                if not cid:
                    continue
                p = mastery_map.get(cid, DEFAULT_PRIOR)
                raw_eqs = c.get("example_questions")
                eqs = _normalize_questions(raw_eqs)
                scored.append((cid, p, eqs))
            scored_sorted = sorted(scored, key=lambda t: t[1])[:n]
            recs = [{"concept_id": cid, "p_mastery": p, "example_questions": eqs} for cid, p, eqs in scored_sorted]

    finally:
        client.close()

    return RecommendationResponse(student_id=student_id, recommendations=[RecommendationItem(**r) for r in recs])
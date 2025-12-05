# Retrieval/api/quiz.py
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

import logging

# Import your mongo helpers and the BKT update function
from infra.mongo_client_ import (
    _get_client as get_db_client,  # returns a MongoClient
    log_quiz_attempt,
    get_mastery,   # get_mastery(student_id, concept_id, db=None)
    set_mastery,   # set_mastery(..., db=None)
    DEFAULT_DB,    # default database name
)
from Model.bkt import bkt_update  # bkt_update(p_mastery, correct: bool, params: dict) -> float

logger = logging.getLogger(__name__)

router = APIRouter()


class QuizSubmitRequest(BaseModel):
    student_id: str
    question_id: str
    concept_ids: List[str] = Field(..., min_length=1)
    correct: bool
    response_time: Optional[float] = None  # seconds
    difficulty: Optional[float] = None     # 0.0-1.0 or custom scale
    metadata: Optional[Dict[str, Any]] = None


class MasteryResult(BaseModel):
    concept_id: str
    old_p: float
    new_p: float
    params: Dict[str, Any]


class QuizSubmitResponse(BaseModel):
    student_id: str
    question_id: str
    timestamp: str
    updated_mastery: List[MasteryResult]


@router.post("/submit", response_model=QuizSubmitResponse)
def submit_quiz(payload: QuizSubmitRequest):
    """
    Record a student's quiz attempt and update mastery for each related concept.

    Note: include this router with prefix="/quiz" so endpoint becomes POST /quiz/submit
    """
    # Basic validation
    if not payload.student_id or not payload.question_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="student_id and question_id required")

    timestamp = datetime.utcnow()

    # Default BKT parameters (can be overridden per-concept in DB later)
    default_params = {
        "p_trans": 0.15,  # probability of learning between attempts
        "p_guess": 0.2,   # guessing probability
        "p_slip": 0.1,    # slip probability
    }

    updated_mastery = []

    client = None
    try:
        # create a MongoClient and derive the Database object we will pass to helpers
        client = get_db_client()
        db = client[DEFAULT_DB]

        # 1) Log the quiz attempt skeleton
        attempt_doc = {
            "student_id": payload.student_id,
            "question_id": payload.question_id,
            "concept_ids": payload.concept_ids,
            "correct": payload.correct,
            "response_time": payload.response_time,
            "difficulty": payload.difficulty,
            "metadata": payload.metadata or {},
            "timestamp": timestamp,
        }


        # For each concept, fetch current mastery and update
        for concept_id in payload.concept_ids:
            try:
                m = get_mastery(payload.student_id, concept_id, db=db)
            except Exception as e:
                logger.exception("Error fetching mastery for %s/%s", payload.student_id, concept_id)
                raise HTTPException(status_code=500, detail="Error fetching mastery from DB")

            if m:
                old_p = float(m.get("p_mastery", 0.05))
                params = m.get("params", default_params)
            else:
                old_p = 0.05  # default prior
                params = default_params

            # Run BKT update (wrap in try to catch any algorithm errors)
            try:
                new_p = bkt_update(old_p, payload.correct, params)
            except Exception:
                logger.exception("BKT update failed for %s on %s", payload.student_id, concept_id)
                raise HTTPException(status_code=500, detail="BKT update failed")

            # Persist new mastery back to DB (upsert) — pass db=db
            try:
                set_mastery(
                    student_id=payload.student_id,
                    concept_id=concept_id,
                    p_mastery=new_p,
                    meta=None,
                    source="quiz_submit",
                    db=db,
                )
            except Exception:
                logger.exception("Failed to persist mastery for %s/%s", payload.student_id, concept_id)
                raise HTTPException(status_code=500, detail="Failed to persist mastery to DB")

            updated_mastery.append({
                "concept_id": concept_id,
                "old_p": old_p,
                "new_p": new_p,
                "params": params,
            })

        # After all updates, add 'updated_mastery' into the quiz log
        attempt_doc["updated_mastery"] = updated_mastery

        # Insert the quiz log (log_quiz_attempt accepts a client)
        try:
            log_id = log_quiz_attempt(attempt_doc, client=client)
            logger.debug("Logged quiz attempt id=%s", log_id)
        except Exception:
            logger.exception("Failed to log quiz attempt")
            raise HTTPException(status_code=500, detail="Failed to log quiz attempt to DB")

    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("Error closing DB client", exc_info=True)

    # Build response
    resp = QuizSubmitResponse(
        student_id=payload.student_id,
        question_id=payload.question_id,
        timestamp=timestamp.isoformat() + "Z",
        updated_mastery=[MasteryResult(**m) for m in updated_mastery],
    )
    return resp

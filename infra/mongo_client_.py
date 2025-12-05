#!/usr/bin/env python3
# infra/mongo_client.py
"""
Simple MongoDB client helpers for mastery documents used by BKT.

Environment:
  - MONGO_URI (required): your MongoDB connection string
  - MONGO_DB  (optional): db name (default: EduNiti)

Mastery document format (example):
{
  "student_id": "student_123",
  "concept_id": "Grade_7__Science__Photosynthesis",
  "p_mastery": 0.72,              # current mastery probability (0..1)
  "last_updated": datetime.utcnow(),  # stored as BSON date
  "history": [
      {"at": datetime(...), "p": 0.5, "source": "quiz_v1", "notes": "..."},
      ...
  ],
  "meta": {...}
}
"""
from __future__ import annotations
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from huggingface_hub import get_collection
from pymongo import MongoClient, errors
from pymongo.collection import Collection

# Defaults
DEFAULT_DB = os.getenv("MONGO_DB", "EduNiti")
_MONGO_URI = os.getenv("MONGO_URI", None)
if not _MONGO_URI:
    # Do not raise here — raise when trying to connect to allow import-time flexibility in tests
    _MONGO_URI = None


def _get_client() -> MongoClient:
    if not _MONGO_URI:
        raise RuntimeError("MONGO_URI environment variable is not set. Set it to your Atlas/local URI.")
    return MongoClient(_MONGO_URI, serverSelectionTimeoutMS=10000)


def get_db(db_name: Optional[str] = None):
    """
    Return a MongoDB database object.
    """
    client = _get_client()
    name = db_name or DEFAULT_DB
    # quick ping to surface connection errors early
    try:
        client.admin.command("ping")
    except errors.PyMongoError as e:
        raise RuntimeError(f"Unable to contact MongoDB server: {e}")
    return client[name]


def _mastery_collection(db=None) -> Collection:
    """
    Internal helper returning the mastery collection for the given db.
    """
    database = db if db is not None else get_db()
    return database["mastery"]


def create_mastery_index(db=None, unique_by_student_concept: bool = True):
    """
    Ensure indexes exist for mastery collection.
    - unique_by_student_concept: create a unique index on (student_id, concept_id)
    """
    col = _mastery_collection(db)
    # index for fast retrieval by student
    col.create_index([("student_id", 1)], name="idx_student_id")
    # index for concept
    col.create_index([("concept_id", 1)], name="idx_concept_id")
    # unique composite index to avoid duplicates (if requested)
    if unique_by_student_concept:
        try:
            col.create_index([("student_id", 1), ("concept_id", 1)], unique=True, name="uniq_student_concept")
        except errors.OperationFailure:
            # index may already exist with different options; ignore and keep going
            pass
    return True


def get_mastery(student_id: str, concept_id: str, db=None) -> Optional[Dict[str, Any]]:
    """
    Fetch current mastery document for a student-concept pair.
    Returns None if not found.
    """
    if not student_id or not concept_id:
        raise ValueError("student_id and concept_id are required")
    col = _mastery_collection(db)
    doc = col.find_one({"student_id": student_id, "concept_id": concept_id}, {"_id": 0})
    return doc

def get_db_client():
    """Return a new MongoClient instance (caller should close it)."""
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)


def _get_collections(client):
    db = client[DEFAULT_DB]
    return {
        "quiz_logs": db["quiz_logs"],
        "mastery": db["mastery"],
    }


def log_quiz_attempt(attempt_doc: Dict[str, Any], client: Optional[MongoClient] = None):
    """
    Insert a quiz attempt. If client not provided, create one and close.
    Returns the inserted_id.
    """
    close_after = False
    if client is None:
        client = get_db_client()
        close_after = True
    try:
        cols = _get_collections(client)
        res = cols["quiz_logs"].insert_one(attempt_doc)
        return res.inserted_id
    finally:
        if close_after:
            client.close()




def set_mastery(student_id: str, concept_id: str, p_mastery: float,
                meta: Optional[Dict[str, Any]] = None,
                source: Optional[str] = None,
                db=None) -> Dict[str, Any]:
    """
    Upsert mastery doc:
     - updates p_mastery and last_updated
     - appends a history entry with timestamp, p, and optional source/meta
    Returns the upserted/updated doc (without _id).
    """
    if not student_id or not concept_id:
        raise ValueError("student_id and concept_id are required")
    if not isinstance(p_mastery, (int, float)) or not (0.0 <= float(p_mastery) <= 1.0):
        raise ValueError("p_mastery must be a number between 0.0 and 1.0")

    col = _mastery_collection(db)
    now = datetime.utcnow()

    # history entry conforms to DB validator: ts and p_mastery required
    history_entry = {
        "ts": now,
        "p_mastery": float(p_mastery),
    }
    if source:
        history_entry["source"] = source
    if meta:
        history_entry["meta"] = meta

    update = {
        "$set": {
            "student_id": student_id,
            "concept_id": concept_id,
            "p_mastery": float(p_mastery),
            "last_updated": now,
        },
        "$push": {
            "history": {"$each": [history_entry], "$slice": -100}  # keep last 100 history entries
        },
        "$setOnInsert": {
            "meta": meta or {},
        }
    }

    res = col.update_one({"student_id": student_id, "concept_id": concept_id}, update, upsert=True)

    # return the current document
    doc = col.find_one({"student_id": student_id, "concept_id": concept_id}, {"_id": 0})
    return doc


def list_mastery_by_student(student_id: str, db=None) -> List[Dict[str, Any]]:
    """
    Return list of mastery docs for the given student (without _id).
    """
    if not student_id:
        raise ValueError("student_id is required")
    col = _mastery_collection(db)
    docs = list(col.find({"student_id": student_id}, {"_id": 0}).sort("last_updated", -1))
    return docs

#!/usr/bin/env python3
# infra/load_concepts.py
"""
Load concepts_seed.jsonl into MongoDB collection `concepts`.

Requirements:
  pip install pymongo python-dotenv

Usage:
  set MONGO_URI env var to your Atlas/local URI
  python infra/load_concepts.py --file data/concepts_seed.jsonl
"""
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from pymongo import MongoClient, errors
from pymongo.errors import OperationFailure

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "data" / "concepts_seed.jsonl"


def get_mongo_client():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI env var not set. Set it to your Atlas or local uri.")
    return MongoClient(uri, serverSelectionTimeoutMS=10000)


def ensure_concept_id_index(col, index_name="concept_id_unique"):
    try:
        col.create_index([("concept_id", 1)], unique=True, name=index_name)
        print(f"[index] Created index '{index_name}' on concept_id.")
        return
    except OperationFailure as e:
        print(f"[index] OperationFailure while creating index: {e}")

    info = col.index_information()
    existing = []
    for nm, meta in info.items():
        keys = meta.get("key", [])
        if any(k[0] == "concept_id" for k in keys):
            existing.append((nm, meta))

    if not existing:
        raise RuntimeError("Failed to create index on concept_id and no existing index found.")

    for nm, meta in existing:
        if meta.get("unique", False):
            print(f"[index] Found existing unique index '{nm}', using it.")
            return

    idx_to_drop = existing[0][0]
    print(f"[index] Dropping non-unique index '{idx_to_drop}' and creating unique index '{index_name}'.")
    try:
        col.drop_index(idx_to_drop)
    except Exception as e:
        raise RuntimeError(f"Failed to drop existing index '{idx_to_drop}': {e}")

    col.create_index([("concept_id", 1)], unique=True, name=index_name)
    print(f"[index] Created unique index '{index_name}' on concept_id.")


def _ensure_required_fields(obj):
    """
    Mutates obj in-place to ensure required fields exist for collection validator.
    - created_at: datetime (BSON date)
    - created_by: identifier string
    Returns a list of fields that were added (for logging).
    """
    added = []
    if "created_at" not in obj:
        # Use datetime object (BSON date) so Mongo validator with bsonType:"date" passes
        obj["created_at"] = datetime.utcnow()
        added.append("created_at")
    else:
        # If present but it's a string, attempt to parse ISO -> datetime
        if isinstance(obj["created_at"], str):
            try:
                obj["created_at"] = datetime.fromisoformat(obj["created_at"].replace("Z", "+00:00"))
                added.append("created_at(parsed)")
            except Exception:
                # if parsing fails, overwrite with current UTC datetime
                obj["created_at"] = datetime.utcnow()
                added.append("created_at(fixed)")
    if "created_by" not in obj:
        obj["created_by"] = "concepts_seed_loader"
        added.append("created_by")
    return added


def load_and_upsert(concepts_file: Path, db_name="EduNiti", collection_name="concepts"):
    print("Connecting to MongoDB...")
    client = get_mongo_client()
    try:
        client.admin.command("ping")
    except errors.ServerSelectionTimeoutError as e:
        raise RuntimeError(f"Cannot connect to MongoDB: {e}")

    db = client[db_name]
    col = db[collection_name]

    ensure_concept_id_index(col)

    total = 0
    inserted = 0
    updated = 0
    skipped = 0
    errors_count = 0

    with concepts_file.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[skip:{ln}] JSON decode error: {e} -- line start: {line[:80]}...")
                skipped += 1
                continue

            total += 1
            concept_id = obj.get("concept_id")
            if not concept_id:
                print(f"[skip:{ln}] Missing concept_id, skipping.")
                skipped += 1
                continue

            added = _ensure_required_fields(obj)
            if added:
                print(f"[fix:{ln}] Added/Fixed fields for {concept_id}: {added}")

            try:
                res = col.update_one(
                    {"concept_id": concept_id},
                    {"$set": obj},
                    upsert=True
                )
            except Exception as e:
                print(f"[error] Failed to upsert concept_id={concept_id}: {e}")
                errors_count += 1
                continue

            if getattr(res, "upserted_id", None):
                inserted += 1
            elif getattr(res, "modified_count", 0) > 0:
                updated += 1

    print(f"Processed {total} concepts; inserted {inserted}, updated {updated}, skipped {skipped}, errors {errors_count}.")

    # Print sample documents (serialize created_at to ISO string for JSON)
    print("Sample document (first 3):")
    for doc in col.find().limit(3):
        sample = {
            "concept_id": doc.get("concept_id"),
            "label": doc.get("label"),
            "grade": doc.get("grade"),
            "subject": doc.get("subject"),
        }
        ca = doc.get("created_at")
        if isinstance(ca, datetime):
            sample["created_at"] = ca.isoformat() + "Z"
        else:
            sample["created_at"] = str(ca)
        print(json.dumps(sample, ensure_ascii=False, indent=2))

    client.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", "-f", type=Path, default=DEFAULT_FILE)
    p.add_argument("--db", type=str, default=os.getenv("MONGO_DB", "EduNiti"))
    p.add_argument("--col", type=str, default="concepts")
    args = p.parse_args()

    if not args.file.exists():
        raise SystemExit(f"Concepts file not found: {args.file}")

    load_and_upsert(args.file, db_name=args.db, collection_name=args.col)


if __name__ == "__main__":
    main()

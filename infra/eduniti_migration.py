#!/usr/bin/env python3
"""
eduniti_migration.py

Creates the `eduniti` database, collections, JSON Schema validators, and indexes using PyMongo.

Usage:
    1) Install dependencies:
       pip install pymongo dnspython

    2) Set your MongoDB connection string in the environment variable MONGO_URI, for example:
       export MONGO_URI="mongodb+srv://user:pass@cluster0.example.mongodb.net/?retryWrites=true&w=majority"

    3) Run:
       python eduniti_migration.py --apply

By default the script will run in "dry-run" mode and print planned operations.
Use --apply to actually create/update collections and indexes.

Notes:
    - The script will use collMod to update validators if a collection already exists.
    - validationLevel is set to "moderate" to avoid breaking existing documents.
    - Use --sample to insert small sample documents for verification.
"""

import os
import json
import argparse
from datetime import datetime

def get_mongo_client(uri):
    from pymongo import MongoClient
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


DB_NAME = "EduNiti"

STUDENTS_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["student_id", "name", "created_at"],
        "properties": {
            "student_id": {"bsonType": "string", "description": "unique string id for student"},
            "name": {"bsonType": "string"},
            "grade": {"bsonType": "string"},
            "created_at": {"bsonType": "date"},
            "meta": {
                "bsonType": "object",
                "description": "free-form metadata (timezone, etc.)",
                "properties": {
                    "timezone": {"bsonType": "string"}
                },
                "additionalProperties": True
            }
        },
        "additionalProperties": True
    }
}

CONCEPTS_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["concept_id", "label", "subject", "created_at"],
        "properties": {
            "concept_id": {"bsonType": "string"},
            "label": {"bsonType": "string"},
            "subject": {"bsonType": "string"},
            "grade": {"bsonType": "string"},
            "parents": {"bsonType": "array", "items": {"bsonType": "string"}},
            "examples": {"bsonType": "array", "items": {"bsonType": "string"}},
            "created_at": {"bsonType": "date"}
        },
        "additionalProperties": True
    }
}

MASTERY_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["student_id", "concept_id", "p_mastery", "last_updated"],
        "properties": {
            "student_id": {"bsonType": "string"},
            "concept_id": {"bsonType": "string"},
            "p_mastery": {
                "bsonType": ["double", "int", "decimal"],
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "probability of mastery (0..1)"
            },
            "last_updated": {"bsonType": "date"},
            "history": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "required": ["ts", "p_mastery"],
                    "properties": {
                        "ts": {"bsonType": "date"},
                        "p_mastery": {"bsonType": ["double", "int", "decimal"], "minimum": 0.0, "maximum": 1.0},
                        "note": {"bsonType": "string"}
                    },
                    "additionalProperties": True
                }
            },
            "params": {
                "bsonType": "object",
                "properties": {
                    "p_transit": {"bsonType": ["double", "int", "decimal"]},
                    "p_guess": {"bsonType": ["double", "int", "decimal"]},
                    "p_slip": {"bsonType": ["double", "int", "decimal"]}
                },
                "additionalProperties": True
            }
        },
        "additionalProperties": True
    }
}

QUIZ_LOGS_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["student_id", "question_id", "concept_ids", "correct", "timestamp"],
        "properties": {
            "student_id": {"bsonType": "string"},
            "question_id": {"bsonType": "string"},
            "concept_ids": {"bsonType": "array", "items": {"bsonType": "string"}},
            "correct": {"bsonType": "bool"},
            "response_time": {"bsonType": ["double", "int", "decimal"], "description": "seconds"},
            "difficulty": {"bsonType": ["double", "int", "decimal"], "minimum": 0.0, "maximum": 1.0},
            "timestamp": {"bsonType": "date"},
            "item_meta": {"bsonType": "object", "additionalProperties": True}
        },
        "additionalProperties": True
    }
}

DIFF_HISTORY_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["question_id", "timestamp", "difficulty"],
        "properties": {
            "question_id": {"bsonType": "string"},
            "timestamp": {"bsonType": "date"},
            "difficulty": {"bsonType": ["double", "int", "decimal"], "minimum": 0.0, "maximum": 1.0},
            "note": {"bsonType": "string"}
        },
        "additionalProperties": True
    }
}

COLLECTIONS = {
    "students": STUDENTS_SCHEMA,
    "concepts": CONCEPTS_SCHEMA,
    "mastery": MASTERY_SCHEMA,
    "quiz_logs": QUIZ_LOGS_SCHEMA,
    "difficulty_history": DIFF_HISTORY_SCHEMA
}

INDEXES = {
    "students": [
        ( [("student_id", 1)], {"unique": True, "name": "student_id_unique"} ),
        ( [("created_at", 1)], {"name": "students_created_at_idx"} )
    ],
    "concepts": [
        ( [("concept_id", 1)], {"unique": True, "name": "concept_id_unique"} ),
        ( [("subject", 1), ("grade", 1)], {"name": "concepts_subject_grade_idx"} )
    ],
    "mastery": [
        ( [("student_id", 1), ("concept_id", 1)], {"unique": True, "name": "student_concept_unique"} ),
        ( [("student_id", 1), ("last_updated", -1)], {"name": "mastery_student_lastupdated"} )
    ],
    "quiz_logs": [
        ( [("student_id", 1), ("timestamp", -1)], {"name": "quizlogs_student_ts_idx"} ),
        ( [("question_id", 1)], {"name": "quizlogs_question_idx"} )
    ],
    "difficulty_history": [
        ( [("question_id", 1), ("timestamp", -1)], {"name": "diff_qid_ts_idx"} )
    ]
}


def plan_operations():
    plans = []
    for coll, schema in COLLECTIONS.items():
        plans.append({
            "collection": coll,
            "validator": schema,
            "validationLevel": "moderate",
            "indexes": INDEXES.get(coll, [])
        })
    return plans


def apply_migration(mongo_uri, apply=False, create_samples=False):
    from pymongo.errors import CollectionInvalid, OperationFailure
    client = get_mongo_client(mongo_uri)
    db = client[DB_NAME]

    plans = plan_operations()

    for p in plans:
        coll_name = p["collection"]
        validator = p["validator"]
        validationLevel = p["validationLevel"]
        print(f"--- Processing collection: {coll_name} ---")

        if coll_name in db.list_collection_names():
            print(f"Collection '{coll_name}' exists — updating validator via collMod (moderate).")
            try:
                db.command({
                    "collMod": coll_name,
                    "validator": validator,
                    "validationLevel": validationLevel
                })
                print("Updated validator (collMod)")
            except OperationFailure as e:
                print("Warning: collMod failed:", e)
                print("Skipping validator update for", coll_name)
        else:
            print(f"Creating collection '{coll_name}' with validator (moderate).")
            try:
                db.create_collection(coll_name, validator=validator, validationLevel=validationLevel)
                print("Created collection", coll_name)
            except CollectionInvalid as e:
                print("Could not create collection:", e)
                continue

        # Create indexes
        idxs = p["indexes"]
        for spec, opts in idxs:
            try:
                print("Creating index on", spec, "opts:", opts)
                db[coll_name].create_index(spec, **opts)
            except Exception as e:
                print("Index creation failed:", e)

    if create_samples:
        print("Inserting sample documents...")
        try:
            db.students.insert_one({"student_id": "user_1234", "name": "Anas", "grade": "Grade_7", "created_at": datetime.utcnow(), "meta": {"timezone": "Asia/Kolkata"}})
            db.concepts.insert_one({"concept_id": "photosynthesis_basic", "label": "Photosynthesis (basic)", "subject": "Science", "grade": "Grade_7", "parents": ["plant_physiology"], "examples": ["QID_1001"], "created_at": datetime.utcnow()})
            db.mastery.insert_one({"student_id": "user_1234", "concept_id": "photosynthesis_basic", "p_mastery": 0.23, "last_updated": datetime.utcnow(), "history": [{"ts": datetime.utcnow(), "p_mastery": 0.18}]})
            db.quiz_logs.insert_one({"student_id": "user_1234", "question_id": "q_2025_001", "concept_ids": ["photosynthesis_basic"], "correct": True, "response_time": 12.3, "difficulty": 0.6, "timestamp": datetime.utcnow()})
            print("Sample documents inserted.")
        except Exception as e:
            print("Failed to insert sample docs:", e)

    client.close()
    print("Migration finished.")


def main():
    parser = argparse.ArgumentParser(description="Apply Eduniti MongoDB schema migration (PyMongo).")
    parser.add_argument("--apply", action="store_true", help="Actually apply changes instead of dry-run.")
    parser.add_argument("--sample", action="store_true", help="Insert sample documents after applying.")
    parser.add_argument("--uri", type=str, default=os.environ.get("MONGO_URI"), help="MongoDB connection string. Defaults to MONGO_URI env var.")
    args = parser.parse_args()

    plans = plan_operations()
    print("Planned operations (dry-run):")
    print(json.dumps(plans, indent=2, default=str))

    if not args.apply:
        print("\nDry-run complete. Re-run with --apply to perform changes.")
        return

    if not args.uri:
        print("Error: no MongoDB URI provided. Set MONGO_URI or pass --uri")
        return

    print("Connecting to MongoDB...")
    try:
        apply_migration(args.uri, apply=args.apply, create_samples=args.sample)
    except Exception as e:
        print("Migration failed:", e)


if __name__ == '__main__':
    main()

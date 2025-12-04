# tests/test_mastery_crud.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # two levels up to repo root
sys.path.insert(0, str(ROOT))

from infra import mongo_client_
import os 
import pytest

# Use a dedicated test DB to avoid accidental production changes
TEST_DB = os.getenv("MONGO_DB_TEST", "EduNiti_test")


@pytest.fixture(scope="module")
def db():
    # Return a DB instance, and ensure the collection is clean
    db = mongo_client_.get_db(db_name=TEST_DB) if hasattr(mongo_client_, "get_db") else None
    # Using get_db signature from our module:
    db = mongo_client_.get_db(TEST_DB)
    # Drop the mastery collection if exists (clean slate)
    db.drop_collection("mastery")
    yield db
    # teardown: drop collection
    db.drop_collection("mastery")


def test_create_index_and_upsert(db):
    # create indexes
    created = mongo_client_.create_mastery_index(db=db, unique_by_student_concept=True)
    assert created is True

    student_id = "test_student_1"
    concept_id = "Grade_7__Science__Photosynthesis"
    # set initial mastery
    doc = mongo_client_.set_mastery(student_id, concept_id, 0.35, meta={"seed": True}, source="unit_test", db=db)
    assert doc is not None
    assert doc["student_id"] == student_id
    assert doc["concept_id"] == concept_id
    assert "p_mastery" in doc and isinstance(doc["p_mastery"], float)
    assert doc["p_mastery"] == pytest.approx(0.35)
    assert "last_updated" in doc
    assert "history" in doc and isinstance(doc["history"], list)
    assert doc["history"][-1]["p"] == pytest.approx(0.35)
    assert doc["history"][-1]["source"] == "unit_test"

    # fetch it via get_mastery
    fetched = mongo_client_.get_mastery(student_id, concept_id, db=db)
    assert fetched is not None
    assert fetched["p_mastery"] == pytest.approx(0.35)

    # update mastery higher
    updated = mongo_client_.set_mastery(student_id, concept_id, 0.6, meta={"reason": "quiz1"}, source="quiz", db=db)
    assert updated["p_mastery"] == pytest.approx(0.6)
    assert len(updated["history"]) >= 2
    assert updated["history"][-1]["p"] == pytest.approx(0.6)
    assert updated["history"][-1]["source"] == "quiz"


def test_list_mastery_and_edge_cases(db):
    student_id = "test_student_1"
    # Ensure list returns at least one record
    lst = mongo_client_.list_mastery_by_student(student_id, db=db)
    assert isinstance(lst, list)
    assert len(lst) >= 1

    # Request non-existent
    missing = mongo_client_.get_mastery("no_such_student", "no_such_concept", db=db)
    assert missing is None

    # Validate error on bad inputs
    with pytest.raises(ValueError):
        mongo_client_.set_mastery("", "c", 0.5, db=db)
    with pytest.raises(ValueError):
        mongo_client_.set_mastery("s", "", 0.5, db=db)
    with pytest.raises(ValueError):
        mongo_client_.set_mastery("s", "c", -0.1, db=db)
    with pytest.raises(ValueError):
        mongo_client_.get_mastery("", "c", db=db)
    with pytest.raises(ValueError):
        mongo_client_.list_mastery_by_student("", db=db)

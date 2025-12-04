from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("❌ MONGO_URI not set.")

client = MongoClient(MONGO_URI)
db = client["EduNiti"]

students = db["students"]
quiz_logs = db["quiz_logs"]
mastery = db["mastery_states"]  # BKT storage

def test_db():
    try:
        client.admin.command("ping")
        print("✅ MongoDB connected")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)

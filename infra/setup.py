from mongo import db, mastery, quiz_logs, students

def setup():
    print("📌 Creating indexes...")
    mastery.create_index("student_id")
    mastery.create_index("concept_id")
    quiz_logs.create_index("student_id")
    quiz_logs.create_index("concept_id")
    print("✅ Setup complete.")

if __name__ == "__main__":
    setup()

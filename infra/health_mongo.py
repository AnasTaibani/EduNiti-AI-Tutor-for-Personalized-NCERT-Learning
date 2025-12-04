# infra/health_mongo.py

import sys
from mongo_client import MongoConnection


def health_check():
    try:
        mongo = MongoConnection()
        db = mongo.get_db()

        print("MongoDB Health Check — OK")
        print(f"Connected to database: {db.name}")
        print("Listing collections and indexes:\n")

        collections = db.list_collection_names()
        if not collections:
            print("⚠️ No collections found in database.")

        for coll in collections:
            print(f"📁 Collection: {coll}")
            indexes = db[coll].list_indexes()
            for idx in indexes:
                print(f"   - Index: {idx}")
            print()

        return 0  # success

    except Exception as e:
        print("❌ MongoDB Health Check Failed")
        print("Reason:", str(e))
        return 1  # failure


if __name__ == "__main__":
    exit_code = health_check()
    sys.exit(exit_code)

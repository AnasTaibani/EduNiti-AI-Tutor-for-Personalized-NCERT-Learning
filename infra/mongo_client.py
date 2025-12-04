# infra/mongo_client.py

import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


class MongoConnection:
    def __init__(self, uri=None, db_name="EduNiti"):
        self.uri = uri or os.getenv("MONGO_URI")
        if not self.uri:
            raise ValueError("MONGO_URI not provided. Set env variable MONGO_URI.")
        
        self.db_name = db_name
        self.client = None
        self.db = None

    def connect(self):
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
            # Trigger connection check
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            return self.db
        except ServerSelectionTimeoutError as e:
            raise ConnectionError(f"MongoDB connection failed: {e}")

    def get_db(self):
        if self.db is None:
            return self.connect()
        return self.db

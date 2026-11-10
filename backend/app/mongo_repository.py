"""
mongo_repository.py
--------------------
MongoDB-backed patient repository (Unit 4: MongoDB, pymongo CRUD).

Implements the exact same interface as api.PatientRepository (add, upsert,
get, delete, all) plus MongoDB-specific aggregation queries, so app/api.py
can switch repositories with a one-line change and every route keeps
working unmodified.

Connects to a real MongoDB instance when MONGODB_URI is set (or defaults
to mongodb://localhost:27017). For local development/testing without a
running mongod, pass use_mock=True to use `mongomock`, which implements
the same pymongo API in-memory — the CRUD/aggregation code below is
identical either way; only the client construction differs.
"""

from __future__ import annotations
import os

from app.models import Patient, VitalsRecord, LabRecord, ImagingRecord
from app.exceptions import PatientNotFoundError, DuplicatePatientError

_RECORD_TYPES = {
    "VitalsRecord": VitalsRecord,
    "LabRecord": LabRecord,
    "ImagingRecord": ImagingRecord,
}


def _record_from_dict(d: dict):
    cls = _RECORD_TYPES[d["type"]]
    d = {k: v for k, v in d.items() if k not in ("type", "recorded_on")}
    return cls(**d)


class MongoPatientRepository:
    def __init__(self, uri: str | None = None, db_name: str = "medisense", use_mock: bool = False):
        if use_mock:
            import mongomock
            self._client = mongomock.MongoClient()
        else:
            import pymongo
            uri = uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
            self._client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
        self._db = self._client[db_name]
        self._collection = self._db["patients"]
        self._collection.create_index("patient_id", unique=True)

    # ---- CRUD ----
    def add(self, patient: Patient) -> None:
        if self._collection.find_one({"patient_id": patient.patient_id}):
            raise DuplicatePatientError(patient.patient_id)
        self._collection.insert_one(patient.to_dict())

    def upsert(self, patient: Patient) -> None:
        self._collection.replace_one(
            {"patient_id": patient.patient_id}, patient.to_dict(), upsert=True
        )

    def get(self, patient_id: str) -> Patient:
        doc = self._collection.find_one({"patient_id": patient_id}, {"_id": 0})
        if not doc:
            raise PatientNotFoundError(patient_id)
        return self._doc_to_patient(doc)

    def delete(self, patient_id: str) -> None:
        result = self._collection.delete_one({"patient_id": patient_id})
        if result.deleted_count == 0:
            raise PatientNotFoundError(patient_id)

    def all(self) -> list[Patient]:
        return [self._doc_to_patient(doc) for doc in self._collection.find({}, {"_id": 0})]

    @staticmethod
    def _doc_to_patient(doc: dict) -> Patient:
        patient = Patient(doc["patient_id"], doc["name"], doc["age"], doc["gender"], doc["contact"])
        for record_dict in doc.get("records", []):
            patient.add_record(_record_from_dict(record_dict))
        return patient

    # ---- Aggregation queries ----
    def count_by_gender(self) -> dict:
        pipeline = [{"$group": {"_id": "$gender", "count": {"$sum": 1}}}]
        return {row["_id"]: row["count"] for row in self._collection.aggregate(pipeline)}

    def average_age(self) -> float | None:
        pipeline = [{"$group": {"_id": None, "avg_age": {"$avg": "$age"}}}]
        rows = list(self._collection.aggregate(pipeline))
        return round(rows[0]["avg_age"], 1) if rows else None

    def patients_with_vitals_above(self, systolic_threshold: float) -> list[str]:
        """Aggregation example: patients whose latest vitals record has
        systolic_bp above a threshold — used to seed the ML risk-labeling
        pipeline in ml_pipeline.py."""
        pipeline = [
            {"$unwind": "$records"},
            {"$match": {"records.type": "VitalsRecord", "records.systolic_bp": {"$gte": systolic_threshold}}},
            {"$group": {"_id": "$patient_id"}},
        ]
        return [row["_id"] for row in self._collection.aggregate(pipeline)]

"""
api.py
------
Flask REST API (Unit 3: Backend Introduction — REST principles).

In-memory store for now (Phase 2). Phase 4 swaps PatientRepository's
internals for MongoDB via pymongo without changing any route below —
that's the point of keeping persistence behind a small repository class.

Endpoints:
    GET    /api/health
    GET    /api/patients
    GET    /api/patients/<patient_id>
    POST   /api/patients
    DELETE /api/patients/<patient_id>
    POST   /api/patients/import        (multipart file upload: .xlsx/.csv)
    GET    /api/reference-ranges
"""

from __future__ import annotations
import os
import tempfile

from flask import Flask, jsonify, request
from flask_cors import CORS

from app.models import Patient, VitalsRecord
from app.exceptions import MediSenseError, PatientNotFoundError, DuplicatePatientError
from app.data_import import import_patients_from_file
from app.scraper import scrape_reference_ranges_from_html, SAMPLE_REFERENCE_HTML
from app.mongo_repository import MongoPatientRepository
from app.ml_pipeline import (
    patients_to_dataframe, train_risk_classifier, predict_risk,
    cluster_patients, reduce_to_2d,
)
from app import model_loader


class InMemoryPatientRepository:
    """Fallback in-memory patient store, used only if MongoDB is
    unreachable (e.g. running the demo with no mongod installed).
    Implements the same interface as MongoPatientRepository."""

    def __init__(self):
        self._patients: dict[str, Patient] = {}

    def add(self, patient: Patient) -> None:
        if patient.patient_id in self._patients:
            raise DuplicatePatientError(patient.patient_id)
        self._patients[patient.patient_id] = patient

    def upsert(self, patient: Patient) -> None:
        self._patients[patient.patient_id] = patient

    def get(self, patient_id: str) -> Patient:
        if patient_id not in self._patients:
            raise PatientNotFoundError(patient_id)
        return self._patients[patient_id]

    def delete(self, patient_id: str) -> None:
        if patient_id not in self._patients:
            raise PatientNotFoundError(patient_id)
        del self._patients[patient_id]

    def all(self) -> list[Patient]:
        return list(self._patients.values())


def _build_repository():
    """Try a real MongoDB connection first; fall back to mongomock (in-memory,
    same pymongo API) if no mongod is reachable, so the app still runs on a
    laptop that doesn't have MongoDB installed. Set MEDISENSE_FORCE_MOCK_DB=1
    to skip straight to the mock (useful for CI/tests)."""
    if os.environ.get("MEDISENSE_FORCE_MOCK_DB") == "1":
        return MongoPatientRepository(use_mock=True)
    try:
        repo = MongoPatientRepository()
        repo._client.admin.command("ping")
        return repo
    except Exception:
        return MongoPatientRepository(use_mock=True)


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)  # allow the static dashboard (served separately) to call this API
    repo = _build_repository()

    # Unit 5: load persisted models at startup (trained + saved offline by
    # train_and_save_models.py) so the API doesn't need to retrain in memory
    # on every restart. Falls back to None if nothing has been trained yet —
    # /api/ml/train still works as an in-memory dev/demo path either way.
    app.config["PERSISTED_RISK_MODEL"] = model_loader.load_risk_model()
    app.config["PERSISTED_SCAN_MODEL"] = None  # Disabled for low-memory deployment

    @app.errorhandler(MediSenseError)
    def handle_domain_error(exc: MediSenseError):
        status = 404 if isinstance(exc, PatientNotFoundError) else 400
        return jsonify({"error": exc.__class__.__name__, "message": str(exc)}), status

    @app.get("/api/health")
    def health():
        backend = "mongomock (in-memory)" if getattr(repo, "_client", None).__class__.__module__.startswith("mongomock") else "mongodb"
        return jsonify({"status": "ok", "patients_loaded": len(repo.all()), "db_backend": backend})

    @app.get("/api/stats")
    def stats():
        return jsonify({
            "count_by_gender": repo.count_by_gender(),
            "average_age": repo.average_age(),
            "high_bp_patient_ids": repo.patients_with_vitals_above(140),
        })

    @app.get("/api/patients")
    def list_patients():
        return jsonify([p.to_dict() for p in repo.all()])

    @app.get("/api/patients/<patient_id>")
    def get_patient(patient_id: str):
        return jsonify(repo.get(patient_id).to_dict())

    @app.post("/api/patients")
    def create_patient():
        body = request.get_json(force=True) or {}
        patient = Patient(
            patient_id=body["patient_id"],
            name=body["name"],
            age=int(body["age"]),
            gender=body.get("gender", "U"),
            contact=body.get("contact", "N/A"),
        )
        vitals = body.get("vitals")
        if vitals:
            patient.add_record(VitalsRecord(
                patient_id=patient.patient_id,
                systolic_bp=float(vitals["systolic_bp"]),
                diastolic_bp=float(vitals["diastolic_bp"]),
                heart_rate=float(vitals["heart_rate"]),
                temperature_c=float(vitals["temperature_c"]),
                spo2=float(vitals["spo2"]),
            ))
        repo.add(patient)
        return jsonify(patient.to_dict()), 201

    @app.delete("/api/patients/<patient_id>")
    def delete_patient(patient_id: str):
        repo.delete(patient_id)
        return jsonify({"deleted": patient_id})

    @app.post("/api/patients/import")
    def bulk_import():
        if "file" not in request.files:
            return jsonify({"error": "NoFile", "message": "Attach a file under form field 'file'"}), 400
        upload = request.files["file"]
        suffix = os.path.splitext(upload.filename or "")[1] or ".xlsx"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            upload.save(tmp.name)
            tmp_path = tmp.name
        try:
            report = import_patients_from_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        for p in report.patients:
            repo.upsert(p)

        return jsonify({
            "imported": report.success_count,
            "failed": report.error_count,
            "errors": report.errors,
            "patients": [p.to_dict() for p in report.patients],
        })

    @app.post("/api/ml/train")
    def ml_train():
        df = patients_to_dataframe(repo.all())
        try:
            trained = train_risk_classifier(df)
        except ValueError as exc:
            return jsonify({"error": "InsufficientData", "message": str(exc)}), 400
        app.config["TRAINED_RISK_MODEL"] = trained  # kept in memory for /predict below
        return jsonify({
            "trained_on_patients": len(df),
            "accuracy": round(trained.accuracy, 4),
            "feature_importances": trained.feature_importances,
            "classification_report": trained.report,
        })

    @app.get("/api/ml/model-status")
    def ml_model_status():
        return jsonify({
            "risk_model_persisted": app.config["PERSISTED_RISK_MODEL"] is not None,
            "risk_model_trained_in_memory": app.config.get("TRAINED_RISK_MODEL") is not None,
            "scan_model_persisted": app.config["PERSISTED_SCAN_MODEL"] is not None,
        })

    @app.post("/api/ml/predict")
    def ml_predict():
        body = request.get_json(force=True) or {}
        persisted = app.config["PERSISTED_RISK_MODEL"]
        try:
            if persisted is not None:
                label = model_loader.predict_risk_persisted(persisted, body)
                source = "persisted"
            else:
                trained = app.config.get("TRAINED_RISK_MODEL")
                if trained is None:
                    return jsonify({
                        "error": "ModelNotTrained",
                        "message": "No persisted model found and none trained this session. "
                                   "Run train_and_save_models.py, or call POST /api/ml/train first.",
                    }), 400
                label = predict_risk(trained, body)
                source = "in_memory"
        except KeyError as exc:
            return jsonify({"error": "MissingField", "message": f"Missing field {exc}"}), 400
        return jsonify({"predicted_risk": label, "model_source": source})

    @app.post("/api/ml/predict/scan")
    def ml_predict_scan():
        """Predict NORMAL/ABNORMAL from a 32x32 grayscale pixel grid (values
        0-1), e.g. {"pixels": [[0.3, 0.4, ...], ...]}. Requires a persisted
        scan_classifier.keras (see train_and_save_models.py)."""
        model = app.config["PERSISTED_SCAN_MODEL"]
        if model is None:
            return jsonify({
                "error": "ModelNotTrained",
                "message": "No persisted scan classifier found. Run train_and_save_models.py first.",
            }), 400
        import numpy as np
        from app.dl_pipeline import predict_scan, IMG_SIZE
        body = request.get_json(force=True) or {}
        pixels = body.get("pixels")
        if pixels is None:
            return jsonify({"error": "MissingField", "message": "Expected a 'pixels' 2D array"}), 400
        image = np.array(pixels, dtype="float32")
        if image.shape != (IMG_SIZE, IMG_SIZE):
            return jsonify({"error": "BadShape", "message": f"Expected a {IMG_SIZE}x{IMG_SIZE} array"}), 400
        label, confidence = predict_scan(model, image)
        return jsonify({"prediction": label, "confidence": round(confidence, 4)})

    @app.get("/api/ml/clusters")
    def ml_clusters():
        df = patients_to_dataframe(repo.all())
        if len(df) < 3:
            return jsonify({"error": "InsufficientData", "message": "Need at least 3 patients with vitals"}), 400
        n_clusters = min(3, len(df))
        result = cluster_patients(df, n_clusters=n_clusters)
        return jsonify({
            "patient_ids": df["patient_id"].tolist(),
            "cluster_labels": result.labels.tolist(),
            "inertia": round(float(result.inertia), 2),
        })

    @app.get("/api/ml/pca")
    def ml_pca():
        df = patients_to_dataframe(repo.all())
        if len(df) < 3:
            return jsonify({"error": "InsufficientData", "message": "Need at least 3 patients with vitals"}), 400
        result = reduce_to_2d(df)
        return jsonify({
            "patient_ids": df["patient_id"].tolist(),
            "points": result.components.round(3).tolist(),
            "explained_variance_ratio": result.explained_variance_ratio.round(4).tolist(),
        })

    @app.get("/api/reference-ranges")
    def reference_ranges():
        # In production this would call scrape_reference_ranges_from_url()
        # against a live medical reference site; the offline fixture keeps
        # this endpoint deterministic for demos/tests.
        ranges = scrape_reference_ranges_from_html(SAMPLE_REFERENCE_HTML)
        return jsonify([r.__dict__ for r in ranges])

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)


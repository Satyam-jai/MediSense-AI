"""Tests for the Flask REST API, using Flask's test client + mongomock."""
import os
import pytest

os.environ["MEDISENSE_FORCE_MOCK_DB"] = "1"
from app.api import create_app  # noqa: E402


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_create_and_get_patient(client):
    payload = {
        "patient_id": "P001", "name": "Asha Rao", "age": 45, "gender": "F", "contact": "a@example.com",
        "vitals": {"systolic_bp": 148, "diastolic_bp": 95, "heart_rate": 102, "temperature_c": 37.1, "spo2": 96},
    }
    resp = client.post("/api/patients", json=payload)
    assert resp.status_code == 201

    resp = client.get("/api/patients/P001")
    assert resp.status_code == 200
    assert resp.get_json()["name"] == "Asha Rao"


def test_duplicate_patient_rejected(client):
    payload = {"patient_id": "P002", "name": "Dup", "age": 30, "gender": "M", "contact": "d@example.com"}
    client.post("/api/patients", json=payload)
    resp = client.post("/api/patients", json=payload)
    assert resp.status_code == 400


def test_get_missing_patient_404(client):
    resp = client.get("/api/patients/DOES_NOT_EXIST")
    assert resp.status_code == 404


def test_delete_patient(client):
    payload = {"patient_id": "P003", "name": "ToDelete", "age": 30, "gender": "M", "contact": "d@example.com"}
    client.post("/api/patients", json=payload)
    resp = client.delete("/api/patients/P003")
    assert resp.status_code == 200
    assert client.get("/api/patients/P003").status_code == 404


def test_reference_ranges_endpoint(client):
    resp = client.get("/api/reference-ranges")
    assert resp.status_code == 200
    names = {r["test_name"] for r in resp.get_json()}
    assert "SpO2" in names


def test_model_status_endpoint(client):
    resp = client.get("/api/ml/model-status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "risk_model_persisted" in body

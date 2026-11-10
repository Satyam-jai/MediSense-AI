"""Tests for Units 1-2: OOP domain models, exceptions, file storage."""
import pytest

from app.models import Patient, VitalsRecord, LabRecord
from app.exceptions import InvalidRecordError, DuplicatePatientError
from app.storage import export_patients_csv, save_patients_pickle, load_patients_pickle
from app.concurrency import simulate_concurrent_booking


def test_patient_creation_and_records():
    p = Patient("P001", "Asha Rao", 45, "F", "asha@example.com")
    p.add_record(VitalsRecord("P001", 148, 95, 102, 37.1, 96))
    assert len(p.records) == 1
    assert "HYPERTENSION" in p.all_risk_flags()
    assert "TACHYCARDIA" in p.all_risk_flags()


def test_invalid_age_raises():
    with pytest.raises(InvalidRecordError):
        Patient("P002", "Bad Age", 999, "M", "x@example.com")


def test_record_patient_id_mismatch_raises():
    p = Patient("P003", "Mismatch", 30, "F", "y@example.com")
    with pytest.raises(InvalidRecordError):
        p.add_record(VitalsRecord("P999", 120, 80, 70, 36.6, 98))


def test_lab_record_risk_flags():
    high = LabRecord("P004", "Fasting Glucose", 145, "mg/dL", 70, 100)
    assert high.risk_flags() == ["HIGH_FASTING_GLUCOSE"]
    normal = LabRecord("P004", "Fasting Glucose", 90, "mg/dL", 70, 100)
    assert normal.risk_flags() == []


def test_csv_and_pickle_roundtrip(tmp_path, monkeypatch):
    import app.storage as storage
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    p = Patient("P005", "Roundtrip", 50, "M", "z@example.com")
    p.add_record(VitalsRecord("P005", 118, 76, 72, 36.8, 98))

    export_patients_csv([p], filename="test.csv")
    assert (tmp_path / "test.csv").exists()

    save_patients_pickle([p], filename="test.pkl")
    reloaded = load_patients_pickle(filename="test.pkl")
    assert len(reloaded) == 1
    assert reloaded[0].name == "Roundtrip"


def test_concurrent_booking_only_one_winner():
    report = simulate_concurrent_booking("2026-08-01 09:00", [f"P{i:03d}" for i in range(1, 9)])
    winners = [pid for pid, outcome in report["results"].items() if outcome == "BOOKED"]
    assert len(winners) == 1

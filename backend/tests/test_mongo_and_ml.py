"""Tests for Unit 4: MongoDB (mock) repository, ML pipeline."""
import pytest

from app.mongo_repository import MongoPatientRepository
from app.models import Patient, VitalsRecord
from app.exceptions import PatientNotFoundError
from app.ml_pipeline import patients_to_dataframe, train_risk_classifier, cluster_patients, reduce_to_2d


@pytest.fixture
def repo_with_patients():
    repo = MongoPatientRepository(use_mock=True)
    for i in range(1, 21):
        p = Patient(f"P{i:03d}", f"Patient {i}", 20 + i, "M" if i % 2 else "F", "x@example.com")
        p.add_record(VitalsRecord(f"P{i:03d}", 100 + i, 70 + (i % 20), 60 + (i % 40),
                                   36.5 + (i % 3) * 0.4, 92 + (i % 8)))
        repo.upsert(p)
    return repo


def test_mongo_repository_crud():
    repo = MongoPatientRepository(use_mock=True)
    p = Patient("P001", "Test Patient", 40, "F", "t@example.com")
    repo.add(p)
    assert repo.get("P001").name == "Test Patient"
    repo.delete("P001")
    with pytest.raises(PatientNotFoundError):
        repo.get("P001")


def test_mongo_aggregations(repo_with_patients):
    assert repo_with_patients.average_age() > 0
    genders = repo_with_patients.count_by_gender()
    assert set(genders.keys()) <= {"M", "F"}


def test_ml_pipeline_train_and_cluster(repo_with_patients):
    df = patients_to_dataframe(repo_with_patients.all())
    assert len(df) == 20

    trained = train_risk_classifier(df)
    assert 0.0 <= trained.accuracy <= 1.0

    clusters = cluster_patients(df, n_clusters=3)
    assert len(clusters.labels) == 20

    pca = reduce_to_2d(df)
    assert pca.components.shape == (20, 2)

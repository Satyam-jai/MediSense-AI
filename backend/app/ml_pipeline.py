"""
ml_pipeline.py
---------------
Machine learning layer (Unit 4: Data Science + Machine Learning).

Given a set of patients, this module:
  1. Builds a clean NumPy/Pandas feature matrix from their vitals
  2. Labels each patient's risk level using the same clinical rules as
     Patient.all_risk_flags() (so training labels agree with the rule-based
     flags already shown on the dashboard)
  3. Trains a classifier (RandomForest) to predict risk level from vitals
  4. Runs K-Means clustering to discover natural patient groupings
  5. Runs PCA to reduce the vitals feature space to 2D for visualization

This mirrors the "compare Scikit-learn classifiers on Iris dataset" style
exercise from the syllabus, but applied to our own domain's data instead
of a toy dataset.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from app.models import Patient

FEATURE_COLUMNS = ["systolic_bp", "diastolic_bp", "heart_rate", "temperature_c", "spo2", "age"]
RISK_LABELS = ["Low", "Medium", "High"]


def patients_to_dataframe(patients: list[Patient]) -> pd.DataFrame:
    """Flatten each patient's latest vitals + age into one feature row.
    Patients with no vitals record are skipped (nothing to score)."""
    rows = []
    for p in patients:
        vitals = next((r for r in p.records if r.__class__.__name__ == "VitalsRecord"), None)
        if vitals is None:
            continue
        rows.append({
            "patient_id": p.patient_id,
            "age": p.age,
            "systolic_bp": vitals.systolic_bp,
            "diastolic_bp": vitals.diastolic_bp,
            "heart_rate": vitals.heart_rate,
            "temperature_c": vitals.temperature_c,
            "spo2": vitals.spo2,
            "risk_label": _rule_based_label(vitals),
        })
    return pd.DataFrame(rows)


def _rule_based_label(vitals) -> str:
    """Same clinical logic as VitalsRecord.risk_flags(), collapsed into a
    single ordinal label used to train/evaluate the classifier against."""
    flags = vitals.risk_flags()
    danger = {"HYPERTENSION", "LOW_SPO2", "TACHYCARDIA"}
    if any(f in danger for f in flags):
        return "High"
    if flags:
        return "Medium"
    return "Low"


@dataclass
class TrainedRiskModel:
    model: RandomForestClassifier
    scaler: StandardScaler
    accuracy: float
    report: str
    feature_importances: dict


def train_risk_classifier(df: pd.DataFrame, random_state: int = 42) -> TrainedRiskModel:
    """Train a RandomForest classifier to predict risk_label from vitals+age.
    Requires at least ~10 rows with more than one class present to give a
    meaningful train/test split; raises ValueError otherwise."""
    if len(df) < 8 or df["risk_label"].nunique() < 2:
        raise ValueError("Not enough varied data to train a classifier (need >=8 rows, >=2 classes)")

    X = df[FEATURE_COLUMNS]
    y = df["risk_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_counts = y.value_counts()
    can_stratify = class_counts.min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=random_state,
        stratify=y if can_stratify else None,
    )

    clf = RandomForestClassifier(n_estimators=150, random_state=random_state, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    importances = dict(zip(FEATURE_COLUMNS, clf.feature_importances_.round(3)))

    return TrainedRiskModel(model=clf, scaler=scaler, accuracy=accuracy, report=report, feature_importances=importances)


def predict_risk(trained: TrainedRiskModel, vitals_row: dict) -> str:
    """Predict a risk label for a single new patient's vitals+age."""
    X = pd.DataFrame([vitals_row])[FEATURE_COLUMNS]
    X_scaled = trained.scaler.transform(X)
    return trained.model.predict(X_scaled)[0]


@dataclass
class ClusteringResult:
    labels: np.ndarray
    cluster_centers: np.ndarray
    inertia: float


def cluster_patients(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> ClusteringResult:
    """K-Means clustering over standardized vitals — groups patients by
    overall physiological profile rather than a single rule-based flag."""
    X = df[FEATURE_COLUMNS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return ClusteringResult(labels=labels, cluster_centers=km.cluster_centers_, inertia=km.inertia_)


@dataclass
class PCAResult:
    components: np.ndarray          # (n_samples, 2)
    explained_variance_ratio: np.ndarray


def reduce_to_2d(df: pd.DataFrame, random_state: int = 42) -> PCAResult:
    """PCA down to 2 principal components, for plotting patients on a
    simple risk map (used by the /api/ml/pca endpoint + dashboard chart)."""
    X = df[FEATURE_COLUMNS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(X_scaled)
    return PCAResult(components=components, explained_variance_ratio=pca.explained_variance_ratio_)

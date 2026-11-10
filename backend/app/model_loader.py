"""
model_loader.py
-----------------
Loads persisted models (trained + saved by train_and_save_models.py) from
disk at API startup, so predictions don't require retraining in memory on
every process restart. Falls back gracefully — with a clear error surfaced
through the API — if a model hasn't been trained/saved yet.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class LoadedRiskModel:
    model: object
    scaler: object


def load_risk_model() -> Optional[LoadedRiskModel]:
    model_path = MODELS_DIR / "risk_classifier.joblib"
    scaler_path = MODELS_DIR / "risk_scaler.joblib"
    if not (model_path.exists() and scaler_path.exists()):
        return None
    return LoadedRiskModel(model=joblib.load(model_path), scaler=joblib.load(scaler_path))


def predict_risk_persisted(loaded: LoadedRiskModel, vitals_row: dict) -> str:
    from app.ml_pipeline import FEATURE_COLUMNS
    X = pd.DataFrame([vitals_row])[FEATURE_COLUMNS]
    X_scaled = loaded.scaler.transform(X)
    return loaded.model.predict(X_scaled)[0]


def load_scan_model():
    """Lazily import TensorFlow only if a saved scan model actually exists —
    keeps API startup fast when the imaging model hasn't been trained yet."""
    model_path = MODELS_DIR / "scan_classifier.keras"
    if not model_path.exists():
        return None
    from tensorflow import keras
    return keras.models.load_model(model_path)

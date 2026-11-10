"""
storage.py
----------
File-based persistence layer (Unit 2: File Handling).

Demonstrates:
- Context managers for safe file open/close
- CSV read/write for tabular patient summaries
- Binary serialization (pickle) for full Patient objects (with nested
  MedicalRecord objects) so state survives across runs without a database
- Custom exception translation (FileProcessingError)

Note: MongoDB (pymongo) becomes the primary store in Unit 4; this module
remains useful for local export/import and quick CLI demos.
"""

from __future__ import annotations
import csv
import pickle
from pathlib import Path
from typing import Iterable

from app.models import Patient
from app.exceptions import FileProcessingError

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def export_patients_csv(patients: Iterable[Patient], filename: str = "patients_summary.csv") -> Path:
    """Write a flat CSV summary (one row per patient) — good for quick review
    in Excel, or as input to the openpyxl/pandas pipeline in Unit 3."""
    path = DATA_DIR / filename
    fieldnames = ["patient_id", "name", "age", "gender", "contact", "num_records", "risk_flags"]
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in patients:
                writer.writerow({
                    "patient_id": p.patient_id,
                    "name": p.name,
                    "age": p.age,
                    "gender": p.gender,
                    "contact": p.contact,
                    "num_records": len(p.records),
                    "risk_flags": ";".join(p.all_risk_flags()),
                })
    except OSError as exc:
        raise FileProcessingError(f"Failed to write CSV to {path}: {exc}") from exc
    return path


def import_patients_csv(filename: str = "patients_summary.csv") -> list[dict]:
    """Read patient summary rows back out of a CSV file."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileProcessingError(f"CSV file not found: {path}")
    try:
        with open(path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except OSError as exc:
        raise FileProcessingError(f"Failed to read CSV from {path}: {exc}") from exc


def save_patients_pickle(patients: Iterable[Patient], filename: str = "patients.pkl") -> Path:
    """Serialize full Patient objects (including nested record objects)."""
    path = DATA_DIR / filename
    try:
        with open(path, "wb") as f:
            pickle.dump(list(patients), f)
    except (OSError, pickle.PickleError) as exc:
        raise FileProcessingError(f"Failed to pickle patients to {path}: {exc}") from exc
    return path


def load_patients_pickle(filename: str = "patients.pkl") -> list[Patient]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileProcessingError(f"Pickle file not found: {path}")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError) as exc:
        raise FileProcessingError(f"Failed to load pickle from {path}: {exc}") from exc

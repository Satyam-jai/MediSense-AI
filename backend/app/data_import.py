"""
data_import.py
---------------
Bulk patient-data ingestion (Unit 3: Data Automation).

Reads an Excel workbook (or CSV) of raw patient/vitals rows, cleans it
with pandas, validates each row, and converts it into Patient / VitalsRecord
domain objects from Unit 1. Rows that fail validation are collected as
errors rather than crashing the whole import (partial success is reported).

Expected input columns:
    patient_id, name, age, gender, contact,
    systolic_bp, diastolic_bp, heart_rate, temperature_c, spo2
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd

from app.models import Patient, VitalsRecord
from app.exceptions import InvalidRecordError, FileProcessingError

REQUIRED_COLUMNS = [
    "patient_id", "name", "age", "gender", "contact",
    "systolic_bp", "diastolic_bp", "heart_rate", "temperature_c", "spo2",
]


@dataclass
class ImportReport:
    patients: list[Patient] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.patients)

    @property
    def error_count(self) -> int:
        return len(self.errors)


def _read_raw(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path, engine="openpyxl")
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise FileProcessingError(f"Unsupported file type: {suffix}")
    except (OSError, ValueError) as exc:
        raise FileProcessingError(f"Failed to read {path}: {exc}") from exc
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """pandas-based cleaning: trim whitespace, drop fully-empty rows,
    coerce numeric columns, fill missing contact with a placeholder."""
    df = df.dropna(how="all").copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise FileProcessingError(f"Missing required columns: {missing}")

    for col in ["name", "gender", "contact", "patient_id"]:
        df[col] = df[col].astype(str).str.strip()

    numeric_cols = ["age", "systolic_bp", "diastolic_bp", "heart_rate", "temperature_c", "spo2"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["contact"] = df["contact"].replace({"nan": "N/A", "": "N/A"})
    return df


def import_patients_from_file(path: str | Path) -> ImportReport:
    """
    Bulk-import patients from an Excel or CSV file.
    Returns an ImportReport with successfully built Patient objects and
    a list of human-readable error strings for rows that failed validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileProcessingError(f"File not found: {path}")

    raw = _read_raw(path)
    cleaned = _clean(raw)

    report = ImportReport()
    for idx, row in cleaned.iterrows():
        row_num = idx + 2  # +1 for 0-index, +1 for header row
        try:
            if pd.isna(row["age"]):
                raise InvalidRecordError("age", "non-numeric or missing")
            if not row["patient_id"] or row["patient_id"] == "nan":
                raise InvalidRecordError("patient_id", "missing")

            patient = Patient(
                patient_id=row["patient_id"],
                name=row["name"],
                age=int(row["age"]),
                gender=row["gender"],
                contact=row["contact"],
            )

            if not pd.isna(row["systolic_bp"]):
                vitals = VitalsRecord(
                    patient_id=row["patient_id"],
                    systolic_bp=float(row["systolic_bp"]),
                    diastolic_bp=float(row["diastolic_bp"]),
                    heart_rate=float(row["heart_rate"]),
                    temperature_c=float(row["temperature_c"]),
                    spo2=float(row["spo2"]),
                )
                patient.add_record(vitals)

            report.patients.append(patient)
        except InvalidRecordError as exc:
            report.errors.append(f"Row {row_num} ({row.get('patient_id', '?')}): {exc}")
        except (ValueError, TypeError) as exc:
            report.errors.append(f"Row {row_num} ({row.get('patient_id', '?')}): {exc}")

    return report


def generate_sample_workbook(path: str | Path, n: int = 12) -> Path:
    """Generate a small sample Excel workbook for demo/testing purposes."""
    import random

    path = Path(path)
    random.seed(42)
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "patient_id": f"P{i:03d}",
            "name": f"Patient {i}",
            "age": random.randint(18, 80),
            "gender": random.choice(["M", "F"]),
            "contact": f"patient{i}@example.com",
            "systolic_bp": random.randint(100, 150),
            "diastolic_bp": random.randint(65, 95),
            "heart_rate": random.randint(58, 105),
            "temperature_c": round(random.uniform(36.2, 38.3), 1),
            "spo2": random.randint(93, 100),
        })
    # inject one intentionally bad row to demonstrate partial-failure handling
    rows.append({
        "patient_id": "", "name": "Broken Row", "age": "not-a-number",
        "gender": "F", "contact": "broken@example.com",
        "systolic_bp": 120, "diastolic_bp": 80, "heart_rate": 70,
        "temperature_c": 36.9, "spo2": 98,
    })
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, engine="openpyxl")
    return path

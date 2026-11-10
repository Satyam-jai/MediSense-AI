"""
models.py
---------
Core domain layer for MediSense AI (Unit 1: OOP Foundations).

Demonstrates:
- Encapsulation: private attributes exposed via properties with validation
- Inheritance: MedicalRecord (ABC) -> VitalsRecord, LabRecord, ImagingRecord
- Polymorphism: each record type overrides summary() and risk_flags()
- Abstraction: MedicalRecord defines the contract every record type must
  implement, using Python's abc module
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional

from app.exceptions import InvalidRecordError


# ---------------------------------------------------------------------------
# Abstract base: MedicalRecord
# ---------------------------------------------------------------------------
class MedicalRecord(ABC):
    """Abstract base class for every kind of medical record we store."""

    def __init__(self, patient_id: str, recorded_on: Optional[date] = None):
        if not patient_id:
            raise InvalidRecordError("patient_id", "must not be empty")
        self._patient_id = patient_id
        self._recorded_on = recorded_on or date.today()

    @property
    def patient_id(self) -> str:
        return self._patient_id

    @property
    def recorded_on(self) -> date:
        return self._recorded_on

    @abstractmethod
    def summary(self) -> str:
        """Human-readable one-line summary of this record."""
        raise NotImplementedError

    @abstractmethod
    def risk_flags(self) -> list[str]:
        """Return a list of risk flags this record raises, if any."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Base serialization; subclasses extend this."""
        return {
            "type": self.__class__.__name__,
            "patient_id": self._patient_id,
            "recorded_on": self._recorded_on.isoformat(),
        }


class VitalsRecord(MedicalRecord):
    """Blood pressure, heart rate, temperature, SpO2 readings."""

    def __init__(self, patient_id: str, systolic_bp: float, diastolic_bp: float,
                 heart_rate: float, temperature_c: float, spo2: float,
                 recorded_on: Optional[date] = None):
        super().__init__(patient_id, recorded_on)
        self.systolic_bp = systolic_bp
        self.diastolic_bp = diastolic_bp
        self.heart_rate = heart_rate
        self.temperature_c = temperature_c
        self.spo2 = spo2

    def summary(self) -> str:
        return (f"Vitals[{self.patient_id}] BP {self.systolic_bp}/{self.diastolic_bp}, "
                f"HR {self.heart_rate}, Temp {self.temperature_c}C, SpO2 {self.spo2}%")

    def risk_flags(self) -> list[str]:
        flags = []
        if self.systolic_bp >= 140 or self.diastolic_bp >= 90:
            flags.append("HYPERTENSION")
        if self.heart_rate > 100:
            flags.append("TACHYCARDIA")
        if self.heart_rate < 60:
            flags.append("BRADYCARDIA")
        if self.temperature_c >= 38.0:
            flags.append("FEVER")
        if self.spo2 < 94:
            flags.append("LOW_SPO2")
        return flags

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "systolic_bp": self.systolic_bp,
            "diastolic_bp": self.diastolic_bp,
            "heart_rate": self.heart_rate,
            "temperature_c": self.temperature_c,
            "spo2": self.spo2,
        })
        return d


class LabRecord(MedicalRecord):
    """A single lab test result, e.g. glucose, cholesterol, HbA1c."""

    def __init__(self, patient_id: str, test_name: str, value: float, unit: str,
                 reference_low: float, reference_high: float,
                 recorded_on: Optional[date] = None):
        super().__init__(patient_id, recorded_on)
        if not test_name:
            raise InvalidRecordError("test_name", "must not be empty")
        self.test_name = test_name
        self.value = value
        self.unit = unit
        self.reference_low = reference_low
        self.reference_high = reference_high

    def summary(self) -> str:
        return f"Lab[{self.patient_id}] {self.test_name}: {self.value} {self.unit}"

    def risk_flags(self) -> list[str]:
        if self.value < self.reference_low:
            return [f"LOW_{self.test_name.upper().replace(' ', '_')}"]
        if self.value > self.reference_high:
            return [f"HIGH_{self.test_name.upper().replace(' ', '_')}"]
        return []

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "test_name": self.test_name,
            "value": self.value,
            "unit": self.unit,
            "reference_low": self.reference_low,
            "reference_high": self.reference_high,
        })
        return d


class ImagingRecord(MedicalRecord):
    """Metadata + file path for an imaging scan (X-ray, MRI, etc.),
    later consumed by the Keras CNN classifier in Unit 4/5."""

    def __init__(self, patient_id: str, modality: str, file_path: str,
                 recorded_on: Optional[date] = None):
        super().__init__(patient_id, recorded_on)
        if modality not in {"XRAY", "MRI", "CT", "ULTRASOUND"}:
            raise InvalidRecordError("modality", f"unsupported modality '{modality}'")
        self.modality = modality
        self.file_path = file_path
        self.prediction: Optional[str] = None  # filled in later by the ML layer
        self.confidence: Optional[float] = None

    def summary(self) -> str:
        pred = f", prediction={self.prediction} ({self.confidence:.2%})" if self.prediction else ""
        return f"Imaging[{self.patient_id}] {self.modality} -> {self.file_path}{pred}"

    def risk_flags(self) -> list[str]:
        if self.prediction and self.prediction.upper() != "NORMAL":
            return [f"IMAGING_{self.prediction.upper()}"]
        return []

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "modality": self.modality,
            "file_path": self.file_path,
            "prediction": self.prediction,
            "confidence": self.confidence,
        })
        return d


# ---------------------------------------------------------------------------
# Patient / Doctor / Appointment
# ---------------------------------------------------------------------------
class Person:
    """Base class for anyone in the system (encapsulated name/contact)."""

    def __init__(self, name: str, contact: str):
        self._name = name
        self._contact = contact

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        if not value.strip():
            raise InvalidRecordError("name", "must not be empty")
        self._name = value

    @property
    def contact(self) -> str:
        return self._contact


class Patient(Person):
    """A patient and their collection of medical records."""

    def __init__(self, patient_id: str, name: str, age: int, gender: str, contact: str):
        super().__init__(name, contact)
        if age < 0 or age > 130:
            raise InvalidRecordError("age", "must be between 0 and 130")
        self._patient_id = patient_id
        self.age = age
        self.gender = gender
        self._records: list[MedicalRecord] = []

    @property
    def patient_id(self) -> str:
        return self._patient_id

    def add_record(self, record: MedicalRecord) -> None:
        if record.patient_id != self._patient_id:
            raise InvalidRecordError("patient_id", "record does not belong to this patient")
        self._records.append(record)

    @property
    def records(self) -> list[MedicalRecord]:
        return list(self._records)  # return a copy to preserve encapsulation

    def all_risk_flags(self) -> list[str]:
        """Polymorphism in action: each record type computes its own flags."""
        flags: list[str] = []
        for r in self._records:
            flags.extend(r.risk_flags())
        return flags

    def to_dict(self) -> dict:
        return {
            "patient_id": self._patient_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "contact": self.contact,
            "records": [r.to_dict() for r in self._records],
        }


class Doctor(Person):
    def __init__(self, doctor_id: str, name: str, specialty: str, contact: str):
        super().__init__(name, contact)
        self.doctor_id = doctor_id
        self.specialty = specialty


class Appointment:
    def __init__(self, appointment_id: str, patient_id: str, doctor_id: str,
                 slot: str, created_at: Optional[datetime] = None):
        self.appointment_id = appointment_id
        self.patient_id = patient_id
        self.doctor_id = doctor_id
        self.slot = slot  # e.g. "2026-07-25 10:00"
        self.created_at = created_at or datetime.now()

    def __repr__(self) -> str:
        return (f"Appointment({self.appointment_id}, patient={self.patient_id}, "
                f"doctor={self.doctor_id}, slot={self.slot})")

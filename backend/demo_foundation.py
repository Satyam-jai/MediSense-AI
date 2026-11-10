"""
demo_foundation.py
-------------------
Runnable end-to-end demo of the Unit 1 & 2 foundation layer:
OOP models, exception handling, file persistence, and concurrency.

Run with:
    cd backend
    python demo_foundation.py
"""

from datetime import date

from app.models import Patient, VitalsRecord, LabRecord, ImagingRecord
from app.exceptions import MediSenseError, InvalidRecordError
from app.storage import export_patients_csv, save_patients_pickle, load_patients_pickle
from app.concurrency import simulate_concurrent_booking


def build_sample_patients() -> list[Patient]:
    patients = []

    p1 = Patient("P001", "Asha Rao", 45, "F", "asha@example.com")
    p1.add_record(VitalsRecord("P001", systolic_bp=148, diastolic_bp=95,
                                heart_rate=102, temperature_c=37.1, spo2=96))
    p1.add_record(LabRecord("P001", "Fasting Glucose", 145, "mg/dL", 70, 100))
    p1.add_record(ImagingRecord("P001", "XRAY", "scans/p001_chest.png"))
    patients.append(p1)

    p2 = Patient("P002", "Rahul Nair", 29, "M", "rahul@example.com")
    p2.add_record(VitalsRecord("P002", systolic_bp=118, diastolic_bp=76,
                                heart_rate=72, temperature_c=36.8, spo2=98))
    p2.add_record(LabRecord("P002", "Cholesterol", 190, "mg/dL", 125, 200))
    patients.append(p2)

    return patients


def demo_exception_handling():
    print("\n--- Exception handling demo ---")
    try:
        Patient("P003", "Bad Age", age=999, gender="M", contact="x@example.com")
    except InvalidRecordError as exc:
        print(f"Caught expected error: {exc}")

    try:
        p = Patient("P004", "Wrong Owner", 30, "F", "y@example.com")
        p.add_record(VitalsRecord("P999", 120, 80, 70, 36.6, 98))  # mismatched ID
    except MediSenseError as exc:
        print(f"Caught expected error: {exc}")


def main():
    print("--- Building sample patients (OOP: encapsulation/inheritance/polymorphism) ---")
    patients = build_sample_patients()
    for p in patients:
        print(f"\n{p.name} ({p.patient_id}), age {p.age}")
        for r in p.records:
            print("  ", r.summary())
        print("   Risk flags:", p.all_risk_flags() or "None")

    demo_exception_handling()

    print("\n--- File persistence demo (CSV + pickle) ---")
    csv_path = export_patients_csv(patients)
    print("Wrote CSV summary to:", csv_path)
    pkl_path = save_patients_pickle(patients)
    print("Wrote pickle snapshot to:", pkl_path)
    reloaded = load_patients_pickle()
    print(f"Reloaded {len(reloaded)} patients from pickle:",
          [p.name for p in reloaded])

    print("\n--- Concurrency demo (thread-safe slot booking) ---")
    report = simulate_concurrent_booking(
        "2026-07-25 10:00", [p.patient_id for p in patients] + ["P005", "P006"]
    )
    print("Booking report:", report)


if __name__ == "__main__":
    main()

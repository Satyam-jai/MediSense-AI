"""
train_and_save_models.py
--------------------------
Unit 5: Model persistence for deployment.

Trains the RandomForest risk classifier (on the bundled sample workbook,
or any Excel/CSV file passed on the command line) and the Keras scan
classifier, then saves both to disk under backend/models/ so the API can
load them at startup instead of retraining in memory on every request.

Run with:
    cd backend
    python train_and_save_models.py [path/to/patients.xlsx]
"""

from __future__ import annotations
import sys
import json
import joblib
from pathlib import Path

from app.data_import import import_patients_from_file, generate_sample_workbook
from app.ml_pipeline import patients_to_dataframe, train_risk_classifier
from app.dl_pipeline import train_scan_classifier

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else None
    if source is None:
        print("No input file given — generating a sample workbook for training...")
        source = generate_sample_workbook("data/sample_patients.xlsx", n=60)

    print(f"--- Training risk classifier from {source} ---")
    report = import_patients_from_file(source)
    print(f"Loaded {report.success_count} patients ({report.error_count} rows failed validation)")

    df = patients_to_dataframe(report.patients)
    trained = train_risk_classifier(df)
    print(f"Test accuracy: {trained.accuracy:.2%}")
    print(trained.report)

    joblib.dump(trained.model, MODELS_DIR / "risk_classifier.joblib")
    joblib.dump(trained.scaler, MODELS_DIR / "risk_scaler.joblib")
    with open(MODELS_DIR / "risk_classifier_metadata.json", "w") as f:
        json.dump({
            "accuracy": round(trained.accuracy, 4),
            "feature_importances": trained.feature_importances,
            "trained_on_patients": len(df),
        }, f, indent=2)
    print(f"Saved risk_classifier.joblib + risk_scaler.joblib to {MODELS_DIR}")

    print("\n--- Training scan classifier (Keras CNN) ---")
    scan_result = train_scan_classifier(epochs=10, n_per_class=250)
    print(f"Test accuracy: {scan_result.test_accuracy:.2%}, test loss: {scan_result.test_loss:.4f}")
    scan_result.model.save(MODELS_DIR / "scan_classifier.keras")
    print(f"Saved scan_classifier.keras to {MODELS_DIR}")

    print("\nAll models trained and persisted. Restart the API to load them.")


if __name__ == "__main__":
    main()

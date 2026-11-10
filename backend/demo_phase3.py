"""
demo_phase3.py
---------------
Runnable demo of the Unit 4 layer: MongoDB persistence, NumPy/Pandas
feature prep, Scikit-learn classification/clustering/PCA, and a Keras CNN
(both the syllabus's MNIST reference exercise and the applied scan
classifier).

Run with:
    cd backend
    python demo_phase3.py
"""

from app.mongo_repository import MongoPatientRepository
from app.data_import import import_patients_from_file, generate_sample_workbook
from app.ml_pipeline import (
    patients_to_dataframe, train_risk_classifier, predict_risk,
    cluster_patients, reduce_to_2d,
)
from app.dl_pipeline import train_scan_classifier, generate_synthetic_scan_dataset, predict_scan


def main():
    print("--- MongoDB (mongomock) CRUD + aggregation ---")
    repo = MongoPatientRepository(use_mock=True)
    workbook = generate_sample_workbook("data/sample_patients.xlsx", n=40)
    report = import_patients_from_file(workbook)
    for p in report.patients:
        repo.upsert(p)
    print(f"Loaded {len(repo.all())} patients into MongoDB (mock)")
    print("Count by gender:", repo.count_by_gender())
    print("Average age:", repo.average_age())

    print("\n--- Feature prep (Pandas/NumPy) ---")
    df = patients_to_dataframe(repo.all())
    print(df[["patient_id", "systolic_bp", "heart_rate", "spo2", "risk_label"]].head())
    print("Risk label distribution:\n", df["risk_label"].value_counts())

    print("\n--- Scikit-learn: RandomForest risk classifier ---")
    trained = train_risk_classifier(df)
    print(f"Test accuracy: {trained.accuracy:.2%}")
    print("Feature importances:", trained.feature_importances)
    sample_prediction = predict_risk(trained, {
        "systolic_bp": 150, "diastolic_bp": 96, "heart_rate": 108,
        "temperature_c": 37.0, "spo2": 95, "age": 60,
    })
    print("Prediction for a high-BP/high-HR patient:", sample_prediction)

    print("\n--- Scikit-learn: K-Means clustering ---")
    clustering = cluster_patients(df, n_clusters=3)
    print("Cluster assignment counts:", dict(zip(*__import__("numpy").unique(clustering.labels, return_counts=True))))
    print("Inertia:", round(clustering.inertia, 2))

    print("\n--- Scikit-learn: PCA to 2D ---")
    pca_result = reduce_to_2d(df)
    print("Explained variance ratio (2 components):", pca_result.explained_variance_ratio.round(3))
    print("First 3 points in PCA space:\n", pca_result.components[:3].round(2))

    print("\n--- Keras: synthetic scan classifier (NORMAL vs ABNORMAL) ---")
    result = train_scan_classifier(epochs=6, n_per_class=150)
    print(f"Test accuracy: {result.test_accuracy:.2%}, test loss: {result.test_loss:.4f}")

    X, y = generate_synthetic_scan_dataset(n_per_class=1, seed=99)
    label, confidence = predict_scan(result.model, X[0, :, :, 0])
    print(f"Sample prediction: {label} (confidence {confidence:.2%}), true label was {'ABNORMAL' if y[0]==1 else 'NORMAL'}")


if __name__ == "__main__":
    main()

"""
dl_pipeline.py
---------------
Deep learning layer (Unit 4: Neural networks using TensorFlow/Keras).

The syllabus's own worked example is "build and train an MNIST digit
recognition model using Keras Sequential API" — we build exactly that
model architecture here, then repurpose it as the diagnostic-imaging
classifier referenced by ImagingRecord (Unit 1) and the capstone project
(Unit 5): instead of digits 0-9, the same CNN shape classifies scan
patches as NORMAL vs ABNORMAL.

Two entry points:
  - build_mnist_digit_model() / train_on_mnist(): the syllabus's reference
    exercise, trained on real MNIST via keras.datasets.
  - build_scan_classifier_model() / train_scan_classifier(): the same
    architecture repurposed for binary NORMAL/ABNORMAL imaging
    classification, trained on synthetic scan-like patches (in place of a
    real, licensed medical imaging dataset) so the pipeline is fully
    runnable and demo-able end to end without external data downloads.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Reference exercise: MNIST digit recognition (Keras Sequential API)
# ---------------------------------------------------------------------------
def build_mnist_digit_model() -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_on_mnist(epochs: int = 3, subset: int | None = 4000) -> dict:
    """Trains on real MNIST (auto-downloaded by Keras on first run).
    `subset` caps the training set size to keep the demo fast; set to
    None to train on the full 60k images."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    if subset:
        x_train, y_train = x_train[:subset], y_train[:subset]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = build_mnist_digit_model()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=64,
                         validation_split=0.1, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    return {
        "model": model,
        "final_train_accuracy": round(history.history["accuracy"][-1], 4),
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
    }


# ---------------------------------------------------------------------------
# Applied version: NORMAL vs ABNORMAL scan classification
# ---------------------------------------------------------------------------
IMG_SIZE = 32


def build_scan_classifier_model() -> keras.Model:
    """Same Sequential CNN shape as the MNIST model, adapted to a binary
    output — this is the model referenced by ImagingRecord.prediction."""
    model = keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(16, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _generate_synthetic_scan(rng: np.random.Generator, abnormal: bool) -> np.ndarray:
    """
    Generates a synthetic grayscale "scan" patch standing in for a real
    (licensed, patient-consented) medical image, which this project does
    not have access to. NORMAL patches are smooth Gaussian-blob textures;
    ABNORMAL patches have an injected high-intensity lesion-like spot —
    giving the CNN a learnable, non-trivial signal for demo purposes.
    """
    base = rng.normal(loc=0.35, scale=0.05, size=(IMG_SIZE, IMG_SIZE))
    if abnormal:
        cx, cy = rng.integers(10, IMG_SIZE - 10, size=2)
        radius = rng.integers(5, 9)
        yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        base[mask] += 0.5  # strong, consistent lesion contrast (was noisy/weak before)
    return np.clip(base, 0, 1)


def generate_synthetic_scan_dataset(n_per_class: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for _ in range(n_per_class):
        images.append(_generate_synthetic_scan(rng, abnormal=False))
        labels.append(0)
        images.append(_generate_synthetic_scan(rng, abnormal=True))
        labels.append(1)
    X = np.expand_dims(np.array(images, dtype="float32"), -1)
    y = np.array(labels, dtype="float32")
    # shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


@dataclass
class ScanClassifierResult:
    model: keras.Model
    test_accuracy: float
    test_loss: float


def train_scan_classifier(epochs: int = 8, n_per_class: int = 300) -> ScanClassifierResult:
    X, y = generate_synthetic_scan_dataset(n_per_class=n_per_class)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_scan_classifier_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    return ScanClassifierResult(model=model, test_accuracy=round(test_acc, 4), test_loss=round(test_loss, 4))


def predict_scan(model: keras.Model, image: np.ndarray) -> tuple[str, float]:
    """image: (IMG_SIZE, IMG_SIZE) grayscale array, values in [0, 1]."""
    x = image.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype("float32")
    prob = float(model.predict(x, verbose=0)[0][0])
    label = "ABNORMAL" if prob >= 0.5 else "NORMAL"
    confidence = prob if label == "ABNORMAL" else 1 - prob
    return label, confidence

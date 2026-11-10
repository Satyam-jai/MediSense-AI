# MediSense AI — AI-Integrated Patient Risk & Diagnostic Platform

Course: **21CSE746T — AI Integrated Full Stack Application Development with DevOps**

## Status: Complete — all four phases, all five syllabus units

MediSense AI is a full-stack platform that ingests patient vitals (manually or via
bulk Excel/CSV import), stores them in MongoDB, scores each patient's risk using
both rule-based clinical thresholds and a trained machine-learning classifier,
classifies diagnostic scan images with a CNN, and exposes all of it through a
REST API and a responsive web dashboard — packaged for containerized deployment
with an automated CI/CD pipeline.

---

## Project structure

```
medisense-ai/
├── backend/
│   ├── app/
│   │   ├── models.py            # Patient, Doctor, Appointment, MedicalRecord (ABC)
│   │   │                        #   -> VitalsRecord, LabRecord, ImagingRecord
│   │   ├── exceptions.py         # Custom exception hierarchy
│   │   ├── storage.py            # CSV + pickle file persistence, context managers
│   │   ├── concurrency.py         # Thread-safe appointment booking (Lock)
│   │   ├── data_import.py         # Bulk Excel/CSV import + pandas cleaning + validation
│   │   ├── scraper.py             # BeautifulSoup scraper for reference vitals/lab ranges
│   │   ├── mongo_repository.py    # pymongo CRUD + aggregation (mongomock fallback)
│   │   ├── ml_pipeline.py          # Pandas/NumPy features, RandomForest, K-Means, PCA
│   │   ├── dl_pipeline.py           # Keras CNN: MNIST exercise + scan classifier
│   │   ├── model_loader.py          # Loads persisted models at API startup
│   │   ├── api.py                    # Flask REST API — all endpoints
│   │   └── utils/mathutils.py        # Example user-defined module, __name__ guard
│   ├── tests/                         # pytest suite (18 tests, all passing)
│   ├── models/                         # trained models saved here (gitignored)
│   ├── data/                            # sample workbook + generated output
│   ├── demo_foundation.py               # Runnable demo: Units 1-2
│   ├── demo_phase2.py                    # Runnable demo: Unit 3
│   ├── demo_phase3.py                     # Runnable demo: Unit 4
│   ├── train_and_save_models.py            # Unit 5: train + persist models to disk
│   ├── Dockerfile
│   ├── .dockerignore
│   └── requirements.txt
├── frontend/
│   ├── index.html / styles.css / app.js    # Responsive dashboard (Fetch API, DOM)
│   ├── Dockerfile                            # nginx-served static build
│   └── nginx.conf
├── .github/workflows/ci-cd.yml                # Unit 5: test → build → push pipeline
├── docker-compose.yml                          # backend + frontend + MongoDB
└── .gitignore
```

---

## How each syllabus unit maps to this project (CLO/CO coverage)

| Unit | Course outcome | Where it lives |
|---|---|---|
| **1** | OOP: classes, inheritance, polymorphism, abstraction, modules | `app/models.py` (`MedicalRecord` ABC and subclasses), `app/utils/mathutils.py` |
| **2** | Exception handling, file I/O, concurrency | `app/exceptions.py`, `app/storage.py`, `app/concurrency.py` |
| **3** | Data automation, web scraping, frontend, REST principles | `app/data_import.py`, `app/scraper.py`, `frontend/`, `app/api.py` |
| **4** | MongoDB, NumPy/Pandas, ML (classification/clustering/PCA), DL (CNN) | `app/mongo_repository.py`, `app/ml_pipeline.py`, `app/dl_pipeline.py` |
| **5** | Model deployment, Docker, CI/CD | `train_and_save_models.py`, `app/model_loader.py`, `Dockerfile`s, `docker-compose.yml`, `.github/workflows/ci-cd.yml` |

---

## Running it locally (no Docker)

**1. Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```

**2. Run the unit demos** (no server needed)
```bash
python demo_foundation.py   # Units 1-2
python demo_phase2.py       # Unit 3
python demo_phase3.py       # Unit 4
```

**3. Train and persist the models** (Unit 5 — do this once before starting the API in a fresh environment)
```bash
python train_and_save_models.py
# or, against your own data:
python train_and_save_models.py path/to/your_patients.xlsx
```
This writes `risk_classifier.joblib`, `risk_scaler.joblib`, and `scan_classifier.keras`
into `backend/models/`, which the API loads automatically at startup.

**4. Run the test suite**
```bash
pytest tests/ -v
```

**5. Start the API**
```bash
python -m flask --app app.api run --port 5000
# or, to skip trying a real MongoDB connection and use an in-memory mock directly:
MEDISENSE_FORCE_MOCK_DB=1 python -m flask --app app.api run --port 5000
```

**6. Serve the dashboard** (second terminal)
```bash
cd frontend
python -m http.server 8080
```
Open `http://127.0.0.1:8080`.

---

## Running it with Docker (Unit 5)

```bash
docker compose up --build
```
This builds the backend (training + persisting models at image build time),
the frontend (served via nginx), and starts a real MongoDB container — then
wires them together:

- Dashboard: `http://localhost:8080`
- API: `http://localhost:5000/api`
- MongoDB: `localhost:27017`

The backend `Dockerfile` includes a `HEALTHCHECK` calling `/api/health`, and
`docker-compose.yml` makes the backend wait for MongoDB to report healthy
before starting.

---

## CI/CD pipeline (Unit 5)

`.github/workflows/ci-cd.yml` runs on every push/PR to `main`:

1. **test** — installs dependencies and runs the full `pytest` suite against
   an in-memory (`mongomock`) database, so no external services are needed in CI.
2. **build-and-push** — (only on a push to `main`, and only after tests pass)
   builds both Docker images and pushes them to Docker Hub, tagged with both
   `latest` and the commit SHA.

To use the push step in your own fork/repo, add two repository secrets:
`DOCKERHUB_USERNAME` → your Docker Hub username, and `DOCKERHUB_TOKEN` → a
Docker Hub access token.

---

## REST API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Liveness + which DB backend is active |
| GET | `/api/stats` | MongoDB aggregation: gender counts, average age, high-BP patients |
| GET | `/api/patients` | List all patients |
| GET | `/api/patients/<id>` | Get one patient |
| POST | `/api/patients` | Create a patient (optionally with initial vitals) |
| DELETE | `/api/patients/<id>` | Delete a patient |
| POST | `/api/patients/import` | Bulk import from an uploaded Excel/CSV file |
| GET | `/api/reference-ranges` | Scraped normal-range reference table |
| POST | `/api/ml/train` | Train the risk classifier in-memory on current patients (dev/demo path) |
| POST | `/api/ml/predict` | Predict risk level — uses the **persisted** model if available, else the in-memory one |
| GET | `/api/ml/clusters` | K-Means cluster assignment per patient |
| GET | `/api/ml/pca` | 2D PCA projection per patient |
| GET | `/api/ml/model-status` | Whether persisted risk/scan models are loaded |
| POST | `/api/ml/predict/scan` | Predict NORMAL/ABNORMAL from a 32×32 pixel grid |

---

## Known limitations (stated honestly)

- The CNN scan classifier is trained on **synthetically generated** scan-like
  images (a bright lesion-like spot injected into noise textures), not real
  medical imaging data, since no licensed dataset was available. The
  architecture and training/serving pipeline are real; swap
  `generate_synthetic_scan_dataset()` in `dl_pipeline.py` for a real image
  loader (e.g. against the Chest X-Ray Pneumonia dataset) to use genuine data.
- MongoDB falls back to `mongomock` (in-memory) automatically if no real
  `mongod` is reachable — this keeps the project runnable without installing
  MongoDB, but production use should point `MONGODB_URI` at a real instance
  (the `docker-compose.yml` setup already does this).
- Docker builds were validated for syntax/logic in this environment (no Docker
  daemon was available in the environment where this project was built), but
  every underlying Python component they package was tested directly via
  `pytest` and a live Flask server.

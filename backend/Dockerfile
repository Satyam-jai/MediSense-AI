# MediSense AI — backend image (Unit 5: DevOps / containerization)
FROM python:3.11-slim

WORKDIR /app

# System deps for pandas/openpyxl/tensorflow wheels build faster with these present
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Train + persist models at image build time so the container starts ready
# to serve predictions immediately (no cold-start training on first request).
# Skips cleanly if training fails for any reason (e.g. no sample data) —
# the API still starts and reports models as "not persisted" via /api/ml/model-status.
RUN python train_and_save_models.py || echo "Model training skipped at build time"

ENV MEDISENSE_FORCE_MOCK_DB=0
ENV MONGODB_URI=mongodb://mongo:27017
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app.api:app"]

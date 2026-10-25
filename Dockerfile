
# Stage 1: Install production dependencies
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY requirements-prod.txt .

RUN python -m pip install \
    --no-cache-dir \
    --no-compile \
    --prefix=/install \
    -r requirements-prod.txt

# Remove unnecessary package tests and compiled bytecode
RUN find /install -type d \( -name tests -o -name test -o -name __pycache__ \) \
    -prune -exec rm -rf {} + \
    && find /install -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

# Stage 2: Runtime
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ ./app/
COPY artifacts/churn_model.joblib ./artifacts/churn_model.joblib
COPY artifacts/model_metadata.json ./artifacts/model_metadata.json
COPY artifacts/drift_reference.json ./artifacts/drift_reference.json

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

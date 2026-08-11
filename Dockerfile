# ---------------------------------------------------------------------------
# ATM Cash Forecasting & RAG Analyst  --  container image
# Multi-stage build keeps the runtime image lean.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: prophet/pystan & numpy wheels build cleanly with these present.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source and install the package.
COPY pyproject.toml ./
COPY config ./config
COPY src ./src
COPY scripts ./scripts
RUN pip install -e .

# Generate the bundled synthetic dataset at build time so the image is usable
# out of the box (idempotent; overwritten if you mount your own data).
RUN python -m atmforecast.cli generate-data --out data/raw/atm_transactions.csv

EXPOSE 8000

# Default: launch the streaming API. Override CMD to run the CLI / pipeline.
CMD ["uvicorn", "atmforecast.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

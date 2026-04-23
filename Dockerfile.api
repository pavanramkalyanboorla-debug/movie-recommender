# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies and force CPU torch
RUN uv sync --frozen --no-dev --no-cache \
    && uv run pip uninstall -y torch \
    && uv run pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY api/ ./api/
COPY artifacts/ ./artifacts/

ENV ARTIFACTS_DIR=/app/artifacts
ENV PYTHONPATH=/app

EXPOSE 7860

CMD uv run uvicorn api.main:app --host 0.0.0.0 --port 7860
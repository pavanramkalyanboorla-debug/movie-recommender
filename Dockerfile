# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (including huggingface_hub)
RUN uv sync --frozen --no-dev --no-cache \
    && uv run pip uninstall -y torch \
    && uv run pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy application code and download script
COPY api/ ./api/
COPY download_artifacts.py ./

# Create artifacts directory
RUN mkdir -p /app/artifacts

ENV ARTIFACTS_DIR=/app/artifacts
ENV PYTHONPATH=/app
ENV HF_HUB_ENABLE_HF_TRANSFER=1
EXPOSE 7860

# Download artifacts then start API
CMD uv run python download_artifacts.py && uv run uvicorn api.main:app --host 0.0.0.0 --port 7860
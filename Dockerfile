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
COPY start.sh ./

# Create artifacts directory and make start script executable
RUN mkdir -p /app/artifacts && chmod +x start.sh

ENV ARTIFACTS_DIR=/app/artifacts
ENV PYTHONPATH=/app
ENV HF_HUB_ENABLE_HF_TRANSFER=1
EXPOSE 7860

CMD ["./start.sh"]
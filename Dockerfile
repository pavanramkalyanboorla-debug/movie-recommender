# syntax=docker/dockerfile:1

# ------------------------------------------------------------
# Stage 1: Build dependencies in a clean environment
# ------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv (fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python packages into a virtual environment.
# The lock file already ensures CPU-only torch.
RUN uv sync --frozen --no-dev --no-cache

# ------------------------------------------------------------
# Stage 2: Minimal production image
# ------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install only the bare minimum to run Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code and artifacts
COPY recommender.py streamlit_app.py ./
COPY artifacts/ ./artifacts/

# Set environment variables
ENV ARTIFACTS_DIR=/app/artifacts
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock ./

# Install all dependencies (may include GPU torch transiently)
RUN uv sync --frozen --no-dev --no-cache

# Replace torch with CPU-only version
RUN uv run pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
RUN uv run pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Strip leftover NVIDIA packages
RUN uv run pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
    nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 \
    nvidia-nvtx-cu12 2>/dev/null || true


FROM python:3.11-slim AS runtime

WORKDIR /app

# Minimal system dependency for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the clean virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code and artifacts
COPY recommender.py streamlit_app.py ./
COPY artifacts/ ./artifacts/

ENV ARTIFACTS_DIR=/app/artifacts
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
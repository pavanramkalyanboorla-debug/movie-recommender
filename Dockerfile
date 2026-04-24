# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (CPU torch)
RUN uv sync --frozen --no-dev --no-cache \
    && uv run pip uninstall -y torch \
    && uv run pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy application code and artifacts
COPY recommender.py streamlit_app.py ./
COPY artifacts/ ./artifacts/

ENV ARTIFACTS_DIR=/app/artifacts
ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
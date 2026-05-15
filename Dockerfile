# syntax=docker/dockerfile:1

# =========================================================
# STAGE 1 — UV INSTALLER
# =========================================================
FROM ghcr.io/astral-sh/uv:latest AS uv-stage

# =========================================================
# STAGE 2 — DEPENDENCIES
# =========================================================
FROM python:3.11-slim AS deps

WORKDIR /app

# ─────────────────────────────────────────────
# SYSTEM DEPENDENCIES
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# COPY PROJECT FILES
# ─────────────────────────────────────────────
COPY pyproject.toml uv.lock README.md ./

# ─────────────────────────────────────────────
# INSTALL PYTHON DEPENDENCIES
# ─────────────────────────────────────────────
RUN --mount=from=uv-stage,source=/uv,target=/bin/uv \
    uv sync --no-dev --frozen

# ─────────────────────────────────────────────
# PRE-DOWNLOAD EMBEDDING MODEL
# Prevent cold-start delays on HuggingFace Spaces
# ─────────────────────────────────────────────
RUN .venv/bin/python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2') \
"

# =========================================================
# STAGE 3 — FINAL RUNTIME
# =========================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# ─────────────────────────────────────────────
# RUNTIME SYSTEM PACKAGES
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Better HuggingFace caching
ENV HF_HOME="/root/.cache/huggingface"

# =========================================================
# COPY VIRTUAL ENV + CACHE
# =========================================================
COPY --from=deps /app/.venv /app/.venv
COPY --from=deps /root/.cache /root/.cache

# =========================================================
# COPY APPLICATION FILES
# =========================================================
COPY artifacts/ ./artifacts/
COPY src/ ./src/
COPY app/ ./app/


# =========================================================
# PORTS
# =========================================================
# Streamlit frontend
EXPOSE 7860

# FastAPI backend
EXPOSE 8000

# =========================================================
# HEALTHCHECK
# =========================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
CMD curl -f http://localhost:8000/health || exit 1

# =========================================================
# START SERVICES
# =========================================================
CMD ["sh", "-c", "\
echo '🚀 Starting MovieMind Backend...' && \
uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
\
echo '⏳ Waiting for FastAPI health check...' && \
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do \
    sleep 2; \
done && \
\
echo '✅ Backend ready' && \
echo '🎬 Starting Streamlit frontend...' && \
\
streamlit run app/streamlit_app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
"]
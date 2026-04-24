# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

COPY recommender.py streamlit_app.py ./
COPY artifacts/ ./artifacts/

ENV ARTIFACTS_DIR=/app/artifacts
ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
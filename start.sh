#!/bin/bash
set -e

echo "Downloading artifacts from Hugging Face bucket..."
hf buckets sync hf://buckets/PavanBoorla/movie-mind-artifacts /app/artifacts

echo "Starting API..."
uv run uvicorn api.main:app --host 0.0.0.0 --port 7860
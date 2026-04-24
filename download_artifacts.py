
"""Download artifacts from Hugging Face Bucket at startup."""
import os
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET_REPO = "PavanBoorla/movie-mind-artifacts"
LOCAL_DIR = os.environ.get("ARTIFACTS_DIR", "/app/artifacts")

logger.info(f"Downloading artifacts from bucket {BUCKET_REPO} to {LOCAL_DIR}...")
snapshot_download(
    repo_id=BUCKET_REPO,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    repo_type="dataset",   # buckets are treated as datasets by snapshot_download
    token=os.environ.get("HF_TOKEN")  # will use token if set (e.g., from Space secret)
)
logger.info("Artifacts downloaded successfully.")
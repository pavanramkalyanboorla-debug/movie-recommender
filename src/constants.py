import os

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", os.path.join("data"))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join("artifacts"))

# ------------------------------------------------------------
# Raw data file – IMDB & TMDB Movie Metadata Big Dataset (over 1M)
# ------------------------------------------------------------
TMDB_CSV = os.path.join(DATA_DIR, "movies_metadata.csv")

# ------------------------------------------------------------
# Built artifacts
# ------------------------------------------------------------
PROCESSED_PARQUET = os.path.join(ARTIFACTS_DIR, "movies_processed_final.parquet")
FAISS_INDEX       = os.path.join(ARTIFACTS_DIR, "movies_faiss.index")
TFIDF_VECTORIZER  = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
MODEL_NAME_FILE   = os.path.join(ARTIFACTS_DIR, "model_name.txt")

# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ------------------------------------------------------------
# Columns we keep right after loading (raw dataset column names)
# ------------------------------------------------------------
KEEP_COLS = [
    "id", "title", "overview", "release_date",
    "popularity", "vote_average", "vote_count",
    "keywords", "genres_list", "Cast_list", "Director",
    "Star1", "Star2", "Star3", "Star4"
]

# ------------------------------------------------------------
# Final columns stored in the processed parquet
# ------------------------------------------------------------
FINAL_COLS = [
    "movie_id", "title", "year", "genres", "overview",
    "soup", "norm_popularity", "vote_average", "vote_count",
    "avg_rating", "popularity_log", "director", "cast", "keywords"
]
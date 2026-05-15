# src/components/index_builder.py
import os
import sys
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.constants import (
    ARTIFACTS_DIR, PROCESSED_PARQUET, FAISS_INDEX,
    TFIDF_VECTORIZER, MODEL_NAME_FILE, EMBEDDING_MODEL
)


class IndexBuilder:
    """
    Builds the SBERT‑based FAISS index and a TF‑IDF vectorizer from
    the processed parquet. All artifacts are saved to ARTIFACTS_DIR.
    """
    def __init__(self):
        self.model = None

    def run(self, df):

        try:
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)

            # 1. SBERT embeddings
            logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.model = SentenceTransformer(EMBEDDING_MODEL)

            soups = df["soup"].tolist()
            logging.info(f"Encoding {len(soups)} sentences…")
            embeddings = self.model.encode(soups, show_progress_bar=True,
                                           convert_to_numpy=True, batch_size=64)

            # 2. FAISS (exact inner product)
            embeddings_f32 = np.ascontiguousarray(embeddings.astype('float32'))
            dim = embeddings_f32.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings_f32)
            index.add(embeddings_f32)
            faiss.write_index(index, FAISS_INDEX)
            logging.info("FAISS index saved")

            # 3. TF‑IDF vectorizer
            tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf.fit_transform(df["soup"])
            with open(TFIDF_VECTORIZER, "wb") as f:
                pickle.dump(tfidf, f)
            logging.info("TF‑IDF vectorizer saved")

            # 4. Model name
            with open(MODEL_NAME_FILE, "w") as f:
                f.write(EMBEDDING_MODEL)

            logging.info("All artifacts built successfully.")

        except Exception as e:
            raise CustomException(e, sys)
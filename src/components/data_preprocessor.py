# src/components/data_preprocessor.py
import re
import ast
import sys
import numpy as np
import pandas as pd
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.constants import FINAL_COLS, TMDB_CSV


class DataPreprocessor:
    """
    Loads the raw TMDB CSV (Alan Vourch dataset), cleans every column,
    extracts director, cast, genres, keywords, and builds the semantic soup.
    """

    def __init__(self, csv_path: str = TMDB_CSV):
        self.csv_path = csv_path

    # ---------- internal helpers ----------
    @staticmethod
    def _clean_title(title):
        if pd.isna(title):
            return ""
        return re.sub(r'\s*\(\d{4}\)\s*$', '', str(title)).strip()

    @staticmethod
    def _normalize_title(title):
        if pd.isna(title):
            return ""
        title = str(title).strip()
        m = re.match(r'^(.*),\s*(The|A|An)$', title, re.IGNORECASE)
        if m:
            return f"{m.group(2)} {m.group(1)}"
        return title

    @staticmethod
    def _clean_director(val):
        if pd.isna(val):
            return ""
        return str(val).strip()

    @staticmethod
    def _clean_cast(val, top_n=5):
        """Cast is a comma‑separated string."""
        if pd.isna(val) or not val:
            return ""
        names = [name.strip() for name in str(val).split(",")]
        return " ".join(names[:top_n])

    @staticmethod
    def _clean_genres(val):
        """Handle JSON list or plain text."""
        if pd.isna(val) or not val:
            return ""
        val_str = str(val).strip()
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                items = ast.literal_eval(val_str)
                names = [i.get("name", str(i)) if isinstance(i, dict) else str(i) for i in items]
                return " ".join(names)
            except (ValueError, SyntaxError):
                pass
        return val_str

    @staticmethod
    def _clean_keywords(val):
        """JSON or comma‑separated keywords."""
        if pd.isna(val) or not val:
            return ""
        val_str = str(val).strip()
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                items = ast.literal_eval(val_str)
                names = [i.get("name", str(i)) if isinstance(i, dict) else str(i) for i in items]
                return " ".join(names)
            except (ValueError, SyntaxError):
                pass
        return val_str

    # ---------- main entry point ----------
    def run(self) -> pd.DataFrame:
        try:
            logging.info("Loading raw CSV…")
            df = pd.read_csv(self.csv_path, low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]
            logging.info(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

            # Rename id → movie_id
            df.rename(columns={"id": "movie_id"}, inplace=True)

            # Ensure required columns exist
            required = ["title", "overview", "release_date", "popularity",
                        "vote_average", "vote_count", "genres", "keywords",
                        "cast", "director"]
            for col in required:
                if col not in df.columns:
                    df[col] = ""

            # Title
            df["title"] = df["title"].apply(self._clean_title).apply(self._normalize_title)

            # Year
            if "release_date" in df.columns:
                df["year"] = pd.to_datetime(df["release_date"], errors='coerce').dt.year
            else:
                df["year"] = 2000
            df["year"] = df["year"].fillna(2000).astype(int)

            # Text / numbers
            df["overview"]      = df["overview"].fillna("")
            df["keywords"]      = df["keywords"].apply(self._clean_keywords)
            df["vote_average"]  = pd.to_numeric(df["vote_average"], errors='coerce').fillna(0)
            df["vote_count"]    = pd.to_numeric(df["vote_count"], errors='coerce').fillna(0)
            df["popularity"]    = pd.to_numeric(df["popularity"], errors='coerce').fillna(0)

            # Director (plain string)
            df["director"] = df["director"].apply(self._clean_director)

            # Cast (comma‑separated → top‑5)
            df["cast"] = df["cast"].apply(self._clean_cast, top_n=5)

            # Genres / keywords (JSON‑aware)
            df["genres"]   = df["genres"].apply(self._clean_genres)
            df["keywords"] = df["keywords"].apply(self._clean_keywords)

            # Deduplicate
            df = df.sort_values(by="vote_count", ascending=False)
            df = df.drop_duplicates(subset=["title", "year"], keep="first")

            # Semantic soup
            df["soup"] = (
                df["director"].fillna("").str.lower() + " " +
                df["title"].fillna("").str.lower() + " " +
                df["cast"].fillna("").str.lower() + " " +
                df["genres"].fillna("").str.lower() + " " +
                df["overview"].fillna("").str.lower() + " " +
                df["keywords"].fillna("").str.lower()
            )

            df["avg_rating"] = df["vote_average"]
            df["popularity_log"] = np.log1p(df["vote_count"])
            max_log = df["popularity_log"].max() + 1e-8
            min_log = df["popularity_log"].min()
            df["norm_popularity"] = (df["popularity_log"] - min_log) / (max_log - min_log)

            # Keep only FINAL_COLS
            available = [c for c in FINAL_COLS if c in df.columns]
            df_final = df[available].copy()

            logging.info(f"Preprocessed → {df_final.shape}")
            return df_final

        except Exception as e:
            raise CustomException(e, sys)
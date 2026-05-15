# src/components/data_loader.py
import sys
import pandas as pd
from dataclasses import dataclass
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.constants import DATA_DIR, TMDB_CSV


@dataclass
class DataLoaderConfig:
    data_dir: str = DATA_DIR
    csv_filename: str = TMDB_CSV


class DataLoader:
    """
    Loads the raw TMDB CSV (Alan Vourch dataset – 1M+ movies).
    Performs minimal cleaning: lowercases column names for consistency.
    """

    def __init__(self, config: DataLoaderConfig = None):
        self.config = config or DataLoaderConfig()

    def load(self) -> pd.DataFrame:
        try:
            logging.info(f"Loading dataset from {self.config.csv_filename}…")
            df = pd.read_csv(self.config.csv_filename, low_memory=False)

            # Normalise column names
            df.columns = [c.strip().lower() for c in df.columns]

            logging.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            raise CustomException(e, sys)
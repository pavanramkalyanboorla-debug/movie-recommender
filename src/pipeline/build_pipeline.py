# src/pipeline/build_pipeline.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.components.data_preprocessor import DataPreprocessor
from src.components.index_builder import IndexBuilder
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.constants import PROCESSED_PARQUET

if __name__ == "__main__":
    try:
        # 1. Preprocess raw CSV → clean parquet
        preprocessor = DataPreprocessor()
        df_final = preprocessor.run()
        os.makedirs(os.path.dirname(PROCESSED_PARQUET), exist_ok=True)
        df_final.to_parquet(PROCESSED_PARQUET, index=False)
        logging.info(f"Processed parquet saved to {PROCESSED_PARQUET}")

        # 2. Build FAISS + TF‑IDF
        builder = IndexBuilder()
        builder.run(df_final)

        logging.info("MovieMind build pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Build pipeline failed: {e}")
        raise CustomException(e, sys)
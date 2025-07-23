from pathlib import Path
import pandas as pd
from .dataset_utils import unify_column_names, auto_cast
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

DATASETS = {
    "adult":      "autism_screening_adult.csv",
    "adolescent": "autism_screening_adolescent.csv",
    "child":      "autism_screening_child.csv",
}

class ASDDataLoader:
    """
    Loads raw ASD CSV files from data/raw, normalises column names,
    basic type casting, and returns pandas DataFrames.
    """

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)

    def load(self, dataset_key: str) -> pd.DataFrame:
        if dataset_key not in DATASETS:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
        file_path = self.raw_dir / DATASETS[dataset_key]
        logger.info(f"Loading {file_path}")
        df = pd.read_csv(file_path)
        df = unify_column_names(df)
        df = auto_cast(df)
        logger.info(f"{dataset_key}: {df.shape[0]} rows, {df.shape[1]} cols")
        return df

    def load_all(self) -> dict:
        return {k: self.load(k) for k in DATASETS}

from pathlib import Path
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils.logging_utils import get_logger
from .data_loader import ASDDataLoader

logger = get_logger(__name__)

class ASDDataCleaner:
    """
    Cleans each dataset:
      • imputes missing values
      • encodes categorical features
      • balances classes with SMOTE
      • writes cleaned CSVs to data/processed
    """

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.loader = ASDDataLoader()

    def _build_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        # Identify categorical and numerical columns
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if X[c].dtype != "object"]

        # Define pipelines
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        return ColumnTransformer(
            transformers=[
                ("categorical", cat_pipe, cat_cols),
                ("numerical", num_pipe, num_cols),
            ],
            remainder="drop",
            sparse_threshold=0
        )

    def clean_one(self, key: str) -> pd.DataFrame:
        # Load raw data
        df = self.loader.load(key)
        logger.info(f"{key}: loaded raw data with shape {df.shape}")

        # === Identify the correct target column ===
        # Check normalized column names (all lower-case, slash retained)
        if "class/asd" in df.columns:
            target = "class/asd"
        elif "class_asd_traits" in df.columns:
            target = "class_asd_traits"
        elif "class" in df.columns:
            target = "class"
        else:
            raise ValueError(f"No known target column in '{key}': {df.columns.tolist()}")

        # === Split into features and label ===
        X = df.drop(columns=[target])
        try:
            y = df[target].astype(int)
        except ValueError:
            # Fallback for 'yes'/'no' strings
            y = df[target].str.lower().map({"yes": 1, "no": 0, "y": 1, "n": 0}).astype(int)

        # === Preprocessing pipeline ===
        preprocessor = self._build_pipeline(X)
        logger.info(f"{key}: fitting preprocessing pipeline")
        X_processed = preprocessor.fit_transform(X)

        # === SMOTE balancing ===
        logger.info(f"{key}: applying SMOTE balancing")
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X_processed, y)

        # === Save processed DataFrame ===
        processed_df = pd.DataFrame(
            X_bal,
            columns=[f"f{i}" for i in range(X_bal.shape[1])]
        )
        processed_df["target"] = y_bal

        out_path = self.processed_dir / f"{key}_processed.csv"
        processed_df.to_csv(out_path, index=False)
        logger.info(f"{key}: cleaned data written to {out_path}")

        return processed_df

    def clean_all(self) -> dict:
        results = {}
        for dataset_key in ["adult", "adolescent", "child"]:
            results[dataset_key] = self.clean_one(dataset_key)
        return results

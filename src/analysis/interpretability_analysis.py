import shap
import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing.processed_loader import ProcessedDataLoader
from src.preprocessing.feature_harmonizer import FeatureHarmonizer
from src.models.model_factory import ModelFactory
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class InterpretabilityAnalysis:
    """Compute SHAP values for models on each harmonized dataset"""

    def __init__(self):
        self.loader = ProcessedDataLoader()
        self.harmonizer = FeatureHarmonizer()
        self.datasets = ['adult', 'adolescent', 'child']

        # Load raw splits and harmonize
        raw = self.loader.load_all_datasets()
        self.harmonized = self.harmonizer.harmonize_datasets(raw)
        logger.info("Loaded and harmonized data for interpretability")

    def compute_shap(self, dataset_name: str, model_name: str = 'tabnet'):
        """Train fresh model and compute SHAP values on test set"""
        # Get harmonized test data
        X, y = self.harmonized[dataset_name]
        # Split into train/test 80/20
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Create and train fresh model
        model = ModelFactory.create_model(model_name)
        if model_name == 'tabnet':
            # For TabNet use validation
            val_split = int(0.8 * len(X_train))
            model.fit(
                X_train[:val_split], y_train[:val_split],
                X_train[val_split:], y_train[val_split:]
            )
        else:
            model.fit(X_train, y_train)

        logger.info(f"Trained fresh {model_name} on {dataset_name}")

        # Compute SHAP on a subset for speed
        background = X_train[:100]
        sample = X_test[:100]
        explainer = shap.Explainer(model.predict_proba, background)
        shap_vals = explainer(sample)

        # Extract positive-class SHAP values
        if shap_vals.values.ndim == 3 and shap_vals.values.shape[2] > 1:
            vals = shap_vals.values[:, :, 1]
        else:
            vals = shap_vals.values

        # Save SHAP values
        df = pd.DataFrame(vals, columns=[f"feature_{i}" for i in range(vals.shape[1])])
        out_dir = Path("results/figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"shap_{model_name}_{dataset_name}.csv"
        df.to_csv(out_file, index=False)
        logger.info(f"SHAP values saved to {out_file}")

        # Save feature importance (mean absolute)
        imp = pd.Series(np.abs(vals).mean(axis=0),
                        index=df.columns).sort_values(ascending=False)
        imp.to_csv(out_dir / f"shap_importance_{model_name}_{dataset_name}.csv",
                   header=['importance'])
        logger.info(f"Feature importance saved to {out_dir / f'shap_importance_{model_name}_{dataset_name}.csv'}")

        return df, imp

    def compute_all(self):
        """Compute SHAP and importance for all models/datasets"""
        results = {}
        for model in ['tabnet','random_forest','xgboost','svm']:
            for ds in self.datasets:
                try:
                    df, imp = self.compute_shap(ds, model)
                    results[f"{model}_{ds}"] = (df, imp)
                except Exception as e:
                    logger.error(f"Failed SHAP for {model} on {ds}: {e}")
        return results

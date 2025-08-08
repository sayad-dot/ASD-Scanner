import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.utils.data_utils import DataUtils
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class PerformanceAnalysis:
    """Generate summary tables and metrics comparisons"""

    def __init__(self, results_file: str = "experiments/cross_validation/all_cross_validation_results.json"):
        self.results_file = Path(results_file)
        if not self.results_file.exists():
            raise FileNotFoundError(f"{self.results_file} not found")
        self.results = json.loads(self.results_file.read_text())
        self.df = DataUtils.create_results_dataframe(self.results)

    def summary_by_model(self) -> pd.DataFrame:
        """Compute mean and std of AUC per model"""
        df = self.df
        table = df.groupby('model_name')['auc'].agg(['mean','std','min','max','count']).reset_index()
        table.columns = ['Model','Mean AUC','Std AUC','Min AUC','Max AUC','Count']
        return table.round(4)

    def summary_by_dataset_pair(self) -> pd.DataFrame:
        """Pivot AUC by train/test dataset"""
        table = self.df.pivot_table(
            index='train_dataset', columns='test_dataset', values='auc', aggfunc='mean'
        )
        return table.round(4)

    def save_tables(self, out_dir: str = "results/figures"):
        """Save summary tables as CSV"""
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        self.summary_by_model().to_csv(Path(out_dir)/"model_performance_summary.csv", index=False)
        self.summary_by_dataset_pair().to_csv(Path(out_dir)/"dataset_pair_auc_matrix.csv")
        logger.info("Performance summary tables saved")

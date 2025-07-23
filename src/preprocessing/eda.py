from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .data_loader import ASDDataLoader
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
sns.set(style="whitegrid")

class ASDEDA:
    """
    Generates:
      • basic stats summary (.json)
      • correlation heatmaps
      • class distribution bar plots
    """

    def __init__(self, figs_dir: str = "results/figures", metrics_dir: str = "results/metrics"):
        self.loader = ASDDataLoader()
        self.figs_dir = Path(figs_dir);   self.figs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = Path(metrics_dir); self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def _save_stats(self, df: pd.DataFrame, key: str):
        stats = df.describe(include="all").to_dict()
        path = self.metrics_dir / f"{key}_stats.json"
        pd.Series(stats).to_json(path)
        logger.info(f"{key}: stats saved to {path}")

    def _plot_corr(self, df: pd.DataFrame, key: str):
        num_df = df.select_dtypes(include=["number"])
        corr = num_df.corr(method="pearson")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title(f"{key.capitalize()} – Numeric Feature Correlations")
        fig_path = self.figs_dir / f"{key}_corr.png"
        plt.tight_layout(); plt.savefig(fig_path); plt.close()
        logger.info(f"{key}: correlation heatmap saved to {fig_path}")

    def _plot_class_balance(self, df: pd.DataFrame, key: str):
        target = "class_asd_traits" if "class_asd_traits" in df.columns else "class"
        plt.figure(figsize=(4, 3))
        sns.countplot(x=target, data=df)
        plt.title(f"{key.capitalize()} – Class Balance")
        fig_path = self.figs_dir / f"{key}_class_balance.png"
        plt.tight_layout(); plt.savefig(fig_path); plt.close()
        logger.info(f"{key}: class balance plot saved to {fig_path}")

    def analyse_one(self, key: str):
        df = self.loader.load(key)
        self._save_stats(df, key)
        self._plot_corr(df, key)
        self._plot_class_balance(df, key)

    def analyse_all(self):
        for key in ["adult", "adolescent", "child"]:
            self.analyse_one(key)

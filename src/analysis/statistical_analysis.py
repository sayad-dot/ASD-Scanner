import json
from pathlib import Path
from src.evaluation.statistical_tests import StatisticalTester
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class StatisticalAnalysis:
    """Run and save statistical significance tests"""

    def __init__(self, cv_results_file: str = "experiments/cross_validation/all_cross_validation_results.json"):
        self.cv_file = Path(cv_results_file)
        if not self.cv_file.exists():
            raise FileNotFoundError(f"{self.cv_file} not found")
        self.tester = StatisticalTester(results_dir="experiments/statistical_results")

    def run_auc_tests(self):
        """Perform pairwise and overall tests for AUC"""
        result = self.tester.compare_all_models(str(self.cv_file), metric='auc')
        return result

    def save(self):
        """Result already saved in tester.compare_all_models"""
        logger.info("Statistical analysis completed and saved")

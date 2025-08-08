import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, friedmanchisquare
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class StatisticalTester:
    """Perform statistical significance tests on model performance"""
    
    def __init__(self, results_dir: str = "experiments/statistical_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.alpha = 0.05  # Significance level
    
    def load_cross_validation_results(self, cv_results_file: str) -> Dict[str, Any]:
        """Load cross-validation results"""
        with open(cv_results_file, 'r') as f:
            return json.load(f)
    
    def extract_metric_values(self, results: Dict[str, Any], metric: str = 'auc') -> Dict[str, List[float]]:
        """Extract metric values for each model across all dataset combinations"""
        
        model_metrics = {
            'tabnet': [],
            'random_forest': [],
            'xgboost': [],
            'svm': []
        }
        
        for key, result in results.items():
            model_name = result['model_name']
            if model_name in model_metrics:
                model_metrics[model_name].append(result['metrics'][metric])
        
        return model_metrics
    
    def pairwise_comparison(self, values1: List[float], values2: List[float], 
                          model1: str, model2: str, metric: str = 'auc') -> Dict[str, Any]:
        """Perform pairwise statistical comparison between two models"""
        
        values1 = np.array(values1)
        values2 = np.array(values2)
        
        # Ensure equal length
        min_len = min(len(values1), len(values2))
        values1 = values1[:min_len]
        values2 = values2[:min_len]
        
        if len(values1) < 3:
            logger.warning(f"Insufficient data for statistical test: {len(values1)} samples")
            return {
                'model1': model1,
                'model2': model2,
                'metric': metric,
                'test': 'insufficient_data',
                'p_value': np.nan,
                'significant': False,
                'effect_size': np.nan
            }
        
        # Wilcoxon signed-rank test (paired samples)
        try:
            statistic, p_value = wilcoxon(values1, values2, alternative='two-sided')
            test_name = 'wilcoxon'
        except:
            # Fall back to Mann-Whitney U test (independent samples)
            try:
                statistic, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
                test_name = 'mannwhitney'
            except Exception as e:
                logger.error(f"Statistical test failed for {model1} vs {model2}: {e}")
                return {
                    'model1': model1,
                    'model2': model2,
                    'metric': metric,
                    'test': 'failed',
                    'p_value': np.nan,
                    'significant': False,
                    'effect_size': np.nan
                }
        
        # Effect size (Cohen's d approximation)
        try:
            pooled_std = np.sqrt((np.var(values1, ddof=1) + np.var(values2, ddof=1)) / 2)
            effect_size = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
        except:
            effect_size = 0
        
        return {
            'model1': model1,
            'model2': model2,
            'metric': metric,
            'test': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'effect_size': float(effect_size),
            'mean_diff': float(np.mean(values1) - np.mean(values2)),
            'model1_mean': float(np.mean(values1)),
            'model2_mean': float(np.mean(values2)),
            'model1_std': float(np.std(values1)),
            'model2_std': float(np.std(values2))
        }
    
    def compare_all_models(self, cv_results_file: str, metric: str = 'auc') -> Dict[str, Any]:
        """Compare all models pairwise using statistical tests"""
        
        logger.info(f"Performing statistical comparisons for {metric}")
        
        # Load results
        results = self.load_cross_validation_results(cv_results_file)
        model_metrics = self.extract_metric_values(results, metric)
        
        models = list(model_metrics.keys())
        comparisons = []
        
        # Pairwise comparisons
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                
                comparison = self.pairwise_comparison(
                    model_metrics[model1],
                    model_metrics[model2],
                    model1,
                    model2,
                    metric
                )
                comparisons.append(comparison)
                
                logger.info(f"{model1} vs {model2}: p={comparison['p_value']:.4f}, "
                          f"significant={comparison['significant']}")
        
        # Overall test (Friedman test for multiple related samples)
        try:
            # Prepare data for Friedman test
            all_values = [model_metrics[model] for model in models]
            min_len = min(len(values) for values in all_values)
            trimmed_values = [values[:min_len] for values in all_values]
            
            if min_len >= 3:
                friedman_stat, friedman_p = friedmanchisquare(*trimmed_values)
                overall_test = {
                    'test': 'friedman',
                    'statistic': float(friedman_stat),
                    'p_value': float(friedman_p),
                    'significant': friedman_p < self.alpha,
                    'interpretation': 'At least one model significantly different' if friedman_p < self.alpha else 'No significant differences detected'
                }
            else:
                overall_test = {'test': 'insufficient_data'}
                
        except Exception as e:
            logger.error(f"Friedman test failed: {e}")
            overall_test = {'test': 'failed', 'error': str(e)}
        
        # Summary statistics
        summary = {}
        for model, values in model_metrics.items():
            summary[model] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'count': len(values)
            }
        
        result = {
            'metric': metric,
            'pairwise_comparisons': comparisons,
            'overall_test': overall_test,
            'summary_statistics': summary,
            'significance_level': self.alpha
        }
        
        # Save results
        output_file = self.results_dir / f"statistical_comparison_{metric}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Statistical analysis saved to {output_file}")
        return result
    
    def generate_significance_matrix(self, comparisons: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate a matrix showing statistical significance between models"""
        
        models = list(set([c['model1'] for c in comparisons] + [c['model2'] for c in comparisons]))
        models.sort()
        
        matrix = pd.DataFrame(index=models, columns=models, dtype=str)
        
        # Fill diagonal with self-comparison
        for model in models:
            matrix.loc[model, model] = "—"
        
        # Fill matrix with p-values and significance indicators
        for comp in comparisons:
            model1, model2 = comp['model1'], comp['model2']
            p_val = comp['p_value']
            significant = comp['significant']
            
            if pd.isna(p_val):
                symbol = "N/A"
            elif significant:
                symbol = f"{p_val:.3f}*"
            else:
                symbol = f"{p_val:.3f}"
            
            matrix.loc[model1, model2] = symbol
            matrix.loc[model2, model1] = symbol
        
        return matrix

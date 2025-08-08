import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ExperimentManager:
    """Manage and organize cross-dataset validation experiments"""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_experiment_summary(self, cv_results_file: str) -> Dict[str, Any]:
        """Create a comprehensive summary of cross-dataset validation results"""
        
        with open(cv_results_file, 'r') as f:
            results = json.load(f)
        
        # Initialize summary structure
        summary = {
            'total_experiments': len(results),
            'models_tested': set(),
            'datasets_used': set(),
            'metrics_summary': {},
            'best_performers': {},
            'cross_dataset_analysis': {},
            'within_dataset_analysis': {}
        }
        
        # Extract basic info
        for key, result in results.items():
            summary['models_tested'].add(result['model_name'])
            summary['datasets_used'].add(result['train_dataset'])
            summary['datasets_used'].add(result['test_dataset'])
        
        summary['models_tested'] = list(summary['models_tested'])
        summary['datasets_used'] = list(summary['datasets_used'])
        
        # Analyze performance by metric
        metrics_to_analyze = ['auc', 'accuracy', 'f1', 'precision', 'recall']
        
        for metric in metrics_to_analyze:
            metric_values = []
            metric_by_model = {model: [] for model in summary['models_tested']}
            
            for result in results.values():
                if metric in result['metrics']:
                    value = result['metrics'][metric]
                    metric_values.append(value)
                    metric_by_model[result['model_name']].append(value)
            
            if metric_values:
                summary['metrics_summary'][metric] = {
                    'overall_mean': np.mean(metric_values),
                    'overall_std': np.std(metric_values),
                    'overall_min': np.min(metric_values),
                    'overall_max': np.max(metric_values),
                    'by_model': {
                        model: {
                            'mean': np.mean(values) if values else 0,
                            'std': np.std(values) if values else 0,
                            'count': len(values)
                        } for model, values in metric_by_model.items()
                    }
                }
        
        # Find best performers
        for metric in metrics_to_analyze:
            best_score = -1
            best_combination = None
            
            for key, result in results.items():
                if metric in result['metrics']:
                    score = result['metrics'][metric]
                    if score > best_score:
                        best_score = score
                        best_combination = {
                            'model': result['model_name'],
                            'train_dataset': result['train_dataset'],
                            'test_dataset': result['test_dataset'],
                            'score': score
                        }
            
            if best_combination:
                summary['best_performers'][metric] = best_combination
        
        # Cross-dataset vs within-dataset analysis
        cross_dataset_results = []
        within_dataset_results = []
        
        for result in results.values():
            if result['train_dataset'] == result['test_dataset']:
                within_dataset_results.append(result)
            else:
                cross_dataset_results.append(result)
        
        # Analyze cross-dataset performance
        if cross_dataset_results:
            for metric in metrics_to_analyze:
                cross_scores = [r['metrics'][metric] for r in cross_dataset_results if metric in r['metrics']]
                if cross_scores:
                    summary['cross_dataset_analysis'][metric] = {
                        'mean': np.mean(cross_scores),
                        'std': np.std(cross_scores),
                        'count': len(cross_scores)
                    }
        
        # Analyze within-dataset performance
        if within_dataset_results:
            for metric in metrics_to_analyze:
                within_scores = [r['metrics'][metric] for r in within_dataset_results if metric in r['metrics']]
                if within_scores:
                    summary['within_dataset_analysis'][metric] = {
                        'mean': np.mean(within_scores),
                        'std': np.std(within_scores),
                        'count': len(within_scores)
                    }
        
        # Save summary
        summary_file = self.base_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Experiment summary saved to {summary_file}")
        return summary
    
    def generate_results_table(self, cv_results_file: str, metric: str = 'auc') -> pd.DataFrame:
        """Generate a results table for easy interpretation"""
        
        with open(cv_results_file, 'r') as f:
            results = json.load(f)
        
        # Create table data
        table_data = []
        
        for key, result in results.items():
            if metric in result['metrics']:
                table_data.append({
                    'Model': result['model_name'].replace('_', ' ').title(),
                    'Train Dataset': result['train_dataset'].title(),
                    'Test Dataset': result['test_dataset'].title(),
                    'Train Samples': result['train_samples'],
                    'Test Samples': result['test_samples'],
                    metric.upper(): result['metrics'][metric],
                    'Cross-Dataset': result['train_dataset'] != result['test_dataset']
                })
        
        df = pd.DataFrame(table_data)
        df = df.round(4)
        
        # Save table
        table_file = self.base_dir / f"results_table_{metric}.csv"
        df.to_csv(table_file, index=False)
        
        logger.info(f"Results table saved to {table_file}")
        return df

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
from scipy import stats
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class PerformanceAnalyzer:
    """Analyze and compare model performance across different scenarios"""
    
    def __init__(self, results_dir: str = "experiments/performance_matrices"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = ['tabnet', 'random_forest', 'xgboost', 'svm']
        self.datasets = ['adult', 'adolescent', 'child']
        self.metrics = ['auc', 'accuracy', 'f1', 'precision', 'recall', 'specificity']
    
    def analyze_generalization_gap(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the generalization gap between within-dataset and cross-dataset performance"""
        
        logger.info("Analyzing generalization gap")
        
        analysis = {
            'by_model': {},
            'by_source_dataset': {},
            'by_target_dataset': {},
            'overall_summary': {}
        }
        
        # Group results by different criteria
        within_dataset = {}
        cross_dataset = {}
        
        for key, result in cv_results.items():
            model = result['model_name']
            train_ds = result['train_dataset']
            test_ds = result['test_dataset']
            metrics = result['metrics']
            
            if train_ds == test_ds:
                # Within-dataset performance
                if model not in within_dataset:
                    within_dataset[model] = {}
                within_dataset[model][train_ds] = metrics
            else:
                # Cross-dataset performance
                if model not in cross_dataset:
                    cross_dataset[model] = {}
                if train_ds not in cross_dataset[model]:
                    cross_dataset[model][train_ds] = {}
                cross_dataset[model][train_ds][test_ds] = metrics
        
        # Calculate generalization gaps by model
        for model in self.models:
            if model in within_dataset and model in cross_dataset:
                model_analysis = {'generalization_gaps': {}}
                
                for metric in self.metrics:
                    within_scores = []
                    cross_scores = []
                    
                    # Collect within-dataset scores
                    for dataset in self.datasets:
                        if dataset in within_dataset[model] and metric in within_dataset[model][dataset]:
                            within_scores.append(within_dataset[model][dataset][metric])
                    
                    # Collect cross-dataset scores
                    for train_ds in cross_dataset[model]:
                        for test_ds in cross_dataset[model][train_ds]:
                            if metric in cross_dataset[model][train_ds][test_ds]:
                                cross_scores.append(cross_dataset[model][train_ds][test_ds][metric])
                    
                    if within_scores and cross_scores:
                        within_mean = np.mean(within_scores)
                        cross_mean = np.mean(cross_scores)
                        gap = within_mean - cross_mean
                        
                        model_analysis['generalization_gaps'][metric] = {
                            'within_dataset_mean': float(within_mean),
                            'cross_dataset_mean': float(cross_mean),
                            'generalization_gap': float(gap),
                            'gap_percentage': float((gap / within_mean) * 100) if within_mean > 0 else 0.0,
                            'within_dataset_std': float(np.std(within_scores)),
                            'cross_dataset_std': float(np.std(cross_scores))
                        }
                
                analysis['by_model'][model] = model_analysis
        
        # Calculate overall summary statistics
        overall_gaps = {metric: [] for metric in self.metrics}
        
        for model_data in analysis['by_model'].values():
            for metric in self.metrics:
                if metric in model_data['generalization_gaps']:
                    overall_gaps[metric].append(model_data['generalization_gaps'][metric]['generalization_gap'])
        
        for metric in self.metrics:
            if overall_gaps[metric]:
                analysis['overall_summary'][metric] = {
                    'mean_gap': float(np.mean(overall_gaps[metric])),
                    'std_gap': float(np.std(overall_gaps[metric])),
                    'min_gap': float(np.min(overall_gaps[metric])),
                    'max_gap': float(np.max(overall_gaps[metric]))
                }
        
        # Find best generalizing model
        auc_gaps = [(model, data['generalization_gaps'].get('auc', {}).get('generalization_gap', float('inf'))) 
                   for model, data in analysis['by_model'].items()]
        auc_gaps = [(model, gap) for model, gap in auc_gaps if gap != float('inf')]
        
        if auc_gaps:
            best_model = min(auc_gaps, key=lambda x: x[1])
            analysis['best_generalizing_model'] = {
                'model': best_model[0],
                'auc_gap': best_model[1]
            }
        
        # Save analysis
        output_file = self.results_dir / "generalization_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Generalization analysis saved to {output_file}")
        return analysis
    
    def rank_models_by_performance(self, cv_results: Dict[str, Any], 
                                  metric: str = 'auc') -> Dict[str, Any]:
        """Rank models by their performance across different scenarios"""
        
        logger.info(f"Ranking models by {metric} performance")
        
        # Collect performance data
        model_scores = {model: {'within': [], 'cross': [], 'all': []} for model in self.models}
        
        for result in cv_results.values():
            model = result['model_name']
            train_ds = result['train_dataset']
            test_ds = result['test_dataset']
            
            if metric in result['metrics']:
                score = result['metrics'][metric]
                model_scores[model]['all'].append(score)
                
                if train_ds == test_ds:
                    model_scores[model]['within'].append(score)
                else:
                    model_scores[model]['cross'].append(score)
        
        # Calculate rankings
        rankings = {
            'within_dataset': [],
            'cross_dataset': [],
            'overall': []
        }
        
        # Within-dataset ranking
        within_means = [(model, np.mean(scores['within']) if scores['within'] else 0) 
                       for model, scores in model_scores.items()]
        within_means.sort(key=lambda x: x[1], reverse=True)
        rankings['within_dataset'] = [
            {'rank': i+1, 'model': model, 'mean_score': score, 'scenario': 'within_dataset'}
            for i, (model, score) in enumerate(within_means)
        ]
        
        # Cross-dataset ranking
        cross_means = [(model, np.mean(scores['cross']) if scores['cross'] else 0) 
                      for model, scores in model_scores.items()]
        cross_means.sort(key=lambda x: x[1], reverse=True)
        rankings['cross_dataset'] = [
            {'rank': i+1, 'model': model, 'mean_score': score, 'scenario': 'cross_dataset'}
            for i, (model, score) in enumerate(cross_means)
        ]
        
        # Overall ranking
        overall_means = [(model, np.mean(scores['all']) if scores['all'] else 0) 
                        for model, scores in model_scores.items()]
        overall_means.sort(key=lambda x: x[1], reverse=True)
        rankings['overall'] = [
            {'rank': i+1, 'model': model, 'mean_score': score, 'scenario': 'overall'}
            for i, (model, score) in enumerate(overall_means)
        ]
        
        # Create ranking comparison
        ranking_comparison = pd.DataFrame()
        for scenario, ranks in rankings.items():
            df_temp = pd.DataFrame(ranks)
            df_temp['scenario'] = scenario
            ranking_comparison = pd.concat([ranking_comparison, df_temp], ignore_index=True)
        
        # Save rankings
        rankings_file = self.results_dir / f"model_rankings_{metric}.json"
        with open(rankings_file, 'w') as f:
            json.dump(rankings, f, indent=2, default=str)
        
        rankings_csv = self.results_dir / f"model_rankings_{metric}.csv"
        ranking_comparison.to_csv(rankings_csv, index=False)
        
        logger.info(f"Model rankings saved to {rankings_file}")
        return rankings
    
    def analyze_dataset_difficulty(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which datasets are more difficult to classify"""
        
        logger.info("Analyzing dataset classification difficulty")
        
        difficulty_analysis = {
            'as_target': {},  # How difficult each dataset is as a test set
            'as_source': {},  # How well each dataset generalizes as training set
            'overall_ranking': {}
        }
        
        # Analyze difficulty as target dataset
        for target_dataset in self.datasets:
            target_scores = []
            
            for result in cv_results.values():
                if (result['test_dataset'] == target_dataset and 
                    result['train_dataset'] != target_dataset):  # Only cross-dataset
                    if 'auc' in result['metrics']:
                        target_scores.append(result['metrics']['auc'])
            
            if target_scores:
                difficulty_analysis['as_target'][target_dataset] = {
                    'mean_auc': float(np.mean(target_scores)),
                    'std_auc': float(np.std(target_scores)),
                    'min_auc': float(np.min(target_scores)),
                    'max_auc': float(np.max(target_scores)),
                    'num_experiments': len(target_scores),
                    'difficulty_rank': 0  # Will be filled later
                }
        
        # Analyze generalization ability as source dataset
        for source_dataset in self.datasets:
            source_scores = []
            
            for result in cv_results.values():
                if (result['train_dataset'] == source_dataset and 
                    result['test_dataset'] != source_dataset):  # Only cross-dataset
                    if 'auc' in result['metrics']:
                        source_scores.append(result['metrics']['auc'])
            
            if source_scores:
                difficulty_analysis['as_source'][source_dataset] = {
                    'mean_auc': float(np.mean(source_scores)),
                    'std_auc': float(np.std(source_scores)),
                    'min_auc': float(np.min(source_scores)),
                    'max_auc': float(np.max(source_scores)),
                    'num_experiments': len(source_scores),
                    'generalization_rank': 0  # Will be filled later
                }
        
        # Rank datasets by difficulty (as target) - lower AUC = more difficult
        target_ranking = sorted(difficulty_analysis['as_target'].items(), 
                               key=lambda x: x[1]['mean_auc'])
        for i, (dataset, data) in enumerate(target_ranking):
            difficulty_analysis['as_target'][dataset]['difficulty_rank'] = i + 1
        
        # Rank datasets by generalization ability (as source) - higher AUC = better generalization
        source_ranking = sorted(difficulty_analysis['as_source'].items(), 
                               key=lambda x: x[1]['mean_auc'], reverse=True)
        for i, (dataset, data) in enumerate(source_ranking):
            difficulty_analysis['as_source'][dataset]['generalization_rank'] = i + 1
        
        # Overall ranking combining both aspects
        difficulty_analysis['overall_ranking'] = {
            'most_difficult_target': target_ranking[0][0] if target_ranking else None,
            'easiest_target': target_ranking[-1][0] if target_ranking else None,
            'best_source': source_ranking[0][0] if source_ranking else None,
            'worst_source': source_ranking[-1][0] if source_ranking else None
        }
        
        # Save analysis
        output_file = self.results_dir / "dataset_difficulty_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(difficulty_analysis, f, indent=2, default=str)
        
        logger.info(f"Dataset difficulty analysis saved to {output_file}")
        return difficulty_analysis

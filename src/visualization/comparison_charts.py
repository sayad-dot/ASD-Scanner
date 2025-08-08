import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from src.utils.data_utils import DataUtils
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ComparisonCharts:
    """Generate comparison charts for model performance analysis"""
    
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Define consistent colors for models
        self.model_colors = {
            'tabnet': '#2E86AB',
            'random_forest': '#A23B72', 
            'xgboost': '#F18F01',
            'svm': '#C73E1D'
        }
    
    def plot_model_comparison_boxplot(self, results: Dict[str, Any], 
                                    metric: str = 'auc',
                                    scenario: str = 'all') -> str:
        """Create boxplot comparing models across all experiments"""
        
        # Filter results by scenario
        filtered_results = DataUtils.filter_results_by_scenario(results, scenario)
        
        # Create DataFrame for plotting
        df = DataUtils.create_results_dataframe(filtered_results)
        
        if df.empty or metric not in df.columns:
            logger.warning(f"No data available for metric: {metric}")
            return ""
        
        plt.figure(figsize=(12, 8))
        
        # Create boxplot
        box_plot = sns.boxplot(
            data=df, 
            x='model_name', 
            y=metric,
            palette=self.model_colors
        )
        
        # Customize plot
        plt.title(f'Model Comparison - {metric.upper()} ({scenario.replace("_", " ").title()})', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(f'{metric.upper()} Score', fontsize=12)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add mean markers
        means = df.groupby('model_name')[metric].mean()
        for i, model in enumerate(means.index):
            plt.scatter(i, means[model], marker='D', s=100, color='red', zorder=10)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add statistics annotations
        for i, model in enumerate(df['model_name'].unique()):
            model_data = df[df['model_name'] == model][metric]
            mean_val = model_data.mean()
            std_val = model_data.std()
            plt.text(i, mean_val + 0.02, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / f"model_comparison_boxplot_{metric}_{scenario}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison boxplot saved to {output_file}")
        return str(output_file)
    
    def plot_cross_vs_within_performance(self, results: Dict[str, Any], 
                                       metric: str = 'auc') -> str:
        """Compare cross-dataset vs within-dataset performance"""
        
        df = DataUtils.create_results_dataframe(results)
        
        if df.empty or metric not in df.columns:
            logger.warning(f"No data available for metric: {metric}")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Separate data
        within_data = df[df['is_cross_dataset'] == False]
        cross_data = df[df['is_cross_dataset'] == True]
        
        # Plot 1: Side-by-side comparison
        comparison_data = []
        for model in df['model_name'].unique():
            within_scores = within_data[within_data['model_name'] == model][metric].tolist()
            cross_scores = cross_data[cross_data['model_name'] == model][metric].tolist()
            
            for score in within_scores:
                comparison_data.append({'Model': model, 'Scenario': 'Within-Dataset', 'Score': score})
            for score in cross_scores:
                comparison_data.append({'Model': model, 'Scenario': 'Cross-Dataset', 'Score': score})
        
        comp_df = pd.DataFrame(comparison_data)
        
        sns.boxplot(data=comp_df, x='Model', y='Score', hue='Scenario', ax=ax1)
        ax1.set_title(f'Within vs Cross-Dataset Performance - {metric.upper()}')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Performance drop visualization
        performance_drops = []
        for model in df['model_name'].unique():
            within_mean = within_data[within_data['model_name'] == model][metric].mean()
            cross_mean = cross_data[cross_data['model_name'] == model][metric].mean()
            
            if not (np.isnan(within_mean) or np.isnan(cross_mean)):
                drop = within_mean - cross_mean
                drop_pct = (drop / within_mean) * 100 if within_mean > 0 else 0
                performance_drops.append({
                    'Model': model,
                    'Performance Drop': drop,
                    'Drop Percentage': drop_pct
                })
        
        if performance_drops:
            drop_df = pd.DataFrame(performance_drops)
            
            bars = ax2.bar(drop_df['Model'], drop_df['Performance Drop'], 
                          color=[self.model_colors.get(model, 'gray') for model in drop_df['Model']])
            ax2.set_title(f'Performance Drop (Within → Cross Dataset)')
            ax2.set_ylabel(f'{metric.upper()} Drop')
            ax2.set_xticklabels(drop_df['Model'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, drop_df['Drop Percentage']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"cross_vs_within_performance_{metric}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cross vs within performance comparison saved to {output_file}")
        return str(output_file)
    
    def plot_dataset_transfer_heatmap(self, results: Dict[str, Any], 
                                    metric: str = 'auc') -> str:
        """Create heatmap showing transfer performance between datasets"""
        
        datasets = ['adult', 'adolescent', 'child']
        models = ['tabnet', 'random_forest', 'xgboost', 'svm']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            
            # Create transfer matrix
            transfer_matrix = np.zeros((len(datasets), len(datasets)))
            
            for i, train_ds in enumerate(datasets):
                for j, test_ds in enumerate(datasets):
                    key = f"{model}_{train_ds}_to_{test_ds}"
                    if key in results and 'metrics' in results[key]:
                        transfer_matrix[i, j] = results[key]['metrics'].get(metric, 0)
                    else:
                        transfer_matrix[i, j] = np.nan
            
            # Create heatmap
            im = sns.heatmap(
                transfer_matrix,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                xticklabels=[ds.title() for ds in datasets],
                yticklabels=[f"Train {ds.title()}" for ds in datasets],
                ax=ax,
                cbar_kws={'label': metric.upper()}
            )
            
            ax.set_title(f'{model.replace("_", " ").title()}')
            ax.set_xlabel('Test Dataset')
            ax.set_ylabel('Train Dataset')
        
        plt.suptitle(f'Dataset Transfer Performance - {metric.upper()}', fontsize=16)
        plt.tight_layout()
        
        output_file = self.output_dir / f"dataset_transfer_heatmap_{metric}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dataset transfer heatmap saved to {output_file}")
        return str(output_file)
    
    def plot_statistical_significance_matrix(self, stats_results: Dict[str, Any]) -> str:
        """Plot statistical significance matrix between models"""
        
        if 'pairwise_comparisons' not in stats_results:
            logger.warning("No pairwise comparisons found in statistical results")
            return ""
        
        comparisons = stats_results['pairwise_comparisons']
        models = list(set([c['model1'] for c in comparisons] + [c['model2'] for c in comparisons]))
        models.sort()
        
        # Create significance matrix
        sig_matrix = np.ones((len(models), len(models)))  # Start with 1s (not significant)
        p_value_matrix = np.ones((len(models), len(models)))
        
        for comp in comparisons:
            model1_idx = models.index(comp['model1'])
            model2_idx = models.index(comp['model2'])
            
            p_val = comp.get('p_value', 1.0)
            significant = comp.get('significant', False)
            
            # Fill both sides of matrix
            sig_matrix[model1_idx, model2_idx] = 0 if significant else 1
            sig_matrix[model2_idx, model1_idx] = 0 if significant else 1
            p_value_matrix[model1_idx, model2_idx] = p_val
            p_value_matrix[model2_idx, model1_idx] = p_val
        
        # Set diagonal to 0 (self-comparison)
        np.fill_diagonal(sig_matrix, 0.5)
        np.fill_diagonal(p_value_matrix, 0.5)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Significance matrix
        sns.heatmap(
            sig_matrix,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn',
            xticklabels=[m.replace('_', ' ').title() for m in models],
            yticklabels=[m.replace('_', ' ').title() for m in models],
            ax=ax1,
            cbar_kws={'label': 'Significant (0) / Not Significant (1)'}
        )
        ax1.set_title('Statistical Significance Matrix')
        
        # P-value matrix
        sns.heatmap(
            p_value_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            xticklabels=[m.replace('_', ' ').title() for m in models],
            yticklabels=[m.replace('_', ' ').title() for m in models],
            ax=ax2,
            cbar_kws={'label': 'P-value'}
        )
        ax2.set_title('P-value Matrix')
        
        plt.tight_layout()
        
        output_file = self.output_dir / "statistical_significance_matrix.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Statistical significance matrix saved to {output_file}")
        return str(output_file)
    
    def plot_metric_radar_chart(self, results: Dict[str, Any], 
                              models: List[str] = None) -> str:
        """Create radar chart comparing multiple metrics across models"""
        
        if models is None:
            models = ['tabnet', 'random_forest', 'xgboost', 'svm']
        
        metrics = ['auc', 'accuracy', 'f1', 'precision', 'recall', 'specificity']
        
        # Aggregate metrics by model
        model_metrics = {}
        for model in models:
            model_metrics[model] = {}
            for metric in metrics:
                values = []
                for result in results.values():
                    if (result.get('model_name') == model and 
                        'metrics' in result and 
                        metric in result['metrics']):
                        values.append(result['metrics'][metric])
                
                model_metrics[model][metric] = np.mean(values) if values else 0
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for model in models:
            values = [model_metrics[model][metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=model.replace('_', ' ').title(),
                   color=self.model_colors.get(model, 'gray'))
            ax.fill(angles, values, alpha=0.1, 
                   color=self.model_colors.get(model, 'gray'))
        
        # Customize radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Model Performance Radar Chart\n(Average Across All Experiments)', 
                 size=14, fontweight='bold', pad=20)
        
        output_file = self.output_dir / "metric_radar_chart.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metric radar chart saved to {output_file}")
        return str(output_file)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json
from sklearn.metrics import confusion_matrix
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class PerformancePlots:
    """Generate performance visualization plots"""
    
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_performance_matrix(self, performance_matrices: Dict[str, pd.DataFrame], 
                               metric: str = 'auc') -> str:
        """Plot performance matrices for all models"""
        
        n_models = len(performance_matrices)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (model_name, matrix) in enumerate(performance_matrices.items()):
            ax = axes[idx]
            
            # Create heatmap
            sns.heatmap(
                matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                vmin=0, 
                vmax=1,
                ax=ax,
                cbar_kws={'label': metric.upper()}
            )
            
            ax.set_title(f'{model_name.replace("_", " ").title()} - {metric.upper()}')
            ax.set_xlabel('Test Dataset')
            ax.set_ylabel('Train Dataset')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"performance_matrices_{metric}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance matrices saved to {output_file}")
        return str(output_file)
    
    def plot_cross_dataset_comparison(self, results: Dict[str, Any], 
                                    metric: str = 'auc') -> str:
        """Plot cross-dataset performance comparison"""
        
        # Extract data for plotting
        data = []
        for key, result in results.items():
            model_name = result['model_name']
            train_dataset = result['train_dataset']
            test_dataset = result['test_dataset']
            metric_value = result['metrics'][metric]
            
            data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train Dataset': train_dataset.title(),
                'Test Dataset': test_dataset.title(),
                'Metric Value': metric_value,
                'Cross-Dataset': train_dataset != test_dataset
            })
        
        df = pd.DataFrame(data)
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Within-dataset performance
        within_df = df[df['Cross-Dataset'] == False]
        sns.boxplot(data=within_df, x='Model', y='Metric Value', ax=ax1)
        ax1.set_title(f'Within-Dataset Performance ({metric.upper()})')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Cross-dataset performance
        cross_df = df[df['Cross-Dataset'] == True]
        sns.boxplot(data=cross_df, x='Model', y='Metric Value', ax=ax2)
        ax2.set_title(f'Cross-Dataset Performance ({metric.upper()})')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"cross_dataset_comparison_{metric}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cross-dataset comparison saved to {output_file}")
        return str(output_file)
    
    def plot_roc_curves(self, results: Dict[str, Any], 
                       train_dataset: str = 'adult') -> str:
        """Plot ROC curves for all models trained on a specific dataset"""
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange']
        model_color = dict(zip(['tabnet', 'random_forest', 'xgboost', 'svm'], colors))
        
        for test_dataset in ['adult', 'adolescent', 'child']:
            plt.subplot(2, 2, ['adult', 'adolescent', 'child'].index(test_dataset) + 1)
            
            for model_name in ['tabnet', 'random_forest', 'xgboost', 'svm']:
                key = f"{model_name}_{train_dataset}_to_{test_dataset}"
                
                if key in results and 'roc_curve' in results[key]['metrics']:
                    roc_data = results[key]['metrics']['roc_curve']
                    
                    if roc_data['fpr'] and roc_data['tpr']:
                        auc_score = results[key]['metrics']['auc']
                        
                        plt.plot(
                            roc_data['fpr'], 
                            roc_data['tpr'],
                            color=model_color[model_name],
                            label=f"{model_name.replace('_', ' ').title()} (AUC={auc_score:.3f})",
                            linewidth=2
                        )
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves: {train_dataset.title()} → {test_dataset.title()}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"roc_curves_{train_dataset}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {output_file}")
        return str(output_file)
    
    def plot_confusion_matrices(self, results: Dict[str, Any], 
                               model_name: str = 'tabnet') -> str:
        """Plot confusion matrices for a specific model across all dataset combinations"""
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        datasets = ['adult', 'adolescent', 'child']
        
        for i, train_dataset in enumerate(datasets):
            for j, test_dataset in enumerate(datasets):
                ax = axes[i, j]
                key = f"{model_name}_{train_dataset}_to_{test_dataset}"
                
                if key in results:
                    cm = np.array(results[key]['metrics']['confusion_matrix'])
                    
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        ax=ax,
                        xticklabels=['No ASD', 'ASD'],
                        yticklabels=['No ASD', 'ASD']
                    )
                    
                    accuracy = results[key]['metrics']['accuracy']
                    ax.set_title(f'{train_dataset.title()} → {test_dataset.title()}\nAcc: {accuracy:.3f}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{train_dataset.title()} → {test_dataset.title()}')
        
        plt.suptitle(f'Confusion Matrices - {model_name.replace("_", " ").title()}', fontsize=16)
        plt.tight_layout()
        
        output_file = self.output_dir / f"confusion_matrices_{model_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrices saved to {output_file}")
        return str(output_file)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json
from src.utils.data_utils import DataUtils
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class DashboardGenerator:
    """Generate interactive dashboards for cross-dataset validation results"""
    
    def __init__(self, output_dir: str = "results/dashboards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define consistent colors
        self.model_colors = {
            'tabnet': '#2E86AB',
            'random_forest': '#A23B72',
            'xgboost': '#F18F01', 
            'svm': '#C73E1D'
        }
        
        self.dataset_colors = {
            'adult': '#1f77b4',
            'adolescent': '#ff7f0e',
            'child': '#2ca02c'
        }
    
    def create_performance_overview_dashboard(self, results: Dict[str, Any]) -> str:
        """Create comprehensive performance overview dashboard"""
        
        df = DataUtils.create_results_dataframe(results)
        
        if df.empty:
            logger.warning("No data available for dashboard")
            return ""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Model Performance Comparison (AUC)',
                'Cross vs Within Dataset Performance',
                'Performance by Train-Test Dataset Pairs',
                'Metric Distribution by Model',
                'Dataset Transfer Performance',
                'Sample Size vs Performance'
            ],
            specs=[
                [{"type": "box"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "violin"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Model performance comparison (boxplot)
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            fig.add_trace(
                go.Box(
                    y=model_data['auc'],
                    name=model.replace('_', ' ').title(),
                    marker_color=self.model_colors.get(model, 'gray'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Cross vs within performance comparison
        within_means = df[df['is_cross_dataset'] == False].groupby('model_name')['auc'].mean()
        cross_means = df[df['is_cross_dataset'] == True].groupby('model_name')['auc'].mean()
        
        fig.add_trace(
            go.Bar(
                x=within_means.index,
                y=within_means.values,
                name='Within-Dataset',
                marker_color='lightblue',
                showlegend=True
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=cross_means.index,
                y=cross_means.values,
                name='Cross-Dataset',
                marker_color='lightcoral',
                showlegend=True
            ),
            row=1, col=2
        )
        
        # 3. Performance by dataset pairs (scatter plot)
        fig.add_trace(
            go.Scatter(
                x=df['train_dataset'] + ' → ' + df['test_dataset'],
                y=df['auc'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=[self.model_colors.get(model, 'gray') for model in df['model_name']],
                    opacity=0.7
                ),
                text=df['model_name'],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Metric distribution (violin plot)
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            if metric in df.columns:
                fig.add_trace(
                    go.Violin(
                        y=df[metric],
                        name=metric.upper(),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # 5. Dataset transfer heatmap data
        datasets = ['adult', 'adolescent', 'child']
        transfer_data = []
        
        for train_ds in datasets:
            row = []
            for test_ds in datasets:
                # Average AUC across all models for this train-test pair
                pair_data = df[(df['train_dataset'] == train_ds) & (df['test_dataset'] == test_ds)]
                avg_auc = pair_data['auc'].mean() if not pair_data.empty else 0
                row.append(avg_auc)
            transfer_data.append(row)
        
        fig.add_trace(
            go.Heatmap(
                z=transfer_data,
                x=[ds.title() for ds in datasets],
                y=[f"Train {ds.title()}" for ds in datasets],
                colorscale='RdYlBu_r',
                showscale=True,
                zmin=0,
                zmax=1
            ),
            row=3, col=1
        )
        
        # 6. Sample size vs performance
        fig.add_trace(
            go.Scatter(
                x=df['test_samples'],
                y=df['auc'],
                mode='markers',
                marker=dict(
                    size=df['train_samples'] / 50,  # Size proportional to training samples
                    color=[self.dataset_colors.get(ds, 'gray') for ds in df['test_dataset']],
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                text=df['model_name'] + '<br>' + df['train_dataset'] + ' → ' + df['test_dataset'],
                showlegend=False
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Cross-Dataset Validation Performance Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="AUC Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_yaxes(title_text="Mean AUC Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Dataset Pair", row=2, col=1)
        fig.update_yaxes(title_text="AUC Score", row=2, col=1)
        
        fig.update_xaxes(title_text="Metric", row=2, col=2)
        fig.update_yaxes(title_text="Score", row=2, col=2)
        
        fig.update_xaxes(title_text="Test Dataset", row=3, col=1)
        fig.update_yaxes(title_text="Train Dataset", row=3, col=1)
        
        fig.update_xaxes(title_text="Test Sample Size", row=3, col=2)
        fig.update_yaxes(title_text="AUC Score", row=3, col=2)
        
        # Save dashboard
        output_file = self.output_dir / "performance_overview_dashboard.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Performance overview dashboard saved to {output_file}")
        return str(output_file)
    
    def create_model_comparison_dashboard(self, results: Dict[str, Any], 
                                        stats_results: Dict[str, Any] = None) -> str:
        """Create detailed model comparison dashboard"""
        
        df = DataUtils.create_results_dataframe(results)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Model Performance Distribution',
                'Statistical Significance Matrix',
                'Performance by Scenario',
                'Ranking Comparison'
            ],
            specs=[
                [{"type": "box"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. Performance distribution
        models = df['model_name'].unique()
        for model in models:
            model_data = df[df['model_name'] == model]
            fig.add_trace(
                go.Box(
                    y=model_data['auc'],
                    name=model.replace('_', ' ').title(),
                    marker_color=self.model_colors.get(model, 'gray'),
                    boxpoints='outliers'
                ),
                row=1, col=1
            )
        
        # 2. Statistical significance matrix
        if stats_results and 'pairwise_comparisons' in stats_results:
            comparisons = stats_results['pairwise_comparisons']
            model_list = sorted(list(set([c['model1'] for c in comparisons] + [c['model2'] for c in comparisons])))
            
            sig_matrix = np.ones((len(model_list), len(model_list)))
            
            for comp in comparisons:
                i = model_list.index(comp['model1'])
                j = model_list.index(comp['model2'])
                sig_val = 0 if comp.get('significant', False) else 1
                sig_matrix[i, j] = sig_val
                sig_matrix[j, i] = sig_val
            
            np.fill_diagonal(sig_matrix, 0.5)
            
            fig.add_trace(
                go.Heatmap(
                    z=sig_matrix,
                    x=[m.replace('_', ' ').title() for m in model_list],
                    y=[m.replace('_', ' ').title() for m in model_list],
                    colorscale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
                    showscale=True
                ),
                row=1, col=2
            )
        
        # 3. Performance by scenario
        within_data = df[df['is_cross_dataset'] == False].groupby('model_name')['auc'].mean()
        cross_data = df[df['is_cross_dataset'] == True].groupby('model_name')['auc'].mean()
        
        fig.add_trace(
            go.Bar(
                x=within_data.index,
                y=within_data.values,
                name='Within-Dataset',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=cross_data.index,
                y=cross_data.values,
                name='Cross-Dataset',
                marker_color='lightcoral'
            ),
            row=2, col=1
        )
        
        # 4. Overall ranking
        overall_means = df.groupby('model_name')['auc'].mean().sort_values(ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=overall_means.index,
                y=overall_means.values,
                marker_color=[self.model_colors.get(model, 'gray') for model in overall_means.index],
                text=[f"Rank {i+1}" for i in range(len(overall_means))],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Model Comparison Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Save dashboard
        output_file = self.output_dir / "model_comparison_dashboard.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Model comparison dashboard saved to {output_file}")
        return str(output_file)
    
    def create_interactive_results_table(self, results: Dict[str, Any]) -> str:
        """Create interactive results table with filtering and sorting"""
        
        df = DataUtils.create_results_dataframe(results)
        
        if df.empty:
            logger.warning("No data available for results table")
            return ""
        
        # Select relevant columns
        display_columns = [
            'model_name', 'train_dataset', 'test_dataset', 
            'train_samples', 'test_samples', 'is_cross_dataset',
            'auc', 'accuracy', 'f1', 'precision', 'recall'
        ]
        
        display_df = df[display_columns].copy()
        
        # Format column names
        display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
        
        # Round numerical columns
        numerical_cols = ['Auc', 'Accuracy', 'F1', 'Precision', 'Recall']
        for col in numerical_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        
        # Create interactive table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(display_df.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[display_df[col] for col in display_df.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title={
                'text': "Cross-Dataset Validation Results",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=600
        )
        
        # Save table
        output_file = self.output_dir / "interactive_results_table.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Interactive results table saved to {output_file}")
        return str(output_file)

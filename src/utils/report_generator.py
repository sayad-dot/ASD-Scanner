import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.data_utils import DataUtils
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ReportGenerator:
    """Generate comprehensive reports for cross-dataset validation experiments"""
    
    def __init__(self, output_dir: str = "results/cross_dataset_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = ['tabnet', 'random_forest', 'xgboost', 'svm']
        self.datasets = ['adult', 'adolescent', 'child']
        self.metrics = ['auc', 'accuracy', 'f1', 'precision', 'recall', 'specificity']
    
    def generate_executive_summary(self, cv_results: Dict[str, Any], 
                                 stats_results: Dict[str, Any] = None) -> str:
        """Generate executive summary of cross-dataset validation results"""
        
        logger.info("Generating executive summary report")
        
        # Calculate summary statistics
        df = DataUtils.create_results_dataframe(cv_results)
        
        if df.empty:
            logger.warning("No data available for executive summary")
            return ""
        
        # Prepare summary data
        summary_data = {
            'experiment_overview': self._get_experiment_overview(df),
            'key_findings': self._extract_key_findings(df),
            'model_performance': self._summarize_model_performance(df),
            'generalization_analysis': self._analyze_generalization(df),
            'statistical_significance': self._summarize_statistical_results(stats_results) if stats_results else None,
            'recommendations': self._generate_recommendations(df)
        }
        
        # Generate HTML report
        html_content = self._create_executive_summary_html(summary_data)
        
        # Save report
        output_file = self.output_dir / f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Executive summary saved to {output_file}")
        return str(output_file)
    
    def generate_detailed_performance_report(self, cv_results: Dict[str, Any]) -> str:
        """Generate detailed performance analysis report"""
        
        logger.info("Generating detailed performance report")
        
        df = DataUtils.create_results_dataframe(cv_results)
        
        if df.empty:
            logger.warning("No data available for detailed report")
            return ""
        
        # Create detailed analysis
        report_sections = {
            'overview': self._create_overview_section(df),
            'within_dataset_analysis': self._analyze_within_dataset_performance(df),
            'cross_dataset_analysis': self._analyze_cross_dataset_performance(df),
            'model_comparison': self._create_detailed_model_comparison(df),
            'dataset_analysis': self._analyze_dataset_characteristics(df),
            'performance_matrices': self._create_performance_matrices_section(df),
            'failure_analysis': self._analyze_failure_cases(df)
        }
        
        # Generate comprehensive HTML report
        html_content = self._create_detailed_report_html(report_sections)
        
        # Save report
        output_file = self.output_dir / f"detailed_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Detailed performance report saved to {output_file}")
        return str(output_file)
    
    def generate_tabnet_analysis_report(self, cv_results: Dict[str, Any]) -> str:
        """Generate specialized report focusing on TabNet performance vs baselines"""
        
        logger.info("Generating TabNet analysis report")
        
        df = DataUtils.create_results_dataframe(cv_results)
        
        if df.empty:
            logger.warning("No data available for TabNet analysis")
            return ""
        
        # TabNet-specific analysis
        tabnet_analysis = {
            'tabnet_vs_baselines': self._compare_tabnet_vs_baselines(df),
            'tabnet_generalization': self._analyze_tabnet_generalization(df),
            'tabnet_advantages': self._identify_tabnet_advantages(df),
            'tabnet_limitations': self._identify_tabnet_limitations(df),
            'feature_importance_analysis': self._analyze_tabnet_interpretability(cv_results),
            'computational_analysis': self._analyze_computational_aspects(cv_results)
        }
        
        # Generate TabNet-focused HTML report
        html_content = self._create_tabnet_analysis_html(tabnet_analysis)
        
        # Save report
        output_file = self.output_dir / f"tabnet_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"TabNet analysis report saved to {output_file}")
        return str(output_file)
    
    def generate_publication_ready_summary(self, cv_results: Dict[str, Any], 
                                         stats_results: Dict[str, Any] = None) -> str:
        """Generate publication-ready summary with tables and statistics"""
        
        logger.info("Generating publication-ready summary")
        
        df = DataUtils.create_results_dataframe(cv_results)
        
        # Create publication tables
        tables = {
            'performance_summary': self._create_performance_summary_table(df),
            'statistical_comparison': self._create_statistical_comparison_table(stats_results) if stats_results else None,
            'cross_dataset_matrix': self._create_cross_dataset_matrix_table(df),
            'model_rankings': self._create_model_ranking_table(df)
        }
        
        # Generate LaTeX-style HTML
        html_content = self._create_publication_summary_html(tables)
        
        # Save report
        output_file = self.output_dir / f"publication_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Also save as CSV files for easy copying to papers
        for table_name, table_data in tables.items():
            if table_data is not None and isinstance(table_data, pd.DataFrame):
                csv_file = self.output_dir / f"{table_name}_table.csv"
                table_data.to_csv(csv_file, index=False)
        
        logger.info(f"Publication-ready summary saved to {output_file}")
        return str(output_file)
    
    def _get_experiment_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overview statistics of the experiments"""
        return {
            'total_experiments': len(df),
            'models_tested': df['model_name'].nunique(),
            'datasets_used': df['train_dataset'].nunique(),
            'cross_dataset_experiments': len(df[df['is_cross_dataset'] == True]),
            'within_dataset_experiments': len(df[df['is_cross_dataset'] == False]),
            'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _extract_key_findings(self, df: pd.DataFrame) -> Dict[str, str]:
        """Extract key findings from the results"""
        findings = {}
        
        # Best overall model
        best_model = df.groupby('model_name')['auc'].mean().idxmax()
        best_auc = df.groupby('model_name')['auc'].mean().max()
        findings['best_overall_model'] = f"{best_model.replace('_', ' ').title()} achieved the highest mean AUC of {best_auc:.4f}"
        
        # Best cross-dataset model
        cross_df = df[df['is_cross_dataset'] == True]
        if not cross_df.empty:
            best_cross_model = cross_df.groupby('model_name')['auc'].mean().idxmax()
            best_cross_auc = cross_df.groupby('model_name')['auc'].mean().max()
            findings['best_cross_dataset_model'] = f"{best_cross_model.replace('_', ' ').title()} showed best cross-dataset generalization with mean AUC of {best_cross_auc:.4f}"
        
        # Generalization gap
        within_mean = df[df['is_cross_dataset'] == False]['auc'].mean()
        cross_mean = df[df['is_cross_dataset'] == True]['auc'].mean()
        gap = within_mean - cross_mean
        findings['generalization_gap'] = f"Average generalization gap: {gap:.4f} ({gap/within_mean*100:.1f}% performance drop)"
        
        return findings
    
    def _summarize_model_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create model performance summary table"""
        summary_data = []
        
        for model in self.models:
            model_data = df[df['model_name'] == model]
            if not model_data.empty:
                within_data = model_data[model_data['is_cross_dataset'] == False]
                cross_data = model_data[model_data['is_cross_dataset'] == True]
                
                summary_data.append({
                    'Model': model.replace('_', ' ').title(),
                    'Overall_AUC_Mean': model_data['auc'].mean(),
                    'Overall_AUC_Std': model_data['auc'].std(),
                    'Within_Dataset_AUC': within_data['auc'].mean() if not within_data.empty else np.nan,
                    'Cross_Dataset_AUC': cross_data['auc'].mean() if not cross_data.empty else np.nan,
                    'Generalization_Gap': (within_data['auc'].mean() - cross_data['auc'].mean()) if not within_data.empty and not cross_data.empty else np.nan,
                    'Experiments_Count': len(model_data)
                })
        
        return pd.DataFrame(summary_data).round(4)
    
    def _analyze_generalization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze generalization capabilities"""
        analysis = {}
        
        # Calculate generalization gaps for each model
        for model in self.models:
            model_data = df[df['model_name'] == model]
            within_scores = model_data[model_data['is_cross_dataset'] == False]['auc']
            cross_scores = model_data[model_data['is_cross_dataset'] == True]['auc']
            
            if not within_scores.empty and not cross_scores.empty:
                gap = within_scores.mean() - cross_scores.mean()
                analysis[model] = {
                    'gap': gap,
                    'gap_percentage': (gap / within_scores.mean()) * 100 if within_scores.mean() > 0 else 0
                }
        
        # Find best generalizing model
        if analysis:
            best_generalizer = min(analysis.items(), key=lambda x: x[1]['gap'])
            analysis['best_generalizer'] = best_generalizer[0]
            analysis['smallest_gap'] = best_generalizer[1]['gap']
        
        return analysis
    
    def _summarize_statistical_results(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize statistical significance results"""
        if not stats_results or 'pairwise_comparisons' not in stats_results:
            return None
        
        comparisons = stats_results['pairwise_comparisons']
        significant_pairs = [c for c in comparisons if c.get('significant', False)]
        
        return {
            'total_comparisons': len(comparisons),
            'significant_comparisons': len(significant_pairs),
            'significance_rate': len(significant_pairs) / len(comparisons) if comparisons else 0,
            'most_significant': min(comparisons, key=lambda x: x.get('p_value', 1.0)) if comparisons else None
        }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Model selection recommendation
        best_model = df.groupby('model_name')['auc'].mean().idxmax()
        recommendations.append(f"For overall performance, consider using {best_model.replace('_', ' ').title()} as it achieved the highest mean AUC.")
        
        # Cross-dataset recommendation
        cross_df = df[df['is_cross_dataset'] == True]
        if not cross_df.empty:
            best_cross_model = cross_df.groupby('model_name')['auc'].mean().idxmax()
            recommendations.append(f"For cross-dataset generalization, {best_cross_model.replace('_', ' ').title()} shows the best performance.")
        
        # Dataset-specific recommendations
        for dataset in self.datasets:
            dataset_performance = df[df['test_dataset'] == dataset].groupby('model_name')['auc'].mean()
            if not dataset_performance.empty:
                best_for_dataset = dataset_performance.idxmax()
                recommendations.append(f"For {dataset} population, {best_for_dataset.replace('_', ' ').title()} performs best.")
        
        return recommendations
    
    def _compare_tabnet_vs_baselines(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare TabNet performance against baseline models"""
        tabnet_data = df[df['model_name'] == 'tabnet']
        baseline_data = df[df['model_name'] != 'tabnet']
        
        if tabnet_data.empty or baseline_data.empty:
            return {}
        
        comparison = {
            'tabnet_mean_auc': tabnet_data['auc'].mean(),
            'baselines_mean_auc': baseline_data['auc'].mean(),
            'tabnet_advantage': tabnet_data['auc'].mean() - baseline_data['auc'].mean(),
            'wins_against_baselines': 0,
            'total_comparisons': 0
        }
        
        # Count wins against each baseline
        for baseline in ['random_forest', 'xgboost', 'svm']:
            baseline_scores = df[df['model_name'] == baseline]['auc']
            if not baseline_scores.empty:
                comparison['total_comparisons'] += 1
                if tabnet_data['auc'].mean() > baseline_scores.mean():
                    comparison['wins_against_baselines'] += 1
        
        return comparison
    
    def _create_executive_summary_html(self, summary_data: Dict[str, Any]) -> str:
        """Create HTML content for executive summary"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cross-Dataset Validation Executive Summary</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }
                .metric { background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .highlight { background-color: #d5dbdb; font-weight: bold; padding: 2px 5px; border-radius: 3px; }
                table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                th, td { border: 1px solid #bdc3c7; padding: 10px; text-align: center; }
                th { background-color: #3498db; color: white; }
                .recommendation { background-color: #e8f5e8; padding: 10px; margin: 10px 0; border-left: 4px solid #27ae60; }
                .timestamp { text-align: right; color: #7f8c8d; font-style: italic; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Cross-Dataset ASD Detection Validation</h1>
                <h1>Executive Summary</h1>
                
                <div class="timestamp">Generated: {date}</div>
                
                <h2>Experiment Overview</h2>
                <div class="metric">
                    <strong>Total Experiments:</strong> {total_experiments}<br>
                    <strong>Models Tested:</strong> {models_tested}<br>
                    <strong>Datasets Used:</strong> {datasets_used}<br>
                    <strong>Cross-Dataset Experiments:</strong> {cross_dataset}<br>
                    <strong>Within-Dataset Experiments:</strong> {within_dataset}
                </div>
                
                <h2>Key Findings</h2>
                {key_findings_html}
                
                <h2>Model Performance Summary</h2>
                {performance_table_html}
                
                <h2>Statistical Significance</h2>
                {statistical_summary_html}
                
                <h2>Recommendations</h2>
                {recommendations_html}
            </div>
        </body>
        </html>
        """
        
        # Format key findings
        key_findings_html = ""
        if summary_data['key_findings']:
            for finding in summary_data['key_findings'].values():
                key_findings_html += f'<div class="metric">{finding}</div>'
        
        # Format performance table
        performance_table_html = ""
        if isinstance(summary_data['model_performance'], pd.DataFrame):
            performance_table_html = summary_data['model_performance'].to_html(classes='performance-table', escape=False, index=False)
        
        # Format statistical summary
        statistical_summary_html = ""
        if summary_data['statistical_significance']:
            stats = summary_data['statistical_significance']
            statistical_summary_html = f"""
            <div class="metric">
                <strong>Total Comparisons:</strong> {stats.get('total_comparisons', 'N/A')}<br>
                <strong>Significant Differences:</strong> {stats.get('significant_comparisons', 'N/A')}<br>
                <strong>Significance Rate:</strong> {stats.get('significance_rate', 0)*100:.1f}%
            </div>
            """
        else:
            statistical_summary_html = '<div class="metric">Statistical analysis not available</div>'
        
        # Format recommendations
        recommendations_html = ""
        if summary_data['recommendations']:
            for rec in summary_data['recommendations']:
                recommendations_html += f'<div class="recommendation">{rec}</div>'
        
        # Fill template
        return html_template.format(
            date=summary_data['experiment_overview']['date_generated'],
            total_experiments=summary_data['experiment_overview']['total_experiments'],
            models_tested=summary_data['experiment_overview']['models_tested'],
            datasets_used=summary_data['experiment_overview']['datasets_used'],
            cross_dataset=summary_data['experiment_overview']['cross_dataset_experiments'],
            within_dataset=summary_data['experiment_overview']['within_dataset_experiments'],
            key_findings_html=key_findings_html,
            performance_table_html=performance_table_html,
            statistical_summary_html=statistical_summary_html,
            recommendations_html=recommendations_html
        )
    
    def _create_detailed_report_html(self, report_sections: Dict[str, Any]) -> str:
        """Create comprehensive HTML report"""
        # This would be a much longer HTML template with detailed sections
        # For brevity, I'll provide a simplified version
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Detailed Cross-Dataset Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ccc; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Detailed Cross-Dataset Validation Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Overview</h2>
                {report_sections.get('overview', 'No overview data available')}
            </div>
            
            <div class="section">
                <h2>Within-Dataset Analysis</h2>
                {report_sections.get('within_dataset_analysis', 'No within-dataset analysis available')}
            </div>
            
            <div class="section">
                <h2>Cross-Dataset Analysis</h2>
                {report_sections.get('cross_dataset_analysis', 'No cross-dataset analysis available')}
            </div>
            
            <div class="section">
                <h2>Model Comparison</h2>
                {report_sections.get('model_comparison', 'No model comparison available')}
            </div>
        </body>
        </html>
        """
        return html_content
    
    def _create_tabnet_analysis_html(self, tabnet_analysis: Dict[str, Any]) -> str:
        """Create TabNet-specific analysis HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TabNet vs Baseline Models Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .highlight {{ background-color: #ffffcc; padding: 5px; }}
                .metric {{ background-color: #f0f0f0; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>TabNet Performance Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>TabNet vs Baseline Comparison</h2>
            <div class="metric">
                {tabnet_analysis.get('tabnet_vs_baselines', 'No comparison data available')}
            </div>
            
            <h2>Generalization Analysis</h2>
            <div class="metric">
                {tabnet_analysis.get('tabnet_generalization', 'No generalization analysis available')}
            </div>
            
            <h2>TabNet Advantages</h2>
            <div class="metric">
                {tabnet_analysis.get('tabnet_advantages', 'No advantages analysis available')}
            </div>
            
            <h2>TabNet Limitations</h2>
            <div class="metric">
                {tabnet_analysis.get('tabnet_limitations', 'No limitations analysis available')}
            </div>
        </body>
        </html>
        """
        return html_content
    
    def _create_publication_summary_html(self, tables: Dict[str, Any]) -> str:
        """Create publication-ready summary HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Publication Summary - Cross-Dataset ASD Detection</title>
            <style>
                body {{ font-family: 'Times New Roman', serif; margin: 40px; line-height: 1.6; }}
                h1 {{ text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #000; padding: 8px; text-align: center; }}
                th {{ background-color: #f0f0f0; font-weight: bold; }}
                .caption {{ font-style: italic; text-align: center; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Cross-Dataset Validation Results for ASD Detection</h1>
            <p><strong>Publication Summary</strong></p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Table 1: Overall Performance Summary</h2>
            {tables.get('performance_summary', pd.DataFrame()).to_html(index=False, escape=False) if isinstance(tables.get('performance_summary'), pd.DataFrame) else 'Table not available'}
            <div class="caption">Performance metrics across all cross-dataset validation experiments</div>
            
            <h2>Table 2: Cross-Dataset Transfer Matrix</h2>
            {tables.get('cross_dataset_matrix', pd.DataFrame()).to_html(index=False, escape=False) if isinstance(tables.get('cross_dataset_matrix'), pd.DataFrame) else 'Table not available'}
            <div class="caption">AUC scores for training on one dataset and testing on another</div>
            
            <h2>Table 3: Model Rankings</h2>
            {tables.get('model_rankings', pd.DataFrame()).to_html(index=False, escape=False) if isinstance(tables.get('model_rankings'), pd.DataFrame) else 'Table not available'}
            <div class="caption">Model performance rankings across different scenarios</div>
        </body>
        </html>
        """
        return html_content
    
    # Placeholder methods for detailed analysis (can be expanded)
    def _create_overview_section(self, df: pd.DataFrame) -> str:
        return f"Overview analysis with {len(df)} experiments"
    
    def _analyze_within_dataset_performance(self, df: pd.DataFrame) -> str:
        within_df = df[df['is_cross_dataset'] == False]
        return f"Within-dataset analysis with {len(within_df)} experiments"
    
    def _analyze_cross_dataset_performance(self, df: pd.DataFrame) -> str:
        cross_df = df[df['is_cross_dataset'] == True]
        return f"Cross-dataset analysis with {len(cross_df)} experiments"
    
    def _create_detailed_model_comparison(self, df: pd.DataFrame) -> str:
        return "Detailed model comparison analysis"
    
    def _analyze_dataset_characteristics(self, df: pd.DataFrame) -> str:
        return "Dataset characteristics analysis"
    
    def _create_performance_matrices_section(self, df: pd.DataFrame) -> str:
        return "Performance matrices section"
    
    def _analyze_failure_cases(self, df: pd.DataFrame) -> str:
        return "Failure cases analysis"
    
    def _analyze_tabnet_generalization(self, df: pd.DataFrame) -> str:
        return "TabNet generalization analysis"
    
    def _identify_tabnet_advantages(self, df: pd.DataFrame) -> str:
        return "TabNet advantages identification"
    
    def _identify_tabnet_limitations(self, df: pd.DataFrame) -> str:
        return "TabNet limitations analysis"
    
    def _analyze_tabnet_interpretability(self, cv_results: Dict[str, Any]) -> str:
        return "TabNet interpretability analysis"
    
    def _analyze_computational_aspects(self, cv_results: Dict[str, Any]) -> str:
        return "Computational aspects analysis"
    
    def _create_performance_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._summarize_model_performance(df)
    
    def _create_statistical_comparison_table(self, stats_results: Dict[str, Any]) -> Optional[pd.DataFrame]:
        if not stats_results or 'pairwise_comparisons' not in stats_results:
            return None
        
        comparisons = stats_results['pairwise_comparisons']
        table_data = []
        
        for comp in comparisons:
            table_data.append({
                'Model 1': comp.get('model1', '').replace('_', ' ').title(),
                'Model 2': comp.get('model2', '').replace('_', ' ').title(), 
                'P-value': comp.get('p_value', 'N/A'),
                'Significant': 'Yes' if comp.get('significant', False) else 'No',
                'Effect Size': comp.get('effect_size', 'N/A')
            })
        
        return pd.DataFrame(table_data)
    
    def _create_cross_dataset_matrix_table(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create a simplified cross-dataset matrix
        datasets = ['adult', 'adolescent', 'child']
        matrix_data = []
        
        for train_ds in datasets:
            row_data = {'Train Dataset': train_ds.title()}
            for test_ds in datasets:
                pair_data = df[(df['train_dataset'] == train_ds) & (df['test_dataset'] == test_ds)]
                avg_auc = pair_data['auc'].mean() if not pair_data.empty else 0
                row_data[f'Test {test_ds.title()}'] = f"{avg_auc:.3f}"
            matrix_data.append(row_data)
        
        return pd.DataFrame(matrix_data)
    
    def _create_model_ranking_table(self, df: pd.DataFrame) -> pd.DataFrame:
        ranking_data = []
        
        # Overall ranking
        overall_ranking = df.groupby('model_name')['auc'].mean().sort_values(ascending=False)
        for i, (model, score) in enumerate(overall_ranking.items()):
            ranking_data.append({
                'Rank': i + 1,
                'Model': model.replace('_', ' ').title(),
                'Mean AUC': f"{score:.4f}",
                'Scenario': 'Overall'
            })
        
        return pd.DataFrame(ranking_data)

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
import json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class DataUtils:
    """Utility functions for data processing and analysis"""
    
    @staticmethod
    def load_json_results(file_path: str) -> Dict[str, Any]:
        """Safely load JSON results file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return {}
    
    @staticmethod
    def save_json_results(data: Dict[str, Any], file_path: str) -> bool:
        """Safely save data to JSON file"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {e}")
            return False
    
    @staticmethod
    def calculate_class_balance(y: np.ndarray) -> Dict[str, Any]:
        """Calculate class balance statistics"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        balance_info = {
            'total_samples': total,
            'classes': unique.tolist(),
            'counts': counts.tolist(),
            'percentages': (counts / total * 100).tolist(),
            'balance_ratio': float(min(counts) / max(counts)) if len(counts) > 1 else 1.0,
            'is_balanced': abs(counts[0] - counts[1]) / total < 0.1 if len(counts) == 2 else False
        }
        
        return balance_info
    
    @staticmethod
    def create_stratified_splits(X: np.ndarray, y: np.ndarray, 
                               n_splits: int = 5, random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified splits for cross-validation"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        splits = []
        for train_idx, test_idx in skf.split(X, y):
            splits.append((train_idx, test_idx))
        
        logger.info(f"Created {n_splits} stratified splits")
        return splits
    
    @staticmethod
    def standardize_features(X_train: np.ndarray, X_test: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
        """Standardize features using training set statistics"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, scaler
    
    @staticmethod
    def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, LabelEncoder]:
        """Encode string labels to integers"""
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        return y_encoded, encoder
    
    @staticmethod
    def aggregate_metrics(results: Dict[str, Any], metric: str = 'auc') -> Dict[str, Dict[str, float]]:
        """Aggregate metrics by model across all experiments"""
        aggregated = {}
        
        for key, result in results.items():
            model_name = result.get('model_name', 'unknown')
            
            if model_name not in aggregated:
                aggregated[model_name] = []
            
            if 'metrics' in result and metric in result['metrics']:
                aggregated[model_name].append(result['metrics'][metric])
        
        # Calculate summary statistics
        summary = {}
        for model, values in aggregated.items():
            if values:
                summary[model] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'count': len(values),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75))
                }
        
        return summary
    
    @staticmethod
    def filter_results_by_scenario(results: Dict[str, Any], scenario: str = 'cross_dataset') -> Dict[str, Any]:
        """Filter results by scenario (within_dataset, cross_dataset, or all)"""
        filtered = {}
        
        for key, result in results.items():
            train_ds = result.get('train_dataset', '')
            test_ds = result.get('test_dataset', '')
            
            if scenario == 'within_dataset' and train_ds == test_ds:
                filtered[key] = result
            elif scenario == 'cross_dataset' and train_ds != test_ds:
                filtered[key] = result
            elif scenario == 'all':
                filtered[key] = result
        
        logger.info(f"Filtered {len(filtered)} results for scenario: {scenario}")
        return filtered
    
    @staticmethod
    def create_results_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
        """Convert results dictionary to pandas DataFrame for analysis"""
        data = []
        
        for key, result in results.items():
            row = {
                'experiment_id': key,
                'model_name': result.get('model_name', ''),
                'train_dataset': result.get('train_dataset', ''),
                'test_dataset': result.get('test_dataset', ''),
                'train_samples': result.get('train_samples', 0),
                'test_samples': result.get('test_samples', 0),
                'is_cross_dataset': result.get('train_dataset', '') != result.get('test_dataset', '')
            }
            
            # Add metrics
            if 'metrics' in result:
                for metric, value in result['metrics'].items():
                    if isinstance(value, (int, float)):
                        row[metric] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    @staticmethod
    def calculate_confidence_intervals(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence intervals for a list of values"""
        if len(values) < 2:
            return np.nan, np.nan
        
        mean = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(len(values))
        
        # Use t-distribution for small samples
        from scipy.stats import t
        t_val = t.ppf((1 + confidence) / 2, len(values) - 1)
        margin_error = t_val * std_err
        
        return mean - margin_error, mean + margin_error
    
    @staticmethod
    def detect_outliers(values: np.ndarray, method: str = 'iqr') -> Tuple[np.ndarray, np.ndarray]:
        """Detect outliers in data using IQR or Z-score method"""
        if method == 'iqr':
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (values < lower_bound) | (values > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((values - np.mean(values)) / np.std(values))
            outliers = z_scores > 3
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        outlier_indices = np.where(outliers)[0]
        outlier_values = values[outliers]
        
        return outlier_indices, outlier_values
    
    @staticmethod
    def summarize_experiment_results(results_file: str) -> Dict[str, Any]:
        """Create a comprehensive summary of experiment results"""
        results = DataUtils.load_json_results(results_file)
        
        if not results:
            return {}
        
        summary = {
            'total_experiments': len(results),
            'unique_models': len(set(r.get('model_name', '') for r in results.values())),
            'unique_datasets': len(set(r.get('train_dataset', '') for r in results.values())),
            'cross_dataset_experiments': len(DataUtils.filter_results_by_scenario(results, 'cross_dataset')),
            'within_dataset_experiments': len(DataUtils.filter_results_by_scenario(results, 'within_dataset')),
            'metric_summaries': {}
        }
        
        # Aggregate metrics
        metrics = ['auc', 'accuracy', 'f1', 'precision', 'recall']
        for metric in metrics:
            metric_summary = DataUtils.aggregate_metrics(results, metric)
            if metric_summary:
                summary['metric_summaries'][metric] = metric_summary
        
        return summary

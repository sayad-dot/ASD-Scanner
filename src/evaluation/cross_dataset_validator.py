import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import json
import pickle
from src.preprocessing.processed_loader import ProcessedDataLoader
from src.preprocessing.feature_harmonizer import FeatureHarmonizer
from src.models.model_factory import ModelFactory  # Changed from model_loader
from src.utils.logging_utils import get_logger
from src.evaluation.metrics_calculator import MetricsCalculator

logger = get_logger(__name__)

class CrossDatasetValidator:
    """
    Performs cross-dataset validation experiments with feature harmonization.
    Always retrains models on harmonized datasets for fair comparison.
    """
    
    def __init__(self, results_dir: str = "experiments/cross_validation"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = ProcessedDataLoader()
        self.metrics_calc = MetricsCalculator()
        self.harmonizer = FeatureHarmonizer()
        
        self.datasets = ['adult', 'adolescent', 'child']
        self.models = ['tabnet', 'random_forest', 'xgboost', 'svm']
        
        # Load all datasets and harmonize them
        raw_data = self.data_loader.load_all_datasets()
        self.all_data = self.harmonizer.harmonize_datasets(raw_data)
        
        logger.info(f"Loaded and harmonized {len(self.all_data)} datasets for cross-validation")
    
    def _get_train_test_data(self, train_dataset: str, test_dataset: str) -> Tuple[np.ndarray, ...]:
        """Get training and testing data for cross-dataset validation"""
        X_train, y_train = self.all_data[train_dataset]
        X_test, y_test = self.all_data[test_dataset]
        
        logger.info(f"Train on {train_dataset}: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"Test on {test_dataset}: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        return X_train, y_train, X_test, y_test
    
    def _create_and_train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray):
        """Create and train a fresh model on harmonized data"""
        
        logger.info(f"Creating and training fresh {model_name} model")
        
        # Create fresh model
        model = ModelFactory.create_model(model_name)
        
        # Train on harmonized data
        if model_name == 'tabnet':
            # For TabNet, use validation split
            val_split = 0.2
            split_idx = int((1 - val_split) * len(X_train))
            
            X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
            y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
            
            model.fit(X_tr, y_tr, X_val, y_val)
        else:
            # For other models, simple fit
            model.fit(X_train, y_train)
        
        logger.info(f"{model_name} training completed on harmonized data")
        return model
    
    def validate_single_combination(self, model_name: str, train_dataset: str, 
                                  test_dataset: str) -> Dict[str, Any]:
        """Validate a single model-dataset combination"""
        
        logger.info(f"Cross-validating {model_name}: {train_dataset} → {test_dataset}")
        
        try:
            # Get harmonized data
            X_train, y_train, X_test, y_test = self._get_train_test_data(train_dataset, test_dataset)
            
            # Always create and train fresh model on harmonized training data
            model = self._create_and_train_model(model_name, X_train, y_train)
            
            # Predict on test dataset
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Calculate comprehensive metrics
            metrics = self.metrics_calc.calculate_all_metrics(y_test, y_pred, y_proba)
            
            # Add experiment metadata
            result = {
                'model_name': model_name,
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'train_features': X_train.shape[1],
                'test_features': X_test.shape[1],
                'harmonized_features': X_train.shape[1],  # All should be same now
                'metrics': metrics,
                'predictions': {
                    'y_true': y_test.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_proba': y_proba[:, 1].tolist() if y_proba.ndim > 1 and y_proba.shape[1] > 1 else y_proba.ravel().tolist()
                }
            }
            
            # Save individual result
            result_file = self.results_dir / f"{model_name}_{train_dataset}_to_{test_dataset}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            logger.info(f"Cross-validation completed - AUC: {metrics['auc']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed {model_name} {train_dataset}→{test_dataset}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def validate_all_combinations(self) -> Dict[str, Dict[str, Any]]:
        """Perform cross-dataset validation for all model-dataset combinations"""
        
        logger.info("Starting comprehensive cross-dataset validation with fresh model training")
        
        results = {}
        total_combinations = len(self.models) * len(self.datasets) * len(self.datasets)
        current = 0
        
        for model_name in self.models:
            for train_dataset in self.datasets:
                for test_dataset in self.datasets:
                    current += 1
                    logger.info(f"Progress: {current}/{total_combinations}")
                    
                    result = self.validate_single_combination(
                        model_name, train_dataset, test_dataset
                    )
                    
                    if result is not None:
                        key = f"{model_name}_{train_dataset}_to_{test_dataset}"
                        results[key] = result
        
        # Save comprehensive results
        all_results_file = self.results_dir / "all_cross_validation_results.json"
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Cross-dataset validation completed: {len(results)} combinations")
        return results
    
    def get_performance_matrix(self, metric: str = 'auc') -> Dict[str, pd.DataFrame]:
        """Generate performance matrix for a specific metric"""
        
        # Load results
        results_file = self.results_dir / "all_cross_validation_results.json"
        if not results_file.exists():
            logger.error("No cross-validation results found. Run validate_all_combinations first.")
            return None
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create performance matrix for each model
        matrices = {}
        
        for model_name in self.models:
            matrix = np.zeros((len(self.datasets), len(self.datasets)))
            
            for i, train_dataset in enumerate(self.datasets):
                for j, test_dataset in enumerate(self.datasets):
                    key = f"{model_name}_{train_dataset}_to_{test_dataset}"
                    if key in results:
                        matrix[i, j] = results[key]['metrics'][metric]
                    else:
                        matrix[i, j] = np.nan
            
            # Convert to DataFrame
            df = pd.DataFrame(
                matrix, 
                index=[f"Train_{d}" for d in self.datasets],
                columns=[f"Test_{d}" for d in self.datasets]
            )
            matrices[model_name] = df
        
        return matrices

import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from src.models.model_factory import ModelFactory
from src.utils.logging_utils import get_logger
from pathlib import Path
import json

logger = get_logger(__name__)

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, results_dir: str = "models/hyperopt_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_search_space(self, model_name: str, trial):
        """Define hyperparameter search spaces"""
        if model_name == 'tabnet':
            # Only return model architecture parameters, not training parameters
            return {
                'n_d': trial.suggest_int('n_d', 8, 64),
                'n_a': trial.suggest_int('n_a', 8, 64),
                'n_steps': trial.suggest_int('n_steps', 3, 10),
                'gamma': trial.suggest_float('gamma', 1.0, 2.0),
                'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True)
                # Removed max_epochs and batch_size - these are training parameters
            }
        
        elif model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        
        elif model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
        
        elif model_name == 'svm':
            return {
                'C': trial.suggest_float('C', 0.1, 100, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
            }
    
    def _objective(self, trial, model_name: str, X_train: np.ndarray, y_train: np.ndarray):
        """Objective function for optimization"""
        try:
            # Get hyperparameters for this trial
            params = self._get_search_space(model_name, trial)
            
            # For TabNet, handle special training with separate training parameters
            if model_name == 'tabnet':
                # Get training-specific parameters separately
                training_params = {
                    'max_epochs': trial.suggest_int('max_epochs', 50, 200),
                    'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512])
                }
                
                # Create model with only architecture parameters
                model = ModelFactory.create_model(model_name, **params)
                
                # Use validation split for early stopping
                val_split = 0.2
                split_idx = int((1 - val_split) * len(X_train))
                
                X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
                y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
                
                # Pass training parameters to fit method
                model.fit(X_tr, y_tr, X_val, y_val, **training_params)
                
                # Evaluate on validation set
                metrics = model.evaluate(X_val, y_val)
                score = metrics['auc']
            
            else:
                # Create model with suggested parameters for other models
                model = ModelFactory.create_model(model_name, **params)
                
                # Use cross-validation for other models
                model.build_model(input_dim=X_train.shape[1], **params)
                scores = cross_val_score(model.model, X_train, y_train, 
                                       cv=3, scoring='roc_auc', n_jobs=-1)
                score = np.mean(scores)
            
            return score
        
        except Exception as e:
            logger.warning(f"Trial failed for {model_name}: {e}")
            return 0.0
    
    def optimize(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                n_trials: int = 50) -> dict:
        """Optimize hyperparameters for a model"""
        
        logger.info(f"Starting hyperparameter optimization for {model_name} ({n_trials} trials)")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self._objective(trial, model_name, X_train, y_train),
            n_trials=n_trials
        )
        
        best_params = study.best_params
        best_score = study.best_value
        
        # For TabNet, separate model and training parameters in results
        if model_name == 'tabnet':
            model_params = {k: v for k, v in best_params.items() 
                           if k in ['n_d', 'n_a', 'n_steps', 'gamma', 'lambda_sparse']}
            training_params = {k: v for k, v in best_params.items() 
                             if k in ['max_epochs', 'batch_size']}
            
            results = {
                'model_name': model_name,
                'best_model_params': model_params,
                'best_training_params': training_params,
                'best_params': best_params,  # Keep for backward compatibility
                'best_score': best_score,
                'n_trials': n_trials
            }
        else:
            results = {
                'model_name': model_name,
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': n_trials
            }
        
        # Save results
        results_file = self.results_dir / f"{model_name}_hyperopt.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"{model_name} optimization completed. Best AUC: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results

import numpy as np
from pathlib import Path
from typing import Dict, List
from src.models.model_factory import ModelFactory
from src.preprocessing.processed_loader import ProcessedDataLoader
from src.training.hyperopt import HyperparameterOptimizer
from src.utils.logging_utils import get_logger
import json

logger = get_logger(__name__)

class ModelTrainer:
    """Orchestrates model training and evaluation"""
    
    def __init__(self, models_dir: str = "models/saved_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_loader = ProcessedDataLoader()
        self.hyperopt = HyperparameterOptimizer()
        self.trained_models = {}
    
    def train_single_model(self, model_name: str, dataset_name: str, 
                          use_hyperopt: bool = True, hyperopt_trials: int = 30):
        """Train a single model on a dataset"""
        
        logger.info(f"Training {model_name} on {dataset_name}")
        
        # Load and split data
        splits = self.data_loader.split_dataset(dataset_name)
        X_train, y_train = splits['X_train'], splits['y_train']
        X_val, y_val = splits['X_val'], splits['y_val']
        X_test, y_test = splits['X_test'], splits['y_test']
        
        # Hyperparameter optimization
        best_params = {}
        if use_hyperopt:
            logger.info(f"Optimizing hyperparameters for {model_name}")
            hyperopt_results = self.hyperopt.optimize(
                model_name, X_train, y_train, n_trials=hyperopt_trials
            )
            best_params = hyperopt_results['best_params']
        
        # Train final model with best parameters
        if model_name == 'tabnet':
            # Separate model parameters from training parameters for TabNet
            model_params = {k: v for k, v in best_params.items() 
                           if k in ['n_d', 'n_a', 'n_steps', 'gamma', 'lambda_sparse']}
            training_params = {k: v for k, v in best_params.items() 
                              if k in ['max_epochs', 'batch_size']}
            
            # Create model with only architecture parameters
            model = ModelFactory.create_model(model_name, **model_params)
            
            # Train with training parameters
            model.fit(X_train, y_train, X_val, y_val, **training_params)
        else:
            # For other models, use all parameters for model creation
            model = ModelFactory.create_model(model_name, **best_params)
            model.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test)
        
        # Save model
        model_filename = f"{model_name}_{dataset_name}.pkl"
        if model_name == 'tabnet':
            model_filename = f"{model_name}_{dataset_name}.zip"
        
        model_path = self.models_dir / model_filename
        model.save_model(str(model_path))
        
        # Store results
        result = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'best_params': best_params,
            'test_metrics': test_metrics,
            'model_path': str(model_path)
        }
        
        key = f"{model_name}_{dataset_name}"
        self.trained_models[key] = result
        
        logger.info(f"Training completed - Test AUC: {test_metrics['auc']:.4f}")
        return result
    
    def train_all_models(self, model_names: List[str] = None, 
                        dataset_names: List[str] = None,
                        use_hyperopt: bool = True, hyperopt_trials: int = 30):
        """Train all model-dataset combinations"""
        
        if model_names is None:
            model_names = ModelFactory.get_available_models()
        
        if dataset_names is None:
            dataset_names = ['adult', 'adolescent', 'child']
        
        results = {}
        
        for model_name in model_names:
            for dataset_name in dataset_names:
                try:
                    result = self.train_single_model(
                        model_name, dataset_name, use_hyperopt, hyperopt_trials
                    )
                    results[f"{model_name}_{dataset_name}"] = result
                except Exception as e:
                    logger.error(f"Failed to train {model_name} on {dataset_name}: {e}")
        
        # Save all results
        results_file = self.models_dir.parent / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training completed for {len(results)} model-dataset combinations")
        return results

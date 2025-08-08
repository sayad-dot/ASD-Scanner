import joblib
import torch
from pathlib import Path
from typing import Any
from pytorch_tabnet.tab_model import TabNetClassifier
from src.models.model_factory import ModelFactory
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ModelLoader:
    """Load pre-trained models from Phase 2"""
    
    def __init__(self, models_dir: str = "models/saved_models"):
        self.models_dir = Path(models_dir)
        
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")
    
    def load_model(self, model_name: str, dataset_name: str):
        """Load a specific model trained on a specific dataset"""
        
        # Fix file naming - check both naming conventions
        if model_name == 'tabnet':
            # Try both naming patterns
            model_file_pattern1 = self.models_dir / f"{model_name}_{dataset_name}.zip"
            model_file_pattern2 = self.models_dir / f"{dataset_name}.zip"
            
            if model_file_pattern1.exists():
                model_file = model_file_pattern1
            elif model_file_pattern2.exists():
                model_file = model_file_pattern2
            else:
                raise FileNotFoundError(f"TabNet model file not found. Tried: {model_file_pattern1} and {model_file_pattern2}")
        else:
            model_file = self.models_dir / f"{model_name}_{dataset_name}.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        logger.info(f"Loading {model_name} model trained on {dataset_name}")
        
        if model_name == 'tabnet':
            # Load TabNet model
            model = ModelFactory.create_model('tabnet')
            model.model = TabNetClassifier()
            model.model.load_model(str(model_file))
            model.is_trained = True
        else:
            # Load sklearn models
            model = ModelFactory.create_model(model_name)
            model.model = joblib.load(model_file)
            model.is_trained = True
        
        logger.info(f"Successfully loaded {model_name} model")
        return model
    
    def list_available_models(self) -> dict:
        """List all available trained models"""
        
        available = {
            'tabnet': [],
            'random_forest': [],
            'xgboost': [],
            'svm': []
        }
        
        for model_file in self.models_dir.glob("*"):
            if model_file.is_file():
                filename = model_file.stem
                
                # Handle TabNet naming patterns
                if model_file.suffix == '.zip':
                    if filename in ['adult', 'adolescent', 'child']:
                        available['tabnet'].append(filename)
                    elif filename.startswith('tabnet_'):
                        dataset = filename.replace('tabnet_', '')
                        available['tabnet'].append(dataset)
                
                # Handle other models
                for model_type in ['random_forest', 'xgboost', 'svm']:
                    if filename.startswith(model_type):
                        dataset = filename.replace(f"{model_type}_", "")
                        available[model_type].append(dataset)
        
        return available

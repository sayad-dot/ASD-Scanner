from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any, Tuple
import joblib
from pathlib import Path

class BaseModel(ABC):
    """
    Abstract base class for all models in the ASD-Scanner project.
    Provides common interface for training, prediction, and evaluation.
    """
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.config = kwargs
        
    @abstractmethod
    def build_model(self, input_dim: int, **kwargs):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, 'save_model'):  # TabNet case
            self.model.save_model(str(save_path))
        else:  # sklearn models
            joblib.dump(self.model, save_path)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        load_path = Path(filepath)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        if self.model_name == "tabnet":
            # TabNet loading handled in TabNetModel class
            pass
        else:
            self.model = joblib.load(load_path)
        
        self.is_trained = True
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (if supported)"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'feature_importance_'):
            return self.model.feature_importance_
        else:
            return None

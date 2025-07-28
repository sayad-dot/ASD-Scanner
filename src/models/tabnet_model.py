import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from .base_model import BaseModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class TabNetModel(BaseModel):
    """
    TabNet implementation for ASD classification
    """
    
    def __init__(self, **kwargs):
        super().__init__(model_name="tabnet", **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def build_model(self, input_dim: int, **kwargs):
        """Build TabNet model"""
        default_params = {
            'n_d': 32,
            'n_a': 32,
            'n_steps': 5,
            'gamma': 1.3,
            'lambda_sparse': 1e-3,
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2),
            'mask_type': 'entmax',
            'scheduler_params': {"step_size": 10, "gamma": 0.9},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'verbose': 1,
            'device_name': str(self.device)
        }
        
        # Update with provided parameters
        params = {**default_params, **kwargs}
        
        self.model = TabNetClassifier(**params)
        logger.info(f"TabNet model built with parameters: {params}")
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs):
        """Train TabNet model"""
        
        if self.model is None:
            self.build_model(input_dim=X_train.shape[1], **self.config)
        
        train_params = {
            'max_epochs': kwargs.get('max_epochs', 100),
            'patience': kwargs.get('patience', 20),
            'batch_size': kwargs.get('batch_size', 256),
            'virtual_batch_size': kwargs.get('virtual_batch_size', 128),
            'drop_last': False,
        }
        
        # Add validation data if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            train_params['eval_metric'] = ['auc', 'accuracy']
        
        logger.info(f"Training TabNet with params: {train_params}")
        
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            **train_params
        )
        
        self.is_trained = True
        logger.info("TabNet training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get TabNet feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        return self.model.feature_importances_
    
    def explain(self, X: np.ndarray) -> tuple:
        """Get TabNet explanations (masks)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting explanations")
        return self.model.explain(X)

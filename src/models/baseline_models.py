import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from .base_model import BaseModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class RandomForestModel(BaseModel):
    """Random Forest implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="random_forest", **kwargs)
    
    def build_model(self, input_dim: int, **kwargs):
        """Build Random Forest model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        params = {**default_params, **kwargs}
        self.model = RandomForestClassifier(**params)
        logger.info(f"Random Forest built with params: {params}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs):
        """Train Random Forest"""
        if self.model is None:
            self.build_model(input_dim=X_train.shape[1], **self.config)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Random Forest training completed")
    
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

class XGBoostModel(BaseModel):
    """XGBoost implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="xgboost", **kwargs)
    
    def build_model(self, input_dim: int, **kwargs):
        """Build XGBoost model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'enable_categorical': False  # Updated for XGBoost 3.x
        }
        
        # Remove deprecated parameter for XGBoost 3.x
        if 'use_label_encoder' in kwargs:
            kwargs.pop('use_label_encoder')
        
        params = {**default_params, **kwargs}
        self.model = xgb.XGBClassifier(**params)
        logger.info(f"XGBoost built with params: {params}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs):
        """Train XGBoost - Compatible with XGBoost 3.x"""
        if self.model is None:
            self.build_model(input_dim=X_train.shape[1], **self.config)
        
        # For XGBoost 3.x, use simple approach without early stopping complications
        if X_val is not None and y_val is not None:
            # Use eval_set for monitoring but without early stopping
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                logger.info("XGBoost trained with validation monitoring")
            except Exception as e:
                logger.warning(f"Validation approach failed: {e}, using simple fit")
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        logger.info("XGBoost training completed")
    
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

class SVMModel(BaseModel):
    """SVM implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="svm", **kwargs)
    
    def build_model(self, input_dim: int, **kwargs):
        """Build SVM model"""
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        
        params = {**default_params, **kwargs}
        self.model = SVC(**params)
        logger.info(f"SVM built with params: {params}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs):
        """Train SVM"""
        if self.model is None:
            self.build_model(input_dim=X_train.shape[1], **self.config)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("SVM training completed")
    
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

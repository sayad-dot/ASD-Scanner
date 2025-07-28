from .tabnet_model import TabNetModel
from .baseline_models import RandomForestModel, XGBoostModel, SVMModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ModelFactory:
    """Factory class to create model instances"""
    
    _models = {
        'tabnet': TabNetModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'svm': SVMModel
    }
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs):
        """Create a model instance"""
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls._models.keys())}")
        
        model_class = cls._models[model_name]
        model = model_class(**kwargs)
        logger.info(f"Created {model_name} model with config: {kwargs}")
        return model
    
    @classmethod
    def get_available_models(cls):
        """Get list of available models"""
        return list(cls._models.keys())

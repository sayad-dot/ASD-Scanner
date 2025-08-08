import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any  # Added Any here
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class FeatureHarmonizer:
    """Advanced feature harmonization for cross-dataset ASD validation"""
    
    def __init__(self, feature_selection_method: str = 'intersection'):
        """
        Initialize harmonizer
        
        Args:
            feature_selection_method: 'intersection', 'union', 'top_k', or 'min_features'
        """
        self.method = feature_selection_method
        self.common_features = None
        self.feature_selector = None
        self.scaler = StandardScaler()
        
    def find_common_features_intelligent(self, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                       k_best: int = 50) -> List[int]:
        """Find best common features using statistical tests"""
        
        logger.info(f"Finding common features using {self.method} method")
        
        feature_counts = {name: X.shape[1] for name, (X, y) in datasets.items()}
        logger.info(f"Original feature counts: {feature_counts}")
        
        if self.method == 'min_features':
            # Use minimum feature count
            min_features = min(feature_counts.values())
            common_features = list(range(min_features))
            logger.info(f"Using first {min_features} features (minimum approach)")
            
        elif self.method == 'top_k':
            # Use top-k features based on statistical significance
            min_features = min(feature_counts.values())
            k = min(k_best, min_features)
            
            # Combine all datasets to find top features
            all_X = []
            all_y = []
            
            for name, (X, y) in datasets.items():
                # Take only first min_features to ensure compatibility
                X_subset = X[:, :min_features]
                all_X.append(X_subset)
                all_y.extend(y.tolist())
            
            combined_X = np.vstack(all_X)
            combined_y = np.array(all_y)
            
            # Select top k features
            selector = SelectKBest(f_classif, k=k)
            selector.fit(combined_X, combined_y)
            
            common_features = selector.get_support(indices=True).tolist()
            self.feature_selector = selector
            
            logger.info(f"Selected top {k} features based on statistical significance")
            
        else:  # Default: min_features
            min_features = min(feature_counts.values())
            common_features = list(range(min_features))
            logger.info(f"Using first {min_features} features as default")
        
        self.common_features = common_features
        return common_features
    
    def harmonize_datasets(self, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Harmonize all datasets to common feature space with scaling"""
        
        # Find common features
        if self.common_features is None:
            self.find_common_features_intelligent(datasets)
        
        harmonized_datasets = {}
        all_X_harmonized = []
        
        # First pass: extract common features
        for name, (X, y) in datasets.items():
            if self.feature_selector is not None:
                # Use feature selector
                min_features = min(X.shape[1], len(self.common_features))
                X_subset = X[:, :min_features]
                X_harmonized = self.feature_selector.transform(X_subset)
            else:
                # Use common feature indices
                max_idx = min(X.shape[1], max(self.common_features) + 1)
                available_features = [f for f in self.common_features if f < max_idx]
                X_harmonized = X[:, available_features]
            
            all_X_harmonized.append(X_harmonized)
            logger.info(f"Harmonized {name}: {X.shape[1]} → {X_harmonized.shape[1]} features")
        
        # Fit scaler on all harmonized data
        combined_X = np.vstack(all_X_harmonized)
        self.scaler.fit(combined_X)
        
        # Second pass: scale and store
        for i, (name, (X, y)) in enumerate(datasets.items()):
            X_harmonized = all_X_harmonized[i]
            X_scaled = self.scaler.transform(X_harmonized)
            harmonized_datasets[name] = (X_scaled, y)
            
            logger.info(f"Scaled {name}: mean={X_scaled.mean():.3f}, std={X_scaled.std():.3f}")
        
        return harmonized_datasets
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the harmonization process"""
        return {
            'method': self.method,
            'n_common_features': len(self.common_features) if self.common_features else 0,
            'common_features': self.common_features,
            'has_feature_selector': self.feature_selector is not None,
            'has_scaler': hasattr(self.scaler, 'mean_')
        }

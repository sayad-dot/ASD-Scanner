import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ProcessedDataLoader:
    """Load and split processed ASD datasets"""
    
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.datasets = {}
    
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single processed dataset"""
        file_path = self.processed_dir / f"{dataset_name}_processed.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {file_path}")
        
        df = pd.read_csv(file_path)
        X = df.drop(columns=['target']).values
        y = df['target'].values
        
        logger.info(f"Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def load_all_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load all processed datasets"""
        datasets = {}
        for dataset_name in ['adult', 'adolescent', 'child']:
            try:
                X, y = self.load_dataset(dataset_name)
                datasets[dataset_name] = (X, y)
            except FileNotFoundError as e:
                logger.warning(f"Could not load {dataset_name}: {e}")
        
        self.datasets = datasets
        return datasets
    
    def split_dataset(self, dataset_name: str, test_size: float = 0.2, 
                     val_size: float = 0.1, random_state: int = 42) -> Dict[str, np.ndarray]:
        """Split dataset into train/val/test"""
        if dataset_name not in self.datasets:
            X, y = self.load_dataset(dataset_name)
            self.datasets[dataset_name] = (X, y)
        else:
            X, y = self.datasets[dataset_name]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        logger.info(f"{dataset_name} split - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        return splits

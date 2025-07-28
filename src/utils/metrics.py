import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_proba: np.ndarray = None) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary'),
        'specificity': specificity_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics

def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate specificity (true negative rate)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def print_classification_summary(y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model"):
    """Print detailed classification summary"""
    print(f"\n{model_name} Classification Report:")
    print("=" * 50)
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Additional metrics
    metrics = calculate_metrics(y_true, y_pred)
    print(f"\nAdditional Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

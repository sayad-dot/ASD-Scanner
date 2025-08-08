import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from typing import Dict, Any
import warnings

class MetricsCalculator:
    """Calculate comprehensive evaluation metrics for classification tasks"""
    
    def __init__(self):
        self.metric_functions = {
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='binary', zero_division=0),
            'specificity': self._specificity_score
        }
    
    def _specificity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)"""
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except:
            return 0.0
    
    def _safe_auc_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Safely calculate AUC score with proper array handling"""
        try:
            if len(np.unique(y_true)) < 2:
                return 0.5  # Random performance when only one class present
            
            # Fix probability array shape
            if y_proba.ndim > 1:
                if y_proba.shape[1] == 2:
                    # Binary classification - take positive class probability
                    y_proba = y_proba[:, 1]
                elif y_proba.shape[1] == 1:
                    # Single column probability
                    y_proba = y_proba[:, 0]
                else:
                    # Multiple columns, take max probability
                    y_proba = np.max(y_proba, axis=1)
            
            # Ensure 1D array
            y_proba = np.asarray(y_proba).ravel()
            y_true = np.asarray(y_true).ravel()
            
            return roc_auc_score(y_true, y_proba)
        except Exception as e:
            warnings.warn(f"AUC calculation failed: {e}")
            return 0.0
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        metrics = {}
        
        # Ensure arrays are 1D
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        for metric_name, metric_func in self.metric_functions.items():
            try:
                metrics[metric_name] = float(metric_func(y_true, y_pred))
            except Exception as e:
                warnings.warn(f"Failed to calculate {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        return metrics
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_proba: np.ndarray = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        
        # Ensure arrays are 1D
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        # Basic metrics
        metrics = self.calculate_basic_metrics(y_true, y_pred)
        
        # AUC score
        if y_proba is not None:
            metrics['auc'] = self._safe_auc_score(y_true, y_proba)
        else:
            metrics['auc'] = 0.0
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Additional metrics from confusion matrix
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['true_negatives'] = int(tn)
                metrics['false_positives'] = int(fp)
                metrics['false_negatives'] = int(fn)
                metrics['true_positives'] = int(tp)
                
                # Balanced accuracy
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['balanced_accuracy'] = (sensitivity + specificity) / 2
                
        except Exception as e:
            warnings.warn(f"Confusion matrix calculation failed: {e}")
            metrics['confusion_matrix'] = [[0, 0], [0, 0]]
        
        # Classification report
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        except Exception as e:
            warnings.warn(f"Classification report failed: {e}")
            metrics['classification_report'] = {}
        
        # ROC curve data (if probabilities available)
        if y_proba is not None:
            try:
                # Fix probability array for ROC curve
                if y_proba.ndim > 1:
                    if y_proba.shape[1] == 2:
                        y_proba_roc = y_proba[:, 1]
                    else:
                        y_proba_roc = y_proba[:, 0]
                else:
                    y_proba_roc = y_proba
                
                y_proba_roc = np.asarray(y_proba_roc).ravel()
                
                fpr, tpr, thresholds = roc_curve(y_true, y_proba_roc)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            except Exception as e:
                warnings.warn(f"ROC curve calculation failed: {e}")
                metrics['roc_curve'] = {'fpr': [], 'tpr': [], 'thresholds': []}
        
        return metrics

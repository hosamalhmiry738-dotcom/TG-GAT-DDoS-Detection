"""
Metrics calculator for TG-GAT DDoS Detection System.

This module implements comprehensive evaluation metrics for DDoS detection
including accuracy, precision, recall, F1-score, and operational metrics.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, auc
)
import time

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for DDoS detection evaluation.
    
    Calculates standard classification metrics along with operational
    metrics specific to DDoS detection tasks.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.target_metrics = config.get('evaluation', {}).get('metrics', [])
        
        # Performance targets from research paper
        self.targets = {
            'accuracy': 0.998,
            'f1': 0.997,
            'fpr': 0.005,  # False Positive Rate < 0.5%
            'detection_time': 0.020  # < 20ms
        }
    
    def calculate_metrics(self, predictions: List[int], 
                         targets: List[int],
                         probabilities: Optional[List[List[float]]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: Predicted labels
            targets: Ground truth labels
            probabilities: Optional class probabilities
            
        Returns:
            Dictionary of calculated metrics
        """
        if len(predictions) == 0 or len(targets) == 0:
            logger.warning("Empty predictions or targets provided")
            return {}
        
        # Convert to numpy arrays
        y_pred = np.array(predictions)
        y_true = np.array(targets)
        
        # Basic classification metrics
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1-score (binary classification)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', pos_label=1)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', pos_label=1)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', pos_label=1)
        
        # False Positive Rate (FPR)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # True Positive Rate (TPR) / Recall
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # False Negative Rate (FNR)
        metrics['fnr'] = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # True Negative Rate (TNR) / Specificity
        metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Matthews Correlation Coefficient
        metrics['mcc'] = self._calculate_mcc(tp, tn, fp, fn)
        
        # Balanced Accuracy
        metrics['balanced_accuracy'] = (metrics['tpr'] + metrics['tnr']) / 2
        
        # ROC AUC (if probabilities are provided)
        if probabilities is not None:
            try:
                y_proba = np.array(probabilities)
                if y_proba.shape[1] == 2:
                    # Take probability of positive class
                    y_proba_positive = y_proba[:, 1]
                else:
                    y_proba_positive = y_proba[:, 0]
                
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_positive)
                
                # PR AUC
                precision, recall, _ = precision_recall_curve(y_true, y_proba_positive)
                metrics['pr_auc'] = auc(recall, precision)
                
            except Exception as e:
                logger.warning(f"Error calculating AUC metrics: {str(e)}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        # Calculate performance against targets
        metrics = self._calculate_performance_targets(metrics)
        
        return metrics
    
    def _calculate_mcc(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calculate Matthews Correlation Coefficient.
        
        Args:
            tp: True positives
            tn: True negatives
            fp: False positives
            fn: False negatives
            
        Returns:
            MCC score
        """
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            return 0.0
        return (tp * tn - fp * fn) / denominator
    
    def _calculate_performance_targets(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate performance against research targets.
        
        Args:
            metrics: Calculated metrics
            
        Returns:
            Updated metrics with target comparisons
        """
        # Calculate target achievement ratios
        for metric, target in self.targets.items():
            if metric in metrics:
                if metric == 'fpr':
                    # For FPR, lower is better
                    metrics[f'{metric}_achievement'] = min(1.0, target / metrics[metric])
                else:
                    # For other metrics, higher is better
                    metrics[f'{metric}_achievement'] = min(1.0, metrics[metric] / target)
        
        # Overall achievement score
        achievement_metrics = [f'{m}_achievement' for m in self.targets.keys()]
        metrics['overall_achievement'] = np.mean([metrics[m] for m in achievement_metrics if m in metrics])
        
        return metrics
    
    def calculate_detection_time_metrics(self, detection_times: List[float]) -> Dict[str, float]:
        """
        Calculate detection time performance metrics.
        
        Args:
            detection_times: List of detection times in seconds
            
        Returns:
            Dictionary of detection time metrics
        """
        if not detection_times:
            return {}
        
        detection_times = np.array(detection_times)
        
        metrics = {
            'avg_detection_time': np.mean(detection_times),
            'median_detection_time': np.median(detection_times),
            'min_detection_time': np.min(detection_times),
            'max_detection_time': np.max(detection_times),
            'std_detection_time': np.std(detection_times),
            'p95_detection_time': np.percentile(detection_times, 95),
            'p99_detection_time': np.percentile(detection_times, 99)
        }
        
        # Check against target (< 20ms)
        target_time = self.targets['detection_time']
        metrics['detection_time_achievement'] = min(1.0, target_time / metrics['avg_detection_time'])
        
        return metrics
    
    def calculate_resource_usage_metrics(self, cpu_usage: List[float], 
                                       memory_usage: List[float]) -> Dict[str, float]:
        """
        Calculate resource usage metrics.
        
        Args:
            cpu_usage: List of CPU usage percentages
            memory_usage: List of memory usage percentages
            
        Returns:
            Dictionary of resource usage metrics
        """
        metrics = {}
        
        if cpu_usage:
            cpu_usage = np.array(cpu_usage)
            metrics.update({
                'avg_cpu_usage': np.mean(cpu_usage),
                'max_cpu_usage': np.max(cpu_usage),
                'std_cpu_usage': np.std(cpu_usage)
            })
        
        if memory_usage:
            memory_usage = np.array(memory_usage)
            metrics.update({
                'avg_memory_usage': np.mean(memory_usage),
                'max_memory_usage': np.max(memory_usage),
                'std_memory_usage': np.std(memory_usage)
            })
        
        return metrics
    
    def calculate_zero_day_detection_metrics(self, 
                                           zero_day_predictions: List[int],
                                           zero_day_targets: List[int]) -> Dict[str, float]:
        """
        Calculate Zero-Day attack detection metrics.
        
        Args:
            zero_day_predictions: Predictions on Zero-Day samples
            zero_day_targets: Ground truth labels for Zero-Day samples
            
        Returns:
            Dictionary of Zero-Day detection metrics
        """
        if len(zero_day_predictions) == 0 or len(zero_day_targets) == 0:
            return {}
        
        # Calculate standard metrics on Zero-Day samples
        zero_day_metrics = self.calculate_metrics(zero_day_predictions, zero_day_targets)
        
        # Add Zero-Day specific metrics
        zero_day_metrics['zero_day_detection_rate'] = zero_day_metrics.get('recall', 0.0)
        
        return zero_day_metrics
    
    def generate_classification_report(self, predictions: List[int], 
                                     targets: List[int],
                                     class_names: List[str] = ['Benign', 'DDoS']) -> str:
        """
        Generate detailed classification report.
        
        Args:
            predictions: Predicted labels
            targets: Ground truth labels
            class_names: Names of classes
            
        Returns:
            Classification report string
        """
        return classification_report(
            targets, predictions, target_names=class_names, digits=4
        )
    
    def generate_confusion_matrix(self, predictions: List[int], 
                                targets: List[int]) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            predictions: Predicted labels
            targets: Ground truth labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(targets, predictions)
    
    def calculate_per_class_metrics(self, predictions: List[int], 
                                 targets: List[int],
                                 class_names: List[str] = ['Benign', 'DDoS']) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics.
        
        Args:
            predictions: Predicted labels
            targets: Ground truth labels
            class_names: Names of classes
            
        Returns:
            Dictionary of per-class metrics
        """
        y_pred = np.array(predictions)
        y_true = np.array(targets)
        
        # Calculate per-class precision, recall, F1
        precision_per_class = precision_score(y_true, y_pred, average=None, labels=[0, 1])
        recall_per_class = recall_score(y_true, y_pred, average=None, labels=[0, 1])
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1])
        
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i]
            }
        
        return per_class_metrics
    
    def calculate_adaptability_score(self, 
                                   normal_metrics: Dict[str, float],
                                   zero_day_metrics: Dict[str, float]) -> float:
        """
        Calculate adaptability score for Zero-Day detection capability.
        
        Args:
            normal_metrics: Metrics on normal attack samples
            zero_day_metrics: Metrics on Zero-Day samples
            
        Returns:
            Adaptability score (0-1)
        """
        # Weight normal performance and Zero-Day detection
        normal_weight = 0.4
        zero_day_weight = 0.6
        
        normal_score = normal_metrics.get('f1', 0.0)
        zero_day_score = zero_day_metrics.get('zero_day_detection_rate', 0.0)
        
        adaptability_score = normal_weight * normal_score + zero_day_weight * zero_day_score
        
        return adaptability_score
    
    def create_metrics_dataframe(self, metrics: Dict[str, float]) -> pd.DataFrame:
        """
        Create a pandas DataFrame from metrics dictionary.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Metrics DataFrame
        """
        # Flatten nested metrics
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_metrics[f"{key}_{sub_key}"] = sub_value
            else:
                flat_metrics[key] = value
        
        # Create DataFrame
        df = pd.DataFrame(list(flat_metrics.items()), columns=['Metric', 'Value'])
        
        # Sort by metric name
        df = df.sort_values('Metric').reset_index(drop=True)
        
        return df
    
    def log_metrics_summary(self, metrics: Dict[str, float], phase: str = "train"):
        """
        Log a summary of key metrics.
        
        Args:
            metrics: Metrics dictionary
            phase: Phase (train/val/test)
        """
        key_metrics = ['accuracy', 'f1', 'precision', 'recall', 'fpr', 'roc_auc']
        
        logger.info(f"{phase.upper()} METRICS SUMMARY:")
        logger.info("-" * 40)
        
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                target = self.targets.get(metric, None)
                
                if target is not None:
                    if metric == 'fpr':
                        status = "✓" if value <= target else "✗"
                    else:
                        status = "✓" if value >= target else "✗"
                    logger.info(f"{metric:15}: {value:.4f} (target: {target:.4f}) {status}")
                else:
                    logger.info(f"{metric:15}: {value:.4f}")
        
        logger.info("-" * 40)
        
        # Log overall achievement
        if 'overall_achievement' in metrics:
            achievement = metrics['overall_achievement']
            logger.info(f"Overall Achievement: {achievement:.4f} ({achievement*100:.1f}%)")
            logger.info("-" * 40)


class RealTimeMetrics:
    """
    Real-time metrics calculation for streaming DDoS detection.
    
    Calculates metrics in a sliding window fashion for real-time
    monitoring of detection performance.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize real-time metrics.
        
        Args:
            window_size: Size of the sliding window
        """
        self.window_size = window_size
        self.predictions_window = []
        self.targets_window = []
        self.detection_times_window = []
        
    def update(self, prediction: int, target: int, detection_time: float):
        """
        Update metrics with new prediction.
        
        Args:
            prediction: New prediction
            target: Ground truth label
            detection_time: Detection time in seconds
        """
        self.predictions_window.append(prediction)
        self.targets_window.append(target)
        self.detection_times_window.append(detection_time)
        
        # Maintain window size
        if len(self.predictions_window) > self.window_size:
            self.predictions_window.pop(0)
            self.targets_window.pop(0)
            self.detection_times_window.pop(0)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get current window metrics.
        
        Returns:
            Current metrics dictionary
        """
        if len(self.predictions_window) == 0:
            return {}
        
        calculator = MetricsCalculator({})
        
        # Calculate classification metrics
        metrics = calculator.calculate_metrics(
            self.predictions_window, 
            self.targets_window
        )
        
        # Calculate detection time metrics
        time_metrics = calculator.calculate_detection_time_metrics(
            self.detection_times_window
        )
        
        metrics.update(time_metrics)
        
        return metrics
    
    def reset(self):
        """Reset the metrics windows."""
        self.predictions_window.clear()
        self.targets_window.clear()
        self.detection_times_window.clear()

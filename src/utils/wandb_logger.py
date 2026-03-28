"""
W&B (Weights & Biases) logger for TG-GAT DDoS Detection System.

This module provides comprehensive logging integration with W&B for
experiment tracking, model monitoring, and result visualization.
"""

import wandb
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class WandbLogger:
    """
    Comprehensive W&B logger for TG-GAT experiments.
    
    Provides logging for training metrics, model parameters,
    system information, and custom visualizations.
    """
    
    def __init__(self, config: Dict, project_name: Optional[str] = None, 
                 run_name: Optional[str] = None, tags: Optional[List[str]] = None):
        """
        Initialize W&B logger.
        
        Args:
            config: Configuration dictionary
            project_name: W&B project name
            run_name: W&B run name
            tags: Optional list of tags for the run
        """
        self.config = config
        self.project_name = project_name or config.get('logging', {}).get('project_name', 'tg-gat-ddos-detection')
        self.run_name = run_name or f"tg-gat-{int(time.time())}"
        self.tags = tags or ['ddos-detection', 'tg-gat', 'deep-learning']
        
        # Initialize W&B run
        self.run = None
        self.is_initialized = False
        
        # Logging state
        self.step = 0
        self.epoch = 0
        self.best_metrics = {}
        
        # System metrics
        self.system_info = {}
        
        # Custom plots
        self.custom_plots = {}
    
    def init_run(self, reinit: bool = True):
        """
        Initialize W&B run.
        
        Args:
            reinit: Whether to reinitialize if already initialized
        """
        if self.is_initialized and not reinit:
            logger.warning("W&B run already initialized")
            return
        
        try:
            # Initialize W&B run
            self.run = wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                tags=self.tags,
                reinit=reinit
            )
            
            self.is_initialized = True
            
            # Log system information
            self._log_system_info()
            
            # Log model information
            self._log_model_info()
            
            logger.info(f"W&B run initialized: {self.run_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize W&B run: {str(e)}")
            self.is_initialized = False
    
    def _log_system_info(self):
        """Log system information."""
        import platform
        import psutil
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
        }
        
        # GPU information if available
        if torch.cuda.is_available():
            system_info.update({
                'gpu_available': True,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        else:
            system_info['gpu_available'] = False
        
        self.system_info = system_info
        
        if self.is_initialized:
            wandb.config.update(system_info)
        
        logger.info("System information logged")
    
    def _log_model_info(self):
        """Log model information."""
        model_config = self.config.get('model', {})
        
        model_info = {
            'model_name': model_config.get('name', 'TG-GAT'),
            'total_parameters': self._estimate_parameters(model_config),
            'node_dim': model_config.get('node_dim', 80),
            'edge_dim': model_config.get('edge_dim', 37),
            'hidden_dim': model_config.get('hidden_dim', 768),
            'num_heads': model_config.get('num_heads', 8),
            'num_layers': model_config.get('num_layers', 3),
            'gru_layers': model_config.get('gru_layers', 2)
        }
        
        if self.is_initialized:
            wandb.config.update(model_info)
        
        logger.info("Model information logged")
    
    def _estimate_parameters(self, model_config: Dict[str, Any]) -> int:
        """
        Estimate total model parameters.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Estimated parameter count
        """
        # Simplified parameter estimation
        node_dim = model_config.get('node_dim', 80)
        edge_dim = model_config.get('edge_dim', 37)
        hidden_dim = model_config.get('hidden_dim', 768)
        num_heads = model_config.get('num_heads', 8)
        num_layers = model_config.get('num_layers', 3)
        gru_layers = model_config.get('gru_layers', 2)
        
        # Rough estimation
        input_proj_params = (node_dim + edge_dim) * hidden_dim * 2
        attention_params = num_layers * (hidden_dim * hidden_dim * 4 * num_heads)
        transformer_params = num_layers * (hidden_dim * hidden_dim * 8)
        gru_params = gru_layers * (hidden_dim * hidden_dim * 3)
        classifier_params = hidden_dim * hidden_dim * 2
        
        total_params = input_proj_params + attention_params + transformer_params + gru_params + classifier_params
        
        return total_params
    
    def log_training_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log training metrics.
        
        Args:
            metrics: Training metrics dictionary
            step: Optional step number
        """
        if not self.is_initialized:
            return
        
        if step is not None:
            self.step = step
        
        # Add prefix to metrics
        prefixed_metrics = {f'train/{k}': v for k, v in metrics.items()}
        prefixed_metrics['step'] = self.step
        prefixed_metrics['epoch'] = self.epoch
        
        wandb.log(prefixed_metrics, step=self.step)
        
        # Update best metrics
        self._update_best_metrics('train', metrics)
    
    def log_validation_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log validation metrics.
        
        Args:
            metrics: Validation metrics dictionary
            step: Optional step number
        """
        if not self.is_initialized:
            return
        
        if step is not None:
            self.step = step
        
        # Add prefix to metrics
        prefixed_metrics = {f'val/{k}': v for k, v in metrics.items()}
        prefixed_metrics['step'] = self.step
        prefixed_metrics['epoch'] = self.epoch
        
        wandb.log(prefixed_metrics, step=self.step)
        
        # Update best metrics
        self._update_best_metrics('val', metrics)
    
    def log_test_metrics(self, metrics: Dict[str, float]):
        """
        Log test metrics.
        
        Args:
            metrics: Test metrics dictionary
        """
        if not self.is_initialized:
            return
        
        # Add prefix to metrics
        prefixed_metrics = {f'test/{k}': v for k, v in metrics.items()}
        
        wandb.log(prefixed_metrics)
        
        logger.info("Test metrics logged to W&B")
    
    def log_learning_rate(self, learning_rate: float, step: Optional[int] = None):
        """
        Log learning rate.
        
        Args:
            learning_rate: Current learning rate
            step: Optional step number
        """
        if not self.is_initialized:
            return
        
        if step is not None:
            self.step = step
        
        wandb.log({'train/learning_rate': learning_rate}, step=self.step)
    
    def log_loss_components(self, losses: Dict[str, float], step: Optional[int] = None):
        """
        Log individual loss components.
        
        Args:
            losses: Loss components dictionary
            step: Optional step number
        """
        if not self.is_initialized:
            return
        
        if step is not None:
            self.step = step
        
        prefixed_losses = {f'loss/{k}': v for k, v in losses.items()}
        prefixed_losses['step'] = self.step
        
        wandb.log(prefixed_losses, step=self.step)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: List[str] = None, step: Optional[int] = None):
        """
        Log confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names
            step: Optional step number
        """
        if not self.is_initialized:
            return
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if class_names is None:
            class_names = ['Benign', 'DDoS']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        # Log to W&B
        if step is not None:
            self.step = step
        
        wandb.log({
            'confusion_matrix': wandb.Image(fig),
            'step': self.step
        })
        
        plt.close(fig)
    
    def log_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                     step: Optional[int] = None):
        """
        Log ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            step: Optional step number
        """
        if not self.is_initialized:
            return
        
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Create ROC curve plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        
        # Log to W&B
        if step is not None:
            self.step = step
        
        wandb.log({
            'roc_curve': wandb.Image(fig),
            'roc_auc': roc_auc,
            'step': self.step
        })
        
        plt.close(fig)
    
    def log_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                 step: Optional[int] = None):
        """
        Log Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            step: Optional step number
        """
        if not self.is_initialized:
            return
        
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        # Create PR curve plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, color='blue', lw=2,
               label=f'PR curve (AP = {avg_precision:.2f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        
        # Log to W&B
        if step is not None:
            self.step = step
        
        wandb.log({
            'pr_curve': wandb.Image(fig),
            'avg_precision': avg_precision,
            'step': self.step
        })
        
        plt.close(fig)
    
    def log_feature_importance(self, feature_names: List[str], 
                             importance_scores: np.ndarray,
                             step: Optional[int] = None):
        """
        Log feature importance.
        
        Args:
            feature_names: List of feature names
            importance_scores: Feature importance scores
            step: Optional step number
        """
        if not self.is_initialized:
            return
        
        # Create feature importance plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by importance
        indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_scores = importance_scores[indices]
        
        # Create bar plot
        bars = ax.bar(range(len(sorted_features)), sorted_scores)
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance Score')
        ax.set_title('Feature Importance')
        ax.set_xticks(range(len(sorted_features)))
        ax.set_xticklabels(sorted_features, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, sorted_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Log to W&B
        if step is not None:
            self.step = step
        
        wandb.log({
            'feature_importance': wandb.Image(fig),
            'step': self.step
        })
        
        # Also log as table
        feature_importance_table = wandb.Table(
            columns=['feature', 'importance'],
            data=[[name, score] for name, score in zip(sorted_features, sorted_scores)]
        )
        wandb.log({'feature_importance_table': feature_importance_table})
        
        plt.close(fig)
    
    def log_training_curves(self, metrics_history: Dict[str, List[float]]):
        """
        Log training curves.
        
        Args:
            metrics_history: Dictionary of metric histories
        """
        if not self.is_initialized:
            return
        
        # Create training curves plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot main metrics
        main_metrics = ['accuracy', 'f1', 'precision', 'recall']
        
        for i, metric in enumerate(main_metrics):
            if metric in metrics_history:
                axes[i].plot(metrics_history[metric], label=metric)
                axes[i].set_title(f'{metric.capitalize()} over time')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True)
        
        plt.tight_layout()
        
        wandb.log({'training_curves': wandb.Image(fig)})
        plt.close(fig)
    
    def log_model_checkpoint(self, model_path: str, metrics: Dict[str, float]):
        """
        Log model checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            metrics: Current metrics
        """
        if not self.is_initialized:
            return
        
        # Log model as artifact
        artifact = wandb.Artifact('model_checkpoint', type='model')
        artifact.add_file(model_path)
        
        # Log metrics with artifact
        wandb.log_artifact(artifact, aliases=['latest', f'epoch_{self.epoch}'])
        
        # Log metrics
        prefixed_metrics = {f'checkpoint/{k}': v for k, v in metrics.items()}
        wandb.log(prefixed_metrics)
        
        logger.info(f"Model checkpoint logged: {model_path}")
    
    def log_custom_plot(self, plot_name: str, figure: plt.Figure, 
                       step: Optional[int] = None):
        """
        Log custom plot.
        
        Args:
            plot_name: Name of the plot
            figure: Matplotlib figure
            step: Optional step number
        """
        if not self.is_initialized:
            return
        
        if step is not None:
            self.step = step
        
        wandb.log({plot_name: wandb.Image(figure)}, step=self.step)
    
    def log_table(self, table_name: str, data: List[List[Any]], 
                  columns: List[str]):
        """
        Log table data.
        
        Args:
            table_name: Name of the table
            data: Table data
            columns: Column names
        """
        if not self.is_initialized:
            return
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({table_name: table})
    
    def log_hyperparameter_sweep_results(self, sweep_results: List[Dict[str, Any]]):
        """
        Log hyperparameter sweep results.
        
        Args:
            sweep_results: List of sweep result dictionaries
        """
        if not self.is_initialized:
            return
        
        # Create sweep table
        columns = ['run_name', 'accuracy', 'f1', 'precision', 'recall', 'fpr']
        data = []
        
        for result in sweep_results:
            row = [
                result.get('run_name', 'unknown'),
                result.get('accuracy', 0.0),
                result.get('f1', 0.0),
                result.get('precision', 0.0),
                result.get('recall', 0.0),
                result.get('fpr', 0.0)
            ]
            data.append(row)
        
        self.log_table('hyperparameter_sweep', data, columns)
        
        # Log best configuration
        if sweep_results:
            best_result = max(sweep_results, key=lambda x: x.get('f1', 0.0))
            wandb.log({'best_sweep_config': best_result})
    
    def set_epoch(self, epoch: int):
        """
        Set current epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch
    
    def increment_step(self):
        """Increment step counter."""
        self.step += 1
    
    def finish_run(self):
        """Finish W&B run."""
        if self.is_initialized and self.run:
            # Log final summary
            if self.best_metrics:
                wandb.log({'best_metrics': self.best_metrics})
            
            self.run.finish()
            self.is_initialized = False
            
            logger.info("W&B run finished")
    
    def _update_best_metrics(self, phase: str, metrics: Dict[str, float]):
        """
        Update best metrics.
        
        Args:
            phase: Phase (train/val/test)
            metrics: Current metrics
        """
        for metric_name, value in metrics.items():
            key = f'{phase}_{metric_name}'
            
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            else:
                # For metrics where higher is better
                if metric_name in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
                    self.best_metrics[key] = max(self.best_metrics[key], value)
                # For metrics where lower is better
                elif metric_name in ['fpr', 'loss', 'fnr']:
                    self.best_metrics[key] = min(self.best_metrics[key], value)
    
    def create_summary_report(self) -> str:
        """
        Create summary report of the run.
        
        Returns:
            Summary report as string
        """
        if not self.is_initialized:
            return "No W&B run initialized"
        
        report = []
        report.append(f"# W&B Run Summary")
        report.append(f"Project: {self.project_name}")
        report.append(f"Run: {self.run_name}")
        report.append(f"URL: {self.run.url}")
        report.append("")
        
        # System information
        report.append("## System Information")
        for key, value in self.system_info.items():
            report.append(f"- {key}: {value}")
        report.append("")
        
        # Best metrics
        report.append("## Best Metrics")
        for key, value in self.best_metrics.items():
            report.append(f"- {key}: {value:.4f}")
        report.append("")
        
        return "\n".join(report)
    
    def export_run_data(self, export_path: str):
        """
        Export run data to file.
        
        Args:
            export_path: Path to export run data
        """
        if not self.is_initialized:
            logger.warning("No W&B run to export")
            return
        
        export_data = {
            'run_name': self.run_name,
            'project_name': self.project_name,
            'config': self.config,
            'system_info': self.system_info,
            'best_metrics': self.best_metrics,
            'url': self.run.url
        }
        
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Run data exported to {export_path}")


# Global W&B logger instance
_global_logger = None


def get_wandb_logger(config: Dict, **kwargs) -> WandbLogger:
    """
    Get or create global W&B logger instance.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional arguments for WandbLogger
        
    Returns:
        WandbLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = WandbLogger(config, **kwargs)
    
    return _global_logger


def log_experiment_start(config: Dict, **kwargs):
    """
    Log experiment start.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional arguments
    """
    logger = get_wandb_logger(config, **kwargs)
    logger.init_run()


def log_experiment_end():
    """Log experiment end."""
    global _global_logger
    
    if _global_logger is not None:
        _global_logger.finish_run()
        _global_logger = None

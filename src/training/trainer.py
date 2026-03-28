"""
Trainer module for TG-GAT DDoS Detection System.

This module implements the complete training pipeline including:
- Model training and validation
- Experiment tracking with W&B
- Checkpoint management
- Mixed precision training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import time
import json
import wandb
from tqdm import tqdm

from ..models.tg_gat import TGGATModel
from ..data.graph_builder import GraphBuilder
from .losses import FocalLoss, TGATLoss
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Comprehensive trainer for TG-GAT DDoS detection model.
    
    Handles the complete training pipeline including data loading,
    model training, validation, checkpointing, and experiment tracking.
    """
    
    def __init__(self, config: Dict, model: Optional[TGGATModel] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            model: Optional pre-initialized model
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = model or TGGATModel(config).to(self.device)
        
        # Training parameters
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.mixed_precision = config['training']['mixed_precision']
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(config)
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.training_history = {
            'train_loss': [],
            'train_metrics': [],
            'val_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Checkpoint management
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # W&B integration
        self.use_wandb = config['logging']['use_wandb']
        if self.use_wandb:
            self._init_wandb()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.config['training']['optimizer'].lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_config = self.config['loss']
        loss_type = loss_config['type'].lower()
        
        if loss_type == 'focal':
            return FocalLoss(
                alpha=loss_config['alpha'],
                gamma=loss_config['gamma']
            )
        elif loss_type == 'tgat':
            return TGATLoss(config=self.config)
        elif loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_name = self.config['training'].get('scheduler', '').lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.epochs // 3, gamma=0.1
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        else:
            return None
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config['logging']['project_name'],
            config=self.config,
            name=f"tg-gat-{int(time.time())}"
        )
        
        # Log model information
        model_info = self.model.get_model_info()
        wandb.config.update(model_info)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        batch_times = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # Move data to device
            if isinstance(batch_data, dict):
                graph_batch = batch_data['graph'].to(self.device)
                targets = batch_data['targets'].to(self.device)
                temporal_sequence = batch_data.get('temporal_sequence', None)
            else:
                graph_batch = batch_data.to(self.device)
                targets = graph_batch.y.to(self.device)
                temporal_sequence = None
            
            # Forward pass
            with autocast(enabled=self.mixed_precision):
                outputs = self.model(graph_batch, temporal_sequence)
                loss = self.model.compute_loss(outputs, targets, self.criterion)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Get predictions
            with torch.no_grad():
                predictions = torch.argmax(outputs['probabilities'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            
            # Update temporal buffer
            self.model.update_temporal_buffer(graph_batch, outputs)
            
            # Track batch time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'batch_time': f"{batch_time:.3f}s"
            })
            
            # Log to W&B periodically
            if self.use_wandb and batch_idx % self.config['logging']['log_frequency'] == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_time': batch_time,
                    'epoch': self.current_epoch,
                    'batch': batch_idx
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        metrics['avg_batch_time'] = np.mean(batch_times)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        detection_times = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch_data in progress_bar:
                # Move data to device
                if isinstance(batch_data, dict):
                    graph_batch = batch_data['graph'].to(self.device)
                    targets = batch_data['targets'].to(self.device)
                    temporal_sequence = batch_data.get('temporal_sequence', None)
                else:
                    graph_batch = batch_data.to(self.device)
                    targets = graph_batch.y.to(self.device)
                    temporal_sequence = None
                
                # Measure detection time
                start_time = time.time()
                
                # Forward pass
                outputs = self.model(graph_batch, temporal_sequence)
                loss = self.model.compute_loss(outputs, targets, self.criterion)
                
                detection_time = time.time() - start_time
                detection_times.append(detection_time)
                
                # Update metrics
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs['probabilities'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        metrics['avg_detection_time'] = np.mean(detection_times)
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {self.model.get_model_info()}")
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation
            if val_loader is not None:
                val_loss, val_metrics = self.validate_epoch(val_loader)
            else:
                val_loss, val_metrics = 0.0, {}
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('f1', 0.0))
                else:
                    self.scheduler.step()
            
            # Store training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_metrics'].append(val_metrics)
            self.training_history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log epoch results
            self._log_epoch_results(train_loss, train_metrics, val_loss, val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging']['checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Check for best model
            if val_metrics.get('f1', 0.0) > self.best_val_f1:
                self.best_val_f1 = val_metrics.get('f1', 0.0)
                self.save_checkpoint(epoch, is_best=True)
        
        logger.info("Training completed")
        
        # Return training results
        return {
            'training_history': self.training_history,
            'best_val_f1': self.best_val_f1,
            'model_info': self.model.get_model_info()
        }
    
    def _log_epoch_results(self, train_loss: float, train_metrics: Dict[str, float],
                          val_loss: float, val_metrics: Dict[str, float]):
        """Log epoch results to console and W&B."""
        # Console logging
        logger.info(f"Epoch {self.current_epoch + 1}/{self.epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        for metric, value in train_metrics.items():
            logger.info(f"Train {metric}: {value:.4f}")
        
        if val_metrics:
            logger.info(f"Val Loss: {val_loss:.4f}")
            for metric, value in val_metrics.items():
                logger.info(f"Val {metric}: {value:.4f}")
        
        logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        logger.info("-" * 50)
        
        # W&B logging
        if self.use_wandb:
            log_dict = {
                'epoch': self.current_epoch + 1,
                'train/loss': train_loss,
                'train/learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            # Add training metrics
            for metric, value in train_metrics.items():
                log_dict[f'train/{metric}'] = value
            
            # Add validation metrics
            if val_metrics:
                log_dict['val/loss'] = val_loss
                for metric, value in val_metrics.items():
                    log_dict[f'val/{metric}'] = value
            
            wandb.log(log_dict)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_f1': self.best_val_f1,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")
        
        # Save latest model
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_f1 = checkpoint['best_val_f1']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation results
        """
        logger.info("Starting model evaluation")
        
        # Load best model
        best_checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        if best_checkpoint_path.exists():
            self.load_checkpoint(str(best_checkpoint_path))
        
        # Evaluate
        test_loss, test_metrics = self.validate_epoch(test_loader)
        
        results = {
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'model_info': self.model.get_model_info()
        }
        
        logger.info("Evaluation completed")
        logger.info(f"Test Loss: {test_loss:.4f}")
        
        for metric, value in test_metrics.items():
            logger.info(f"Test {metric}: {value:.4f}")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'test/loss': test_loss,
                **{f'test/{k}': v for k, v in test_metrics.items()}
            })
        
        return results

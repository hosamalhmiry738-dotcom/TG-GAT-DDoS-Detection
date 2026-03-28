"""
Model testing module for TG-GAT DDoS Detection System.

This module implements comprehensive testing and benchmarking of the TG-GAT model
including performance evaluation, robustness testing, and comparison with baselines.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..models.tg_gat import TGGATModel
from ..training.metrics import MetricsCalculator, RealTimeMetrics
from ..data.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class ModelTester:
    """
    Comprehensive model tester for TG-GAT DDoS detection.
    
    Provides thorough evaluation including standard metrics,
    real-time performance, robustness testing, and baseline comparisons.
    """
    
    def __init__(self, config: Dict, model_path: str):
        """
        Initialize the model tester.
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained model checkpoint
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(config)
        
        # Initialize real-time metrics
        self.real_time_metrics = RealTimeMetrics(window_size=1000)
        
        # Results storage
        self.test_results = {}
        self.baseline_results = {}
        
        # Output directory
        self.output_dir = Path(config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self, model_path: str) -> TGGATModel:
        """
        Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = TGGATModel(self.config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model info: {model.get_model_info()}")
        
        return model
    
    def evaluate_standard_metrics(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate standard classification metrics.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of standard metrics
        """
        logger.info("Evaluating standard metrics")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        detection_times = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Standard Evaluation"):
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
                
                detection_time = time.time() - start_time
                
                # Get predictions and probabilities
                probabilities = outputs['probabilities']
                predictions = torch.argmax(probabilities, dim=-1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                detection_times.append(detection_time)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            all_predictions, all_targets, all_probabilities
        )
        
        # Add detection time metrics
        time_metrics = self.metrics_calculator.calculate_detection_time_metrics(detection_times)
        metrics.update(time_metrics)
        
        # Generate detailed reports
        classification_report = self.metrics_calculator.generate_classification_report(
            all_predictions, all_targets
        )
        confusion_matrix = self.metrics_calculator.generate_confusion_matrix(
            all_predictions, all_targets
        )
        
        results = {
            'metrics': metrics,
            'classification_report': classification_report,
            'confusion_matrix': confusion_matrix.tolist(),
            'detection_times': detection_times,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        # Log metrics summary
        self.metrics_calculator.log_metrics_summary(metrics, "test")
        
        return results
    
    def evaluate_real_time_performance(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate real-time performance metrics.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of real-time performance results
        """
        logger.info("Evaluating real-time performance")
        
        # Reset real-time metrics
        self.real_time_metrics.reset()
        
        real_time_results = {
            'window_metrics': [],
            'throughput': [],
            'latency_distribution': []
        }
        
        batch_times = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Real-time Evaluation")):
                # Move data to device
                if isinstance(batch_data, dict):
                    graph_batch = batch_data['graph'].to(self.device)
                    targets = batch_data['targets'].to(self.device)
                    temporal_sequence = batch_data.get('temporal_sequence', None)
                else:
                    graph_batch = batch_data.to(self.device)
                    targets = graph_batch.y.to(self.device)
                    temporal_sequence = None
                
                # Measure batch processing time
                batch_start_time = time.time()
                
                # Forward pass
                outputs = self.model(graph_batch, temporal_sequence)
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                # Get predictions
                predictions = torch.argmax(outputs['probabilities'], dim=-1)
                
                # Update real-time metrics for each sample
                for i in range(len(predictions)):
                    self.real_time_metrics.update(
                        predictions[i].item(),
                        targets[i].item(),
                        batch_time / len(predictions)  # Average per sample
                    )
                
                # Get window metrics periodically
                if batch_idx % 10 == 0:
                    window_metrics = self.real_time_metrics.get_current_metrics()
                    real_time_results['window_metrics'].append({
                        'batch': batch_idx,
                        'metrics': window_metrics
                    })
                
                # Calculate throughput (samples per second)
                throughput = len(predictions) / batch_time
                real_time_results['throughput'].append(throughput)
        
        # Final metrics
        final_metrics = self.real_time_metrics.get_current_metrics()
        real_time_results['final_metrics'] = final_metrics
        
        # Performance statistics
        real_time_results['performance_stats'] = {
            'avg_batch_time': np.mean(batch_times),
            'avg_throughput': np.mean(real_time_results['throughput']),
            'max_throughput': np.max(real_time_results['throughput']),
            'min_throughput': np.min(real_time_results['throughput'])
        }
        
        logger.info(f"Average throughput: {real_time_results['performance_stats']['avg_throughput']:.2f} samples/sec")
        logger.info(f"Average batch time: {real_time_results['performance_stats']['avg_batch_time']:.4f}s")
        
        return real_time_results
    
    def evaluate_robustness(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model robustness under various conditions.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of robustness evaluation results
        """
        logger.info("Evaluating model robustness")
        
        robustness_results = {
            'noise_robustness': {},
            'adversarial_robustness': {},
            'missing_data_robustness': {}
        }
        
        # Test noise robustness
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        for noise_level in noise_levels:
            logger.info(f"Testing noise robustness with level {noise_level}")
            
            metrics = self._test_noise_robustness(test_loader, noise_level)
            robustness_results['noise_robustness'][noise_level] = metrics
        
        # Test missing data robustness
        missing_rates = [0.05, 0.1, 0.2, 0.3]
        for missing_rate in missing_rates:
            logger.info(f"Testing missing data robustness with rate {missing_rate}")
            
            metrics = self._test_missing_data_robustness(test_loader, missing_rate)
            robustness_results['missing_data_robustness'][missing_rate] = metrics
        
        return robustness_results
    
    def _test_noise_robustness(self, test_loader: DataLoader, noise_level: float) -> Dict[str, float]:
        """
        Test model robustness to input noise.
        
        Args:
            test_loader: Test data loader
            noise_level: Level of Gaussian noise to add
            
        Returns:
            Metrics under noise conditions
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc=f"Noise {noise_level}"):
                # Move data to device
                if isinstance(batch_data, dict):
                    graph_batch = batch_data['graph'].to(self.device)
                    targets = batch_data['targets'].to(self.device)
                    temporal_sequence = batch_data.get('temporal_sequence', None)
                else:
                    graph_batch = batch_data.to(self.device)
                    targets = graph_batch.y.to(self.device)
                    temporal_sequence = None
                
                # Add noise to node features
                if hasattr(graph_batch, 'x') and graph_batch.x is not None:
                    noise = torch.randn_like(graph_batch.x) * noise_level
                    graph_batch.x = graph_batch.x + noise
                
                # Forward pass
                outputs = self.model(graph_batch, temporal_sequence)
                predictions = torch.argmax(outputs['probabilities'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        return self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
    
    def _test_missing_data_robustness(self, test_loader: DataLoader, missing_rate: float) -> Dict[str, float]:
        """
        Test model robustness to missing data.
        
        Args:
            test_loader: Test data loader
            missing_rate: Rate of features to randomly set to zero
            
        Returns:
            Metrics under missing data conditions
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc=f"Missing {missing_rate}"):
                # Move data to device
                if isinstance(batch_data, dict):
                    graph_batch = batch_data['graph'].to(self.device)
                    targets = batch_data['targets'].to(self.device)
                    temporal_sequence = batch_data.get('temporal_sequence', None)
                else:
                    graph_batch = batch_data.to(self.device)
                    targets = graph_batch.y.to(self.device)
                    temporal_sequence = None
                
                # Randomly set features to zero
                if hasattr(graph_batch, 'x') and graph_batch.x is not None:
                    mask = torch.rand_like(graph_batch.x) > missing_rate
                    graph_batch.x = graph_batch.x * mask
                
                # Forward pass
                outputs = self.model(graph_batch, temporal_sequence)
                predictions = torch.argmax(outputs['probabilities'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        return self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
    
    def evaluate_zero_day_detection(self, zero_day_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate Zero-Day attack detection capability.
        
        Args:
            zero_day_loader: Zero-Day attack data loader
            
        Returns:
            Zero-Day detection results
        """
        logger.info("Evaluating Zero-Day attack detection")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_data in tqdm(zero_day_loader, desc="Zero-Day Evaluation"):
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
                outputs = self.model(graph_batch, temporal_sequence)
                predictions = torch.argmax(outputs['probabilities'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(outputs['probabilities'].cpu().numpy())
        
        # Calculate Zero-Day specific metrics
        zero_day_metrics = self.metrics_calculator.calculate_zero_day_detection_metrics(
            all_predictions, all_targets
        )
        
        results = {
            'zero_day_metrics': zero_day_metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        logger.info(f"Zero-Day detection rate: {zero_day_metrics.get('zero_day_detection_rate', 0):.4f}")
        
        return results
    
    def compare_with_baselines(self, test_loader: DataLoader, 
                            baseline_models: Dict[str, nn.Module]) -> Dict[str, Any]:
        """
        Compare TG-GAT performance with baseline models.
        
        Args:
            test_loader: Test data loader
            baseline_models: Dictionary of baseline models
            
        Returns:
            Comparison results
        """
        logger.info("Comparing with baseline models")
        
        comparison_results = {}
        
        # Evaluate TG-GAT
        tg_gat_results = self.evaluate_standard_metrics(test_loader)
        comparison_results['TG-GAT'] = tg_gat_results['metrics']
        
        # Evaluate baselines
        for model_name, model in baseline_models.items():
            logger.info(f"Evaluating baseline: {model_name}")
            
            # Temporarily replace model
            original_model = self.model
            self.model = model
            self.model.eval()
            
            try:
                baseline_results = self.evaluate_standard_metrics(test_loader)
                comparison_results[model_name] = baseline_results['metrics']
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                comparison_results[model_name] = {}
            
            # Restore original model
            self.model = original_model
        
        # Create comparison table
        comparison_df = self._create_comparison_table(comparison_results)
        comparison_results['comparison_table'] = comparison_df
        
        return comparison_results
    
    def _create_comparison_table(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Create comparison table of model performances.
        
        Args:
            results: Dictionary of model results
            
        Returns:
            Comparison DataFrame
        """
        # Extract key metrics
        key_metrics = ['accuracy', 'f1', 'precision', 'recall', 'fpr', 'roc_auc']
        
        table_data = []
        for model_name, metrics in results.items():
            row = {'Model': model_name}
            for metric in key_metrics:
                row[metric] = metrics.get(metric, 0.0)
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        df = df.sort_values('f1', ascending=False).reset_index(drop=True)
        
        return df
    
    def generate_evaluation_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Evaluation report as string
        """
        report = []
        report.append("# TG-GAT DDoS Detection Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Model information
        model_info = self.model.get_model_info()
        report.append("## Model Information")
        report.append(f"- Total Parameters: {model_info['total_parameters']:,}")
        report.append(f"- Hidden Dimension: {model_info['hidden_dim']}")
        report.append(f"- Number of Heads: {model_info['num_heads']}")
        report.append(f"- Number of Layers: {model_info['num_layers']}")
        report.append("")
        
        # Standard metrics
        if 'standard' in self.test_results:
            metrics = self.test_results['standard']['metrics']
            report.append("## Standard Metrics")
            report.append(f"- Accuracy: {metrics.get('accuracy', 0):.4f}")
            report.append(f"- F1-Score: {metrics.get('f1', 0):.4f}")
            report.append(f"- Precision: {metrics.get('precision', 0):.4f}")
            report.append(f"- Recall: {metrics.get('recall', 0):.4f}")
            report.append(f"- False Positive Rate: {metrics.get('fpr', 0):.4f}")
            report.append(f"- ROC AUC: {metrics.get('roc_auc', 0):.4f}")
            report.append("")
        
        # Real-time performance
        if 'real_time' in self.test_results:
            rt_results = self.test_results['real_time']
            final_metrics = rt_results['final_metrics']
            perf_stats = rt_results['performance_stats']
            
            report.append("## Real-time Performance")
            report.append(f"- Average Detection Time: {final_metrics.get('avg_detection_time', 0):.4f}s")
            report.append(f"- Average Throughput: {perf_stats.get('avg_throughput', 0):.2f} samples/sec")
            report.append(f"- Max Throughput: {perf_stats.get('max_throughput', 0):.2f} samples/sec")
            report.append("")
        
        # Zero-Day detection
        if 'zero_day' in self.test_results:
            zd_results = self.test_results['zero_day']
            zd_metrics = zd_results['zero_day_metrics']
            
            report.append("## Zero-Day Attack Detection")
            report.append(f"- Zero-Day Detection Rate: {zd_metrics.get('zero_day_detection_rate', 0):.4f}")
            report.append(f"- Zero-Day F1-Score: {zd_metrics.get('f1', 0):.4f}")
            report.append("")
        
        # Baseline comparison
        if 'baseline' in self.test_results:
            comparison_df = self.test_results['baseline']['comparison_table']
            
            report.append("## Baseline Comparison")
            report.append(comparison_df.to_string(index=False))
            report.append("")
        
        # Performance targets
        report.append("## Performance Targets Achievement")
        targets = self.metrics_calculator.targets
        
        if 'standard' in self.test_results:
            metrics = self.test_results['standard']['metrics']
            for metric, target in targets.items():
                if metric in metrics:
                    value = metrics[metric]
                    if metric == 'fpr':
                        achieved = value <= target
                    else:
                        achieved = value >= target
                    status = "✓" if achieved else "✗"
                    report.append(f"- {metric}: {value:.4f} (target: {target:.4f}) {status}")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text
    
    def save_results(self, output_dir: Optional[str] = None):
        """
        Save all evaluation results.
        
        Args:
            output_dir: Optional output directory
        """
        save_dir = Path(output_dir) if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_path = save_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Save evaluation report
        report_path = save_dir / 'evaluation_report.md'
        self.generate_evaluation_report(str(report_path))
        
        # Save comparison table if available
        if 'baseline' in self.test_results:
            comparison_df = self.test_results['baseline']['comparison_table']
            comparison_df.to_csv(save_dir / 'baseline_comparison.csv', index=False)
        
        logger.info(f"Evaluation results saved to {save_dir}")
    
    def run_full_evaluation(self, test_loader: DataLoader, 
                           zero_day_loader: Optional[DataLoader] = None,
                           baseline_models: Optional[Dict[str, nn.Module]] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        
        Args:
            test_loader: Test data loader
            zero_day_loader: Optional Zero-Day attack loader
            baseline_models: Optional baseline models for comparison
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting comprehensive model evaluation")
        
        # Standard evaluation
        self.test_results['standard'] = self.evaluate_standard_metrics(test_loader)
        
        # Real-time evaluation
        self.test_results['real_time'] = self.evaluate_real_time_performance(test_loader)
        
        # Robustness evaluation
        self.test_results['robustness'] = self.evaluate_robustness(test_loader)
        
        # Zero-Day evaluation
        if zero_day_loader is not None:
            self.test_results['zero_day'] = self.evaluate_zero_day_detection(zero_day_loader)
        
        # Baseline comparison
        if baseline_models is not None:
            self.test_results['baseline'] = self.compare_with_baselines(test_loader, baseline_models)
        
        # Save results
        self.save_results()
        
        logger.info("Comprehensive evaluation completed")
        
        return self.test_results

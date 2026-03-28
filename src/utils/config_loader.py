"""
Configuration loader for TG-GAT DDoS Detection System.

This module handles loading, validation, and management of configuration
files for the TG-GAT model and training pipeline.
"""

import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration data class."""
    name: str = "TG-GAT"
    node_dim: int = 80
    edge_dim: int = 37
    hidden_dim: int = 768
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    gru_layers: int = 2


@dataclass
class TrainingConfig:
    """Training configuration data class."""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    gradient_clip: float = 1.0
    mixed_precision: bool = True


@dataclass
class LossConfig:
    """Loss function configuration data class."""
    type: str = "focal"
    alpha: float = 1.0
    gamma: float = 2.0
    label_smoothing: float = 0.1


@dataclass
class DataConfig:
    """Data configuration data class."""
    graph_window_ms: int = 100
    max_nodes_per_graph: int = 1000
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    normalize_features: bool = True
    handle_imbalance: bool = True


@dataclass
class GANConfig:
    """GAN configuration data class."""
    latent_dim: int = 100
    generator_layers: int = 4
    discriminator_layers: int = 3
    gan_lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    n_critic: int = 5


@dataclass
class EvaluationConfig:
    """Evaluation configuration data class."""
    metrics: list = None
    save_predictions: bool = True
    generate_explanations: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "fpr", "detection_time"]


@dataclass
class XAIConfig:
    """XAI configuration data class."""
    explainer: str = "gnn_explainer"
    num_samples: int = 100
    visualization: bool = True


@dataclass
class HardwareConfig:
    """Hardware configuration data class."""
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration data class."""
    use_wandb: bool = True
    project_name: str = "tg-gat-ddos-detection"
    log_frequency: int = 10
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5


@dataclass
class PathsConfig:
    """Paths configuration data class."""
    data_dir: str = "datasets"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


class ConfigLoader:
    """
    Configuration loader and validator for TG-GAT system.
    
    Handles loading configuration from YAML/JSON files, validation,
    and conversion to appropriate data structures.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.config = None
        
        # Initialize default configurations
        self.default_config = self._create_default_config()
        
        if config_path:
            self.load_config(config_path)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'model': asdict(ModelConfig()),
            'training': asdict(TrainingConfig()),
            'loss': asdict(LossConfig()),
            'data': asdict(DataConfig()),
            'gan': asdict(GANConfig()),
            'evaluation': asdict(EvaluationConfig()),
            'xai': asdict(XAIConfig()),
            'hardware': asdict(HardwareConfig()),
            'logging': asdict(LoggingConfig()),
            'paths': asdict(PathsConfig())
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    loaded_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Merge with default config
            self.config = self._merge_configs(self.default_config, loaded_config)
            
            # Validate configuration
            self._validate_config()
            
            logger.info(f"Configuration loaded from {config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge loaded configuration with default configuration.
        
        Args:
            default: Default configuration
            loaded: Loaded configuration
            
        Returns:
            Merged configuration
        """
        merged = default.copy()
        
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _validate_config(self):
        """Validate the current configuration."""
        if not self.config:
            raise ValueError("No configuration loaded")
        
        # Validate model configuration
        model_config = self.config.get('model', {})
        self._validate_model_config(model_config)
        
        # Validate training configuration
        training_config = self.config.get('training', {})
        self._validate_training_config(training_config)
        
        # Validate data configuration
        data_config = self.config.get('data', {})
        self._validate_data_config(data_config)
        
        # Validate paths
        paths_config = self.config.get('paths', {})
        self._validate_paths_config(paths_config)
        
        logger.info("Configuration validation passed")
    
    def _validate_model_config(self, config: Dict[str, Any]):
        """Validate model configuration."""
        required_fields = ['node_dim', 'edge_dim', 'hidden_dim', 'num_heads', 'num_layers']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required model config field: {field}")
        
        # Validate values
        if config['node_dim'] <= 0:
            raise ValueError("node_dim must be positive")
        if config['edge_dim'] <= 0:
            raise ValueError("edge_dim must be positive")
        if config['hidden_dim'] <= 0:
            raise ValueError("hidden_dim must be positive")
        if config['num_heads'] <= 0:
            raise ValueError("num_heads must be positive")
        if config['num_layers'] <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= config['dropout'] <= 1:
            raise ValueError("dropout must be between 0 and 1")
    
    def _validate_training_config(self, config: Dict[str, Any]):
        """Validate training configuration."""
        required_fields = ['epochs', 'batch_size', 'learning_rate']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required training config field: {field}")
        
        # Validate values
        if config['epochs'] <= 0:
            raise ValueError("epochs must be positive")
        if config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        if config['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")
        if config['weight_decay'] < 0:
            raise ValueError("weight_decay must be non-negative")
        
        # Validate optimizer
        valid_optimizers = ['adam', 'adamw', 'sgd']
        if config['optimizer'] not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}")
    
    def _validate_data_config(self, config: Dict[str, Any]):
        """Validate data configuration."""
        # Validate splits sum to 1
        splits_sum = config['train_split'] + config['val_split'] + config['test_split']
        if abs(splits_sum - 1.0) > 0.001:
            raise ValueError("train_split + val_split + test_split must equal 1.0")
        
        # Validate individual splits
        for split_name in ['train_split', 'val_split', 'test_split']:
            if not 0 < config[split_name] < 1:
                raise ValueError(f"{split_name} must be between 0 and 1")
        
        # Validate graph window
        if config['graph_window_ms'] <= 0:
            raise ValueError("graph_window_ms must be positive")
        if config['max_nodes_per_graph'] <= 0:
            raise ValueError("max_nodes_per_graph must be positive")
    
    def _validate_paths_config(self, config: Dict[str, Any]):
        """Validate paths configuration."""
        required_paths = ['data_dir', 'output_dir', 'checkpoint_dir', 'log_dir']
        
        for path_name in required_paths:
            if path_name not in config:
                raise ValueError(f"Missing required path config field: {path_name}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        if self.config is None:
            return self.default_config.copy()
        return self.config.copy()
    
    def get_model_config(self) -> ModelConfig:
        """
        Get model configuration as dataclass.
        
        Returns:
            Model configuration dataclass
        """
        config = self.get_config().get('model', {})
        return ModelConfig(**config)
    
    def get_training_config(self) -> TrainingConfig:
        """
        Get training configuration as dataclass.
        
        Returns:
            Training configuration dataclass
        """
        config = self.get_config().get('training', {})
        return TrainingConfig(**config)
    
    def get_data_config(self) -> DataConfig:
        """
        Get data configuration as dataclass.
        
        Returns:
            Data configuration dataclass
        """
        config = self.get_config().get('data', {})
        return DataConfig(**config)
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.config is None:
            self.config = self.default_config.copy()
        
        self.config = self._merge_configs(self.config, updates)
        self._validate_config()
        
        logger.info("Configuration updated")
    
    def save_config(self, save_path: str):
        """
        Save current configuration to file.
        
        Args:
            save_path: Path to save configuration
        """
        if self.config is None:
            raise ValueError("No configuration to save")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif save_path.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported save format: {save_path.suffix}")
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def create_directories(self):
        """Create directories specified in configuration."""
        paths_config = self.get_config().get('paths', {})
        
        for path_key, path_value in paths_config.items():
            path = Path(path_value)
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
    
    def get_device(self) -> str:
        """
        Get the device to use based on configuration.
        
        Returns:
            Device string ('cuda', 'cpu', or 'auto')
        """
        hardware_config = self.get_config().get('hardware', {})
        device = hardware_config.get('device', 'auto')
        
        if device == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return device
    
    def get_effective_batch_size(self) -> int:
        """
        Get effective batch size considering hardware constraints.
        
        Returns:
            Effective batch size
        """
        training_config = self.get_config().get('training', {})
        base_batch_size = training_config.get('batch_size', 32)
        
        # Adjust for GPU memory constraints if needed
        device = self.get_device()
        if device == 'cuda':
            import torch
            # Simple heuristic - could be more sophisticated
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory < 8 * 1024**3:  # Less than 8GB
                return min(base_batch_size, 16)
            elif gpu_memory < 16 * 1024**3:  # Less than 16GB
                return min(base_batch_size, 32)
        
        return base_batch_size
    
    def print_config(self):
        """Print the current configuration in a readable format."""
        if self.config is None:
            print("No configuration loaded")
            return
        
        print("Current Configuration:")
        print("=" * 50)
        
        def print_section(config_dict, indent=0):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_section(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value}")
        
        print_section(self.config)
        print("=" * 50)


class ConfigManager:
    """
    High-level configuration manager with additional utilities.
    
    Provides additional functionality for managing configurations
    including environment-specific configs and parameter sweeps.
    """
    
    def __init__(self, base_config_path: Optional[str] = None):
        """
        Initialize config manager.
        
        Args:
            base_config_path: Path to base configuration file
        """
        self.base_loader = ConfigLoader(base_config_path)
        self.environment_configs = {}
        self.sweep_configs = []
    
    def load_environment_config(self, environment: str, config_path: str):
        """
        Load environment-specific configuration.
        
        Args:
            environment: Environment name (e.g., 'development', 'production')
            config_path: Path to environment-specific config
        """
        loader = ConfigLoader(config_path)
        self.environment_configs[environment] = loader
        
        logger.info(f"Loaded environment config for {environment}")
    
    def get_config_for_environment(self, environment: str) -> Dict[str, Any]:
        """
        Get configuration for specific environment.
        
        Args:
            environment: Environment name
            
        Returns:
            Environment-specific configuration
        """
        if environment not in self.environment_configs:
            logger.warning(f"Environment config not found for {environment}, using base config")
            return self.base_loader.get_config()
        
        # Merge base config with environment config
        base_config = self.base_loader.get_config()
        env_config = self.environment_configs[environment].get_config()
        
        return self.base_loader._merge_configs(base_config, env_config)
    
    def create_parameter_sweep(self, param_grid: Dict[str, List[Any]]):
        """
        Create parameter sweep configurations.
        
        Args:
            param_grid: Dictionary of parameter names to list of values
        """
        from itertools import product
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combination in product(*values):
            sweep_config = dict(zip(keys, combination))
            self.sweep_configs.append(sweep_config)
        
        logger.info(f"Created {len(self.sweep_configs)} sweep configurations")
    
    def get_sweep_config(self, sweep_index: int) -> Dict[str, Any]:
        """
        Get configuration for parameter sweep.
        
        Args:
            sweep_index: Index of sweep configuration
            
        Returns:
            Sweep configuration
        """
        if sweep_index >= len(self.sweep_configs):
            raise ValueError(f"Sweep index {sweep_index} out of range")
        
        # Merge sweep config with base config
        base_config = self.base_loader.get_config()
        sweep_config = self.sweep_configs[sweep_index]
        
        return self.base_loader._merge_configs(base_config, {'sweep': sweep_config})
    
    def export_config_template(self, template_path: str):
        """
        Export configuration template with comments.
        
        Args:
            template_path: Path to save template
        """
        template = {
            'model': {
                'name': 'TG-GAT',
                'node_dim': 80,  # Dimension of node features
                'edge_dim': 37,  # Dimension of edge features
                'hidden_dim': 768,  # Hidden dimension
                'num_heads': 8,  # Number of attention heads
                'num_layers': 3,  # Number of layers
                'dropout': 0.1,  # Dropout rate
                'gru_layers': 2  # Number of GRU layers
            },
            'training': {
                'epochs': 10,  # Number of training epochs
                'batch_size': 32,  # Batch size
                'learning_rate': 0.0001,  # Learning rate
                'weight_decay': 1e-5,  # Weight decay
                'optimizer': 'adam',  # Optimizer (adam, adamw, sgd)
                'scheduler': 'cosine',  # Learning rate scheduler
                'mixed_precision': True  # Use mixed precision training
            },
            'data': {
                'graph_window_ms': 100,  # Graph window size in milliseconds
                'max_nodes_per_graph': 1000,  # Maximum nodes per graph
                'train_split': 0.7,  # Training data split
                'val_split': 0.15,  # Validation data split
                'test_split': 0.15,  # Test data split
                'normalize_features': True,  # Normalize features
                'handle_imbalance': True  # Handle class imbalance
            },
            'paths': {
                'data_dir': 'datasets',  # Data directory
                'output_dir': 'outputs',  # Output directory
                'checkpoint_dir': 'checkpoints',  # Checkpoint directory
                'log_dir': 'logs'  # Log directory
            }
        }
        
        loader = ConfigLoader()
        loader.config = template
        loader.save_config(template_path)
        
        logger.info(f"Configuration template exported to {template_path}")

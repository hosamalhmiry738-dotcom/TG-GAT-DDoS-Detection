"""
TG-GAT (Transformer-Gated Graph Attention Network) Model for DDoS Detection.

This module implements the core TG-GAT architecture that combines:
- Graph Attention Networks for spatial relationship modeling
- Temporal Transformers for long-range dependencies
- GRU cells for efficient sequential processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import List, Dict, Optional, Tuple
import logging

from .layers.graph_attention import GraphAttentionLayer, MultiScaleGraphAttention
from .layers.temporal_transformer import TemporalTransformer, CausalTemporalTransformer
from .layers.gru_cell import MultiLayerGRU, AttentionGRU

logger = logging.getLogger(__name__)


class TGGATModel(nn.Module):
    """
    Transformer-Gated Graph Attention Network for DDoS Detection.
    
    This model implements the hybrid architecture described in the research paper,
    combining graph neural networks, transformers, and GRUs for comprehensive
    DDoS attack detection with high accuracy and low latency.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize TG-GAT Model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(TGGATModel, self).__init__()
        
        self.config = config
        self.node_dim = config['model']['node_dim']
        self.edge_dim = config['model']['edge_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.num_heads = config['model']['num_heads']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        self.gru_layers = config['model']['gru_layers']
        
        # Input projections
        self.node_projection = nn.Linear(self.node_dim, self.hidden_dim)
        self.edge_projection = nn.Linear(self.edge_dim, self.hidden_dim)
        
        # Graph attention layers
        self.graph_attention_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                layer = MultiScaleGraphAttention(
                    self.hidden_dim, self.hidden_dim, self.num_heads, self.dropout
                )
            else:
                layer = GraphAttentionLayer(
                    self.hidden_dim, self.hidden_dim, self.num_heads, self.dropout
                )
            self.graph_attention_layers.append(layer)
        
        # Temporal transformer layers
        self.temporal_transformer = TemporalTransformer(
            d_model=self.hidden_dim,
            num_layers=self.num_layers // 2,
            num_heads=self.num_heads,
            d_ff=self.hidden_dim * 4,
            dropout=self.dropout
        )
        
        # GRU layers for sequential processing
        self.gru = AttentionGRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.gru_layers,
            dropout=self.dropout,
            attention_heads=self.num_heads
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 4, 2)  # Binary classification: Benign/DDoS
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # Initialize parameters
        self._reset_parameters()
        
        # Temporal buffer for sequence processing
        self.temporal_buffer = []
        self.max_sequence_length = 100
        
    def _reset_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, graph_batch: Batch, 
                temporal_sequence: Optional[List[Data]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TG-GAT model.
        
        Args:
            graph_batch: Batch of graph data for current time window
            temporal_sequence: Optional sequence of previous graph batches
            
        Returns:
            Dictionary containing predictions and intermediate representations
        """
        # Extract node and edge features
        x = graph_batch.x if hasattr(graph_batch, 'x') else None
        edge_index = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr if hasattr(graph_batch, 'edge_attr') else None
        batch = graph_batch.batch
        
        # Handle missing features
        if x is None:
            x = torch.ones(graph_batch.num_nodes, self.node_dim, device=graph_batch.device)
        
        if edge_attr is None:
            edge_attr = torch.ones(graph_batch.num_edges, self.edge_dim, device=graph_batch.device)
        
        # Project features to hidden dimension
        x = self.node_projection(x)
        edge_attr = self.edge_projection(edge_attr)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Graph attention processing
        graph_outputs = []
        current_x = x
        
        for i, layer in enumerate(self.graph_attention_layers):
            current_x = layer(current_x, edge_index, edge_attr)
            graph_outputs.append(current_x)
        
        # Global graph pooling
        graph_representation = self._graph_pooling(current_x, batch)
        
        # Temporal processing
        temporal_representation = self._temporal_processing(
            graph_representation, temporal_sequence
        )
        
        # GRU processing
        gru_output, context_vector = self.gru(
            temporal_representation.unsqueeze(1)  # Add sequence dimension
        )
        
        # Classification
        logits = self.classifier(context_vector)
        probabilities = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'graph_representation': graph_representation,
            'temporal_representation': temporal_representation,
            'context_vector': context_vector,
            'attention_weights': graph_outputs[-1] if graph_outputs else None
        }
    
    def _graph_pooling(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Perform graph pooling to get graph-level representation.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch indices [num_nodes]
            
        Returns:
            Graph-level representation [batch_size, hidden_dim]
        """
        # Combine mean and max pooling
        mean_pooled = global_mean_pool(x, batch)
        max_pooled = global_max_pool(x, batch)
        
        # Concatenate and project
        combined = torch.cat([mean_pooled, max_pooled], dim=-1)
        graph_repr = self._project_graph_representation(combined)
        
        return graph_repr
    
    def _project_graph_representation(self, combined: torch.Tensor) -> torch.Tensor:
        """
        Project combined pooled representation to hidden dimension.
        
        Args:
            combined: Combined pooled representation [batch_size, 2*hidden_dim]
            
        Returns:
            Projected representation [batch_size, hidden_dim]
        """
        if not hasattr(self, 'graph_proj'):
            self.graph_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
            self.graph_proj = self.graph_proj.to(combined.device)
        
        return self.graph_proj(combined)
    
    def _temporal_processing(self, 
                           current_repr: torch.Tensor,
                           temporal_sequence: Optional[List[Data]] = None) -> torch.Tensor:
        """
        Process temporal sequence of graph representations.
        
        Args:
            current_repr: Current graph representation [batch_size, hidden_dim]
            temporal_sequence: Sequence of previous graph representations
            
        Returns:
            Temporal representation [batch_size, hidden_dim]
        """
        if temporal_sequence is None or len(temporal_sequence) == 0:
            # No temporal context, return current representation
            return current_repr
        
        # Build temporal sequence
        sequence = [current_repr.unsqueeze(1)]  # Current representation
        
        # Add previous representations (limit to max_sequence_length)
        for i, prev_batch in enumerate(temporal_sequence[-self.max_sequence_length+1:]):
            if hasattr(prev_batch, 'graph_representation'):
                seq_repr = prev_batch['graph_representation'].unsqueeze(1)
                sequence.append(seq_repr)
        
        # Concatenate sequence
        temporal_seq = torch.cat(sequence, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # Apply temporal transformer
        temporal_repr = self.temporal_transformer(temporal_seq)
        
        # Return the last representation (current time step)
        return temporal_repr[:, -1, :]
    
    def update_temporal_buffer(self, graph_batch: Batch, outputs: Dict[str, torch.Tensor]):
        """
        Update temporal buffer with current graph representation.
        
        Args:
            graph_batch: Current graph batch
            outputs: Model outputs containing graph representation
        """
        # Store current representation
        current_repr = outputs['graph_representation'].detach()
        self.temporal_buffer.append({
            'graph_representation': current_repr,
            'timestamp': getattr(graph_batch, 'window_timestamp', 0)
        })
        
        # Limit buffer size
        if len(self.temporal_buffer) > self.max_sequence_length:
            self.temporal_buffer.pop(0)
    
    def predict(self, graph_batch: Batch, 
                temporal_sequence: Optional[List[Data]] = None) -> torch.Tensor:
        """
        Make predictions for DDoS detection.
        
        Args:
            graph_batch: Batch of graph data
            temporal_sequence: Optional temporal sequence
            
        Returns:
            Predictions [batch_size]
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(graph_batch, temporal_sequence)
            predictions = torch.argmax(outputs['probabilities'], dim=-1)
        
        return predictions
    
    def predict_proba(self, graph_batch: Batch,
                     temporal_sequence: Optional[List[Data]] = None) -> torch.Tensor:
        """
        Predict class probabilities for DDoS detection.
        
        Args:
            graph_batch: Batch of graph data
            temporal_sequence: Optional temporal sequence
            
        Returns:
            Class probabilities [batch_size, 2]
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(graph_batch, temporal_sequence)
            probabilities = outputs['probabilities']
        
        return probabilities
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: torch.Tensor, 
                    loss_fn: nn.Module) -> torch.Tensor:
        """
        Compute loss for model training.
        
        Args:
            outputs: Model outputs
            targets: Target labels
            loss_fn: Loss function
            
        Returns:
            Computed loss
        """
        logits = outputs['logits']
        return loss_fn(logits, targets)
    
    def get_attention_weights(self, graph_batch: Batch) -> Optional[torch.Tensor]:
        """
        Get attention weights for explainability.
        
        Args:
            graph_batch: Batch of graph data
            
        Returns:
            Attention weights or None if not available
        """
        outputs = self.forward(graph_batch)
        return outputs.get('attention_weights')
    
    def reset_temporal_buffer(self):
        """Reset the temporal buffer for new sequences."""
        self.temporal_buffer.clear()
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'TG-GAT',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'gru_layers': self.gru_layers,
            'dropout': self.dropout
        }


class TGGATLightweight(nn.Module):
    """
    Lightweight version of TG-GAT for resource-constrained environments.
    
    Reduced complexity while maintaining core functionality for
    deployment on edge devices or firewalls with limited resources.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Lightweight TG-GAT.
        
        Args:
            config: Configuration dictionary
        """
        super(TGGATLightweight, self).__init__()
        
        # Reduced dimensions for efficiency
        self.node_dim = config['model']['node_dim']
        self.hidden_dim = config['model']['hidden_dim'] // 2  # Half the size
        self.num_heads = min(config['model']['num_heads'], 4)  # Max 4 heads
        self.num_layers = max(config['model']['num_layers'] // 2, 1)  # At least 1 layer
        
        # Simplified architecture
        self.node_projection = nn.Linear(self.node_dim, self.hidden_dim)
        self.graph_attention = GraphAttentionLayer(
            self.hidden_dim, self.hidden_dim, self.num_heads, dropout=0.1
        )
        
        # Single GRU layer
        self.gru = MultiLayerGRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            dropout=0.1
        )
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 2)
        )
    
    def forward(self, graph_batch: Batch) -> torch.Tensor:
        """
        Forward pass through lightweight TG-GAT.
        
        Args:
            graph_batch: Batch of graph data
            
        Returns:
            Classification logits
        """
        # Extract features
        x = graph_batch.x if hasattr(graph_batch, 'x') else None
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        if x is None:
            x = torch.ones(graph_batch.num_nodes, self.node_dim, device=graph_batch.device)
        
        # Project and process
        x = self.node_projection(x)
        x = self.graph_attention(x, edge_index)
        
        # Pool and classify
        x = global_mean_pool(x, batch)
        logits = self.classifier(x)
        
        return logits

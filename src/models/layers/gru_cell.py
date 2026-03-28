"""
GRU Cell for TG-GAT DDoS Detection System.

This module implements Gated Recurrent Units for efficient sequential
processing of temporal network traffic data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GRUCell(nn.Module):
    """
    Gated Recurrent Unit cell for sequential processing.
    
    Implements the GRU cell as described in Cho et al. (2014)
    with optimizations for DDoS detection tasks.
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize GRU Cell.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden features
            bias: Whether to use bias terms
        """
        super(GRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Reset gate
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        
        # Update gate
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        
        # New gate (candidate hidden state)
        self.new_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize model parameters."""
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
        
        if self.bias:
            for bias in self.parameters():
                if bias.dim() > 0:
                    bias.data.fill_(0)
    
    def forward(self, input: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through GRU cell.
        
        Args:
            input: Input tensor [batch_size, input_size]
            hidden: Hidden state tensor [batch_size, hidden_size]
            
        Returns:
            New hidden state [batch_size, hidden_size]
        """
        if hidden is None:
            hidden = torch.zeros(input.size(0), self.hidden_size, device=input.device)
        
        # Concatenate input and hidden state
        combined = torch.cat([input, hidden], dim=1)
        
        # Compute reset gate
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        
        # Compute update gate
        update_gate = torch.sigmoid(self.update_gate(combined))
        
        # Compute candidate hidden state
        combined_reset = torch.cat([input, hidden * reset_gate], dim=1)
        new_hidden = torch.tanh(self.new_gate(combined_reset))
        
        # Compute new hidden state
        new_hidden = (1 - update_gate) * hidden + update_gate * new_hidden
        
        return new_hidden


class MultiLayerGRU(nn.Module):
    """
    Multi-layer GRU for deep sequential processing.
    
    Stacks multiple GRU layers to capture hierarchical temporal patterns
    in network traffic data.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 batch_first: bool = True):
        """
        Initialize Multi-layer GRU.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden features per layer
            num_layers: Number of GRU layers
            dropout: Dropout probability between layers
            bidirectional: Whether to use bidirectional GRU
            batch_first: Whether batch dimension comes first
        """
        super(MultiLayerGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        # Create GRU layers
        self.layers = nn.ModuleList()
        layer_input_size = input_size
        
        for i in range(num_layers):
            layer = GRUCell(layer_input_size, hidden_size)
            self.layers.append(layer)
            layer_input_size = hidden_size * (2 if bidirectional else 1)
        
        # Dropout layer
        if dropout > 0 and num_layers > 1:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
        # Initialize hidden state
        self.hidden_state = None
    
    def forward(self, input: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through multi-layer GRU.
        
        Args:
            input: Input tensor [batch_size, seq_len, input_size] if batch_first
            hidden: Initial hidden states for each layer
            
        Returns:
            Tuple of (output, hidden_states)
        """
        if self.batch_first:
            batch_size, seq_len, _ = input.size()
        else:
            seq_len, batch_size, _ = input.size()
            input = input.transpose(0, 1)  # Make batch_first
        
        # Initialize hidden states if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, input.device)
        
        # Process sequence
        outputs = []
        layer_hidden_states = []
        
        for t in range(seq_len):
            layer_input = input[:, t, :]
            new_hidden_states = []
            
            for layer_idx, layer in enumerate(self.layers):
                # Get hidden state for current layer
                layer_hidden = hidden[layer_idx] if hidden is not None else None
                
                # Forward pass through layer
                layer_output = layer(layer_input, layer_hidden)
                new_hidden_states.append(layer_output)
                
                # Apply dropout between layers (except last layer)
                if self.dropout_layer is not None and layer_idx < len(self.layers) - 1:
                    layer_output = self.dropout_layer(layer_output)
                
                # Output of this layer becomes input to next layer
                layer_input = layer_output
            
            # Store outputs and hidden states
            outputs.append(layer_input)
            layer_hidden_states = new_hidden_states
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_size]
        
        # Convert to original format if needed
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, tuple(layer_hidden_states)
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """
        Initialize hidden states for all layers.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of hidden states for each layer
        """
        hidden_states = []
        for layer in self.layers:
            hidden_state = torch.zeros(batch_size, layer.hidden_size, device=device)
            hidden_states.append(hidden_state)
        return tuple(hidden_states)


class AttentionGRU(nn.Module):
    """
    GRU with attention mechanism for enhanced sequential processing.
    
    Combines GRU cells with attention to focus on important temporal
    segments in network traffic sequences.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 attention_heads: int = 4):
        """
        Initialize Attention GRU.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden features
            num_layers: Number of GRU layers
            dropout: Dropout probability
            attention_heads: Number of attention heads
        """
        super(AttentionGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        
        # Multi-layer GRU
        self.gru = MultiLayerGRU(
            input_size, hidden_size, num_layers, dropout
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, input: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Attention GRU.
        
        Args:
            input: Input tensor [batch_size, seq_len, input_size]
            hidden: Initial hidden states
            
        Returns:
            Tuple of (attended_output, context_vector)
        """
        # Pass through GRU
        gru_output, gru_hidden = self.gru(input, hidden)
        
        # Apply attention
        attended_output, attention_weights = self.attention(
            gru_output, gru_output, gru_output
        )
        
        # Residual connection and layer normalization
        attended_output = self.layer_norm(gru_output + attended_output)
        
        # Output projection
        attended_output = self.output_proj(attended_output)
        
        # Compute context vector (weighted average)
        context_vector = torch.mean(attended_output, dim=1)
        
        return attended_output, context_vector


class EfficientGRU(nn.Module):
    """
    Memory-efficient GRU optimized for real-time DDoS detection.
    
    Implements optimizations for faster inference and lower memory usage
    while maintaining detection accuracy.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_cudnn: bool = True):
        """
        Initialize Efficient GRU.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden features
            num_layers: Number of GRU layers
            dropout: Dropout probability
            use_cudnn: Whether to use CuDNN optimization (if available)
        """
        super(EfficientGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_cudnn = use_cudnn and torch.cuda.is_available()
        
        # Use PyTorch's optimized GRU if available
        if self.use_cudnn:
            self.gru = nn.GRU(
                input_size, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        else:
            # Use custom implementation
            self.gru = MultiLayerGRU(
                input_size, hidden_size, num_layers, dropout
            )
        
        # Optimization for inference
        self.inference_mode = False
    
    def forward(self, input: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Efficient GRU.
        
        Args:
            input: Input tensor [batch_size, seq_len, input_size]
            hidden: Initial hidden state
            
        Returns:
            Tuple of (output, final_hidden_state)
        """
        if self.use_cudnn and not self.training:
            # Use optimized CuDNN implementation
            output, hidden = self.gru(input, hidden)
        else:
            # Use custom implementation
            output, hidden = self.gru(input, hidden)
        
        return output, hidden
    
    def set_inference_mode(self, mode: bool = True):
        """
        Set inference mode for optimization.
        
        Args:
            mode: Whether to enable inference mode
        """
        self.inference_mode = mode
        if mode:
            self.eval()
            # Apply inference optimizations
            for module in self.modules():
                if hasattr(module, 'eval'):
                    module.eval()
        else:
            self.train()


# Import math for parameter initialization
import math

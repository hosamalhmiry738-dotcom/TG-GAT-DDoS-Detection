"""
Temporal Transformer Layer for TG-GAT DDoS Detection System.

This module implements temporal transformer mechanisms for capturing
long-range dependencies in time-series network traffic data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-Head Temporal Attention for time-series analysis.
    
    Implements scaled dot-product attention for temporal sequences
    with positional encoding and causal masking.
    """
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_len: int = 1000):
        """
        Initialize Multi-Head Temporal Attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_len: Maximum sequence length for positional encoding
        """
        super(MultiHeadTemporalAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize model parameters."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal attention.
        
        Args:
            x: Input sequence [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Attended sequence [batch_size, seq_len, d_model]
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        attn_output = self._attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # Output projection
        output = self.w_o(attn_output)
        
        return output
    
    def _attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor [batch_size, num_heads, seq_len, d_k]
            K: Key tensor [batch_size, num_heads, seq_len, d_k]
            V: Value tensor [batch_size, num_heads, seq_len, d_k]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch_size, num_heads, seq_len, d_k]
        """
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, V)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences.
    
    Adds sinusoidal positional information to token embeddings
    to help the model understand temporal order.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        """
        Initialize Positional Encoding.
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Input with positional encoding
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TemporalTransformerLayer(nn.Module):
    """
    Temporal Transformer Layer for sequence modeling.
    
    Combines multi-head temporal attention with feed-forward networks
    and residual connections for robust temporal feature extraction.
    """
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize Temporal Transformer Layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super(TemporalTransformerLayer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-head attention
        self.self_attention = MultiHeadTemporalAttention(
            d_model, num_heads, dropout
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal transformer layer.
        
        Args:
            x: Input sequence [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output sequence [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for time-series DDoS detection.
    
    Stacks multiple temporal transformer layers to capture
    complex temporal patterns in network traffic.
    """
    
    def __init__(self, 
                 d_model: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize Temporal Transformer.
        
        Args:
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super(TemporalTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Stack transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(
                d_model, num_heads, d_ff, dropout, activation
            ) for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal transformer.
        
        Args:
            x: Input sequence [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output sequence [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.final_norm(x)
        return x


class CausalTemporalTransformer(nn.Module):
    """
    Causal Temporal Transformer for real-time DDoS detection.
    
    Uses causal masking to ensure that predictions only depend on
    past and present information, suitable for real-time applications.
    """
    
    def __init__(self, 
                 d_model: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1):
        """
        Initialize Causal Temporal Transformer.
        
        Args:
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super(CausalTemporalTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Stack transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through causal temporal transformer.
        
        Args:
            x: Input sequence [batch_size, seq_len, d_model]
            
        Returns:
            Output sequence [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)
        
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.final_norm(x)
        return x

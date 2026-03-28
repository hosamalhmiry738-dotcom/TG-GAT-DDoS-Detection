"""
Graph Attention Layer for TG-GAT DDoS Detection System.

This module implements multi-head graph attention mechanisms for modeling
spatial relationships in network traffic graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple
import math

logger = logging.getLogger(__name__)


class GraphAttentionLayer(MessagePassing):
    """
    Multi-Head Graph Attention Layer for network traffic analysis.
    
    Implements the graph attention mechanism from "Graph Attention Networks"
    (Veličković et al., ICLR 2018) with adaptations for DDoS detection.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 bias: bool = True,
                 concat: bool = True):
        """
        Initialize the Graph Attention Layer.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features per head
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias
            concat: Whether to concatenate head outputs (True) or average (False)
        """
        super(GraphAttentionLayer, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        
        # Linear transformations for query, key, value
        self.lin_query = nn.Linear(in_channels, num_heads * out_channels, bias=bias)
        self.lin_key = nn.Linear(in_channels, num_heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, num_heads * out_channels, bias=bias)
        
        # Output linear transformation
        if concat:
            self.lin_out = nn.Linear(num_heads * out_channels, out_channels * num_heads, bias=bias)
        else:
            self.lin_out = nn.Linear(out_channels, out_channels, bias=bias)
        
        # Dropout and layer normalization
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels * num_heads if concat else out_channels)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize model parameters."""
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        
        if self.lin_query.bias is not None:
            nn.init.constant_(self.lin_query.bias, 0.)
        if self.lin_key.bias is not None:
            nn.init.constant_(self.lin_key.bias, 0.)
        if self.lin_value.bias is not None:
            nn.init.constant_(self.lin_value.bias, 0.)
        if self.lin_out.bias is not None:
            nn.init.constant_(self.lin_out.bias, 0.)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the graph attention layer.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge index tensor [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_feat_dim]
            
        Returns:
            Updated node features [num_nodes, out_channels * num_heads] if concat
            or [num_nodes, out_channels] if not concat
        """
        # Linear transformations
        query = self.lin_query(x).view(-1, self.num_heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.num_heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.num_heads, self.out_channels)
        
        # Compute attention coefficients
        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr)
        
        # Apply output transformation
        if self.concat:
            out = out.view(-1, self.num_heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        out = self.lin_out(out)
        out = self.layer_norm(out)
        
        return out
    
    def message(self, query_j: torch.Tensor, key_i: torch.Tensor, 
                value_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None,
                index: torch.Tensor, ptr: Optional[torch.Tensor] = None,
                size_i: Optional[int] = None) -> torch.Tensor:
        """
        Message passing function for graph attention.
        
        Args:
            query_j: Query features of destination nodes
            key_i: Key features of source nodes
            value_j: Value features of source nodes
            edge_attr: Edge features
            index: Edge indices
            ptr: Pointer for batch processing
            size_i: Size of destination nodes
            
        Returns:
            Message tensor
        """
        # Compute attention scores
        alpha = (query_j * key_i).sum(dim=-1) / math.sqrt(self.out_channels)
        
        # Add edge features if available
        if edge_attr is not None:
            # Project edge features to attention space
            edge_proj = self._edge_attention_projection(edge_attr)
            alpha = alpha + edge_proj
        
        # Apply softmax to get attention weights
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.dropout_layer(alpha)
        
        # Apply attention weights to values
        out = value_j * alpha.unsqueeze(-1)
        
        return out
    
    def _edge_attention_projection(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Project edge features to attention space.
        
        Args:
            edge_attr: Edge features [num_edges, edge_feat_dim]
            
        Returns:
            Projected edge features [num_edges, num_heads]
        """
        if not hasattr(self, 'edge_projection'):
            edge_dim = edge_attr.size(-1)
            self.edge_projection = nn.Linear(edge_dim, self.num_heads, bias=False)
            self.edge_projection = self.edge_projection.to(edge_attr.device)
        
        return self.edge_projection(edge_attr)


def softmax(src: torch.Tensor, index: torch.Tensor, 
           ptr: Optional[torch.Tensor] = None, 
           num_nodes: Optional[int] = None) -> torch.Tensor:
    """
    Compute softmax over node neighborhoods.
    
    Args:
        src: Source tensor
        index: Index tensor
        ptr: Pointer for batch processing
        num_nodes: Number of nodes
        
    Returns:
        Softmax tensor
    """
    out = src - src.max().item()
    out = out.exp()
    
    if ptr is not None:
        out_sum = segment_csr(out, ptr, reduce='sum')
        out_sum = out_sum[index]
        out = out / (out_sum + 1e-16)
    else:
        if num_nodes is None:
            num_nodes = index.max().item() + 1
        
        out_sum = scatter_add(out, index, dim=0, dim_size=num_nodes)
        out_sum = out_sum[index]
        out = out / (out_sum + 1e-16)
    
    return out


def segment_csr(src: torch.Tensor, ptr: torch.Tensor, reduce: str = 'sum') -> torch.Tensor:
    """
    Segment reduction for CSR format.
    
    Args:
        src: Source tensor
        ptr: Pointer tensor
        reduce: Reduction operation
        
    Returns:
        Reduced tensor
    """
    if reduce == 'sum':
        out = torch.zeros(ptr.size(0) - 1, dtype=src.dtype, device=src.device)
        for i in range(ptr.size(0) - 1):
            out[i] = src[ptr[i]:ptr[i+1]].sum()
        return out
    else:
        raise ValueError(f"Unsupported reduction: {reduce}")


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0, 
               dim_size: Optional[int] = None) -> torch.Tensor:
    """
    Scatter add operation.
    
    Args:
        src: Source tensor
        index: Index tensor
        dim: Dimension to scatter along
        dim_size: Size of output dimension
        
    Returns:
        Scattered tensor
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, index, src)
    
    return out


class MultiScaleGraphAttention(nn.Module):
    """
    Multi-Scale Graph Attention for capturing different spatial relationships.
    
    Combines multiple graph attention layers with different receptive fields
    to capture both local and global network patterns.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 num_heads: int = 8,
                 num_scales: int = 3,
                 dropout: float = 0.1):
        """
        Initialize Multi-Scale Graph Attention.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            num_heads: Number of attention heads
            num_scales: Number of spatial scales
            dropout: Dropout probability
        """
        super(MultiScaleGraphAttention, self).__init__()
        
        self.num_scales = num_scales
        self.out_channels = out_channels
        
        # Create multiple attention layers for different scales
        self.attention_layers = nn.ModuleList()
        for i in range(num_scales):
            # Vary the number of heads for different scales
            heads = max(1, num_heads // (2 ** i))
            layer = GraphAttentionLayer(
                in_channels if i == 0 else out_channels,
                out_channels // num_scales,
                heads,
                dropout,
                concat=True
            )
            self.attention_layers.append(layer)
        
        # Scale fusion
        self.scale_fusion = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-scale attention.
        
        Args:
            x: Node feature matrix
            edge_index: Edge index tensor
            edge_attr: Edge features
            
        Returns:
            Updated node features
        """
        scale_outputs = []
        
        for i, layer in enumerate(self.attention_layers):
            if i == 0:
                scale_out = layer(x, edge_index, edge_attr)
            else:
                scale_out = layer(scale_outputs[-1], edge_index, edge_attr)
            scale_outputs.append(scale_out)
        
        # Concatenate scale outputs
        multi_scale_out = torch.cat(scale_outputs, dim=-1)
        
        # Apply fusion
        out = self.scale_fusion(multi_scale_out)
        out = self.dropout(out)
        out = self.layer_norm(out + x)  # Residual connection
        
        return out

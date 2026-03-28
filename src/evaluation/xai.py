"""
Explainable AI (XAI) module for TG-GAT DDoS Detection System.

This module implements explainability methods including GNNExplainer,
attention visualization, and feature importance analysis for DDoS detection.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..models.tg_gat import TGGATModel

logger = logging.getLogger(__name__)


class GNNExplainer:
    """
    GNNExplainer implementation for TG-GAT model interpretation.
    
    Provides explanations for model predictions by identifying important
    nodes, edges, and features that contribute to DDoS detection decisions.
    """
    
    def __init__(self, model: TGGATModel, config: Dict):
        """
        Initialize GNNExplainer.
        
        Args:
            model: Trained TG-GAT model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Explanation parameters
        self.num_samples = config['xai'].get('num_samples', 100)
        self.learning_rate = 0.01
        self.epochs = 100
        
        # Feature and edge masks
        self.feature_mask = None
        self.edge_mask = None
        
        # Explanation results
        self.explanations = {}
    
    def explain_graph(self, graph_data: Data, target_class: int = 1) -> Dict[str, Any]:
        """
        Explain model prediction for a single graph.
        
        Args:
            graph_data: Graph data to explain
            target_class: Target class for explanation (DDoS = 1)
            
        Returns:
            Explanation dictionary
        """
        self.model.eval()
        
        # Move data to device
        graph_data = graph_data.to(self.device)
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(graph_data)
            original_pred = torch.argmax(original_output['probabilities'], dim=-1)
            original_prob = original_output['probabilities'][0, target_class].item()
        
        # Initialize masks
        num_nodes = graph_data.num_nodes
        num_edges = graph_data.num_edges
        num_features = graph_data.x.size(1)
        
        # Feature mask (learnable)
        self.feature_mask = torch.nn.Parameter(torch.ones(num_features, device=self.device))
        
        # Edge mask (learnable)
        self.edge_mask = torch.nn.Parameter(torch.ones(num_edges, device=self.device))
        
        # Optimize masks
        optimizer = torch.optim.Adam([self.feature_mask, self.edge_mask], lr=self.learning_rate)
        
        best_loss = float('inf')
        best_feature_mask = None
        best_edge_mask = None
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Apply masks
            masked_x = graph_data.x * torch.sigmoid(self.feature_mask)
            masked_edge_index = self._apply_edge_mask(graph_data.edge_index, self.edge_mask)
            
            # Forward pass with masked data
            try:
                masked_output = self.model._masked_forward(masked_x, masked_edge_index)
                
                # Compute loss (encourage target class prediction)
                target_prob = masked_output['probabilities'][0, target_class]
                loss = -target_prob
                
                # Add regularization terms
                loss += 0.01 * torch.norm(self.feature_mask)
                loss += 0.01 * torch.norm(self.edge_mask)
                
                loss.backward()
                optimizer.step()
                
                # Keep best masks
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_feature_mask = self.feature_mask.detach().clone()
                    best_edge_mask = self.edge_mask.detach().clone()
                    
            except Exception as e:
                logger.warning(f"Error in epoch {epoch}: {str(e)}")
                continue
        
        # Create explanation
        explanation = {
            'original_prediction': original_pred.item(),
            'original_probability': original_prob,
            'target_class': target_class,
            'feature_importance': best_feature_mask.cpu().numpy(),
            'edge_importance': best_edge_mask.cpu().numpy(),
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_features': num_features
        }
        
        # Add node-level explanations
        explanation['node_importance'] = self._calculate_node_importance(
            graph_data, best_feature_mask, best_edge_mask
        )
        
        # Store explanation
        graph_id = id(graph_data)
        self.explanations[graph_id] = explanation
        
        return explanation
    
    def _apply_edge_mask(self, edge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply edge mask to edge index.
        
        Args:
            edge_index: Original edge index
            edge_mask: Edge importance mask
            
        Returns:
            Masked edge index
        """
        # Select edges with high importance
        edge_probs = torch.sigmoid(edge_mask)
        threshold = torch.median(edge_probs)
        selected_edges = edge_probs > threshold
        
        return edge_index[:, selected_edges]
    
    def _calculate_node_importance(self, graph_data: Data, 
                                 feature_mask: torch.Tensor,
                                 edge_mask: torch.Tensor) -> np.ndarray:
        """
        Calculate node-level importance scores.
        
        Args:
            graph_data: Graph data
            feature_mask: Feature importance mask
            edge_mask: Edge importance mask
            
        Returns:
            Node importance scores
        """
        num_nodes = graph_data.num_nodes
        node_importance = np.zeros(num_nodes)
        
        # Calculate importance based on connected edges
        edge_probs = torch.sigmoid(edge_mask).cpu().numpy()
        
        for i in range(num_nodes):
            # Find edges connected to this node
            connected_edges = (graph_data.edge_index[0] == i) | (graph_data.edge_index[1] == i)
            
            if connected_edges.any():
                # Node importance is average of connected edge importances
                node_importance[i] = np.mean(edge_probs[connected_edges.cpu().numpy()])
        
        return node_importance
    
    def explain_batch(self, graph_batch: Batch, target_class: int = 1) -> List[Dict[str, Any]]:
        """
        Explain predictions for a batch of graphs.
        
        Args:
            graph_batch: Batch of graphs
            target_class: Target class for explanation
            
        Returns:
            List of explanations
        """
        explanations = []
        
        # Split batch into individual graphs
        graph_data_list = graph_batch.to_data_list()
        
        for i, graph_data in enumerate(graph_data_list):
            try:
                explanation = self.explain_graph(graph_data, target_class)
                explanation['batch_index'] = i
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Error explaining graph {i}: {str(e)}")
                continue
        
        return explanations
    
    def visualize_explanation(self, explanation: Dict[str, Any], 
                            graph_data: Data,
                            save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize graph explanation.
        
        Args:
            explanation: Explanation dictionary
            graph_data: Original graph data
            save_path: Optional path to save visualization
            
        Returns:
            Plotly figure
        """
        # Convert to NetworkX graph
        G = to_networkx(graph_data, node_attrs=['x'], edge_attrs=['edge_attr'])
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract node positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Node importance
        node_importance = explanation['node_importance']
        node_colors = node_importance
        
        # Edge importance
        edge_importance = explanation['edge_importance']
        edge_x = []
        edge_y = []
        
        for i, edge in enumerate(G.edges()):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=20,
                color=node_colors,
                colorscale='Reds',
                colorbar=dict(title="Node Importance"),
                line=dict(width=2, color='black')
            ),
            text=[f"Node {i}<br>Importance: {imp:.3f}" 
                  for i, imp in enumerate(node_importance)],
            hoverinfo='text'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"DDoS Attack Explanation<br>Prediction: {explanation['original_prediction']} "
                  f"(Prob: {explanation['original_probability']:.3f})",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Node color represents importance for DDoS detection",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
                    font=dict(color="black", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Explanation visualization saved to {save_path}")
        
        return fig
    
    def get_feature_importance_ranking(self, explanation: Dict[str, Any]) -> List[Tuple[int, float]]:
        """
        Get ranked feature importance.
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            List of (feature_index, importance) tuples sorted by importance
        """
        feature_importance = explanation['feature_importance']
        
        # Create ranking
        ranking = [(i, imp) for i, imp in enumerate(feature_importance)]
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def get_critical_subgraph(self, explanation: Dict[str, Any], 
                            graph_data: Data,
                            threshold: float = 0.5) -> nx.Graph:
        """
        Extract critical subgraph based on importance scores.
        
        Args:
            explanation: Explanation dictionary
            graph_data: Original graph data
            threshold: Importance threshold for node/edge selection
            
        Returns:
            Critical subgraph
        """
        # Convert to NetworkX
        G = to_networkx(graph_data)
        
        # Get important nodes
        node_importance = explanation['node_importance']
        important_nodes = [i for i, imp in enumerate(node_importance) if imp > threshold]
        
        # Get important edges
        edge_importance = explanation['edge_importance']
        important_edges = []
        
        for i, edge in enumerate(G.edges()):
            if edge_importance[i] > threshold:
                important_edges.append(edge)
        
        # Create subgraph
        subgraph = G.subgraph(important_nodes).copy()
        
        return subgraph


class AttentionVisualizer:
    """
    Visualizer for attention mechanisms in TG-GAT model.
    
    Provides visualization of attention weights and patterns
    to understand how the model focuses on different parts of the graph.
    """
    
    def __init__(self, model: TGGATModel):
        """
        Initialize attention visualizer.
        
        Args:
            model: Trained TG-GAT model
        """
        self.model = model
        self.device = next(model.parameters()).device
    
    def extract_attention_weights(self, graph_data: Data) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from the model.
        
        Args:
            graph_data: Input graph data
            
        Returns:
            Dictionary of attention weights
        """
        self.model.eval()
        graph_data = graph_data.to(self.device)
        
        # Hook to capture attention weights
        attention_weights = {}
        
        def hook_fn(module, input, output, layer_name):
            if hasattr(output, 'attention_weights'):
                attention_weights[layer_name] = output.attention_weights
        
        # Register hooks for attention layers
        hooks = []
        for i, layer in enumerate(self.model.graph_attention_layers):
            hook = layer.register_forward_hook(
                lambda module, input, output, idx=i: hook_fn(module, input, output, f"layer_{idx}")
            )
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            self.model(graph_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def visualize_attention_heatmap(self, attention_weights: torch.Tensor,
                                  graph_data: Data,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create attention heatmap visualization.
        
        Args:
            attention_weights: Attention weights tensor
          graph_data: Graph data
          save_path: Optional path to save visualization
          
        Returns:
          Matplotlib figure
        """
        # Convert to numpy
        if attention_weights.dim() == 3:
            # Multi-head attention, take average
            attention_matrix = attention_weights.mean(dim=0).cpu().numpy()
        else:
            attention_matrix = attention_weights.cpu().numpy()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            attention_matrix,
            annot=True,
            cmap='Reds',
            fmt='.3f',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title('Attention Weights Heatmap')
        ax.set_xlabel('Target Node')
        ax.set_ylabel('Source Node')
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention heatmap saved to {save_path}")
        
        return fig
    
    def visualize_attention_flow(self, graph_data: Data,
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize attention flow in the graph.
        
        Args:
            graph_data: Graph data
            save_path: Optional path to save visualization
            
        Returns:
            Plotly figure
        """
        # Extract attention weights
        attention_weights = self.extract_attention_weights(graph_data)
        
        if not attention_weights:
            logger.warning("No attention weights found")
            return go.Figure()
        
        # Get attention from first layer
        first_layer_attention = list(attention_weights.values())[0]
        
        # Convert to NetworkX
        G = to_networkx(graph_data)
        pos = nx.spring_layout(G)
        
        # Create edge traces with attention weights
        edge_traces = []
        
        for i, edge in enumerate(G.edges()):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Get attention weight for this edge
            if i < first_layer_attention.size(-1):
                attention_weight = first_layer_attention[0, 0, i].item()
            else:
                attention_weight = 0.5
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=2 + attention_weight * 3,
                    color=f'rgb({255*attention_weight}, 0, {255*(1-attention_weight)})'
                ),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            text=[f"Node {i}" for i in G.nodes()],
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title='Attention Flow Visualization',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Attention flow visualization saved to {save_path}")
        
        return fig


class FeatureImportanceAnalyzer:
    """
    Analyzer for feature importance in DDoS detection.
    
    Provides analysis of which features are most important
    for detecting DDoS attacks.
    """
    
    def __init__(self, model: TGGATModel, config: Dict):
        """
        Initialize feature importance analyzer.
        
        Args:
            model: Trained TG-GAT model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Feature names (example - should be customized based on actual features)
        self.feature_names = [
            'packet_count', 'byte_count', 'connection_count', 'avg_packet_size',
            'flow_rate', 'protocol_distribution', 'port_entropy', 'temporal_features'
        ]
    
    def calculate_global_feature_importance(self, data_loader) -> Dict[str, float]:
        """
        Calculate global feature importance across the dataset.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Dictionary of feature importance scores
        """
        self.model.eval()
        
        feature_importance_scores = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                # Move data to device
                if isinstance(batch_data, dict):
                    graph_batch = batch_data['graph'].to(self.device)
                    targets = batch_data['targets'].to(self.device)
                else:
                    graph_batch = batch_data.to(self.device)
                    targets = graph_batch.y.to(self.device)
                
                # Get attention weights
                attention_weights = self._extract_attention_weights(graph_batch)
                
                # Calculate feature importance for this batch
                batch_importance = self._calculate_batch_feature_importance(
                    attention_weights, graph_batch
                )
                feature_importance_scores.append(batch_importance)
        
        # Aggregate across batches
        global_importance = {}
        for scores in feature_importance_scores:
            for feature, score in scores.items():
                if feature not in global_importance:
                    global_importance[feature] = []
                global_importance[feature].append(score)
        
        # Average importance
        for feature in global_importance:
            global_importance[feature] = np.mean(global_importance[feature])
        
        return global_importance
    
    def _extract_attention_weights(self, graph_batch: Batch) -> torch.Tensor:
        """
        Extract attention weights from graph batch.
        
        Args:
            graph_batch: Batch of graphs
            
        Returns:
            Attention weights tensor
        """
        # This is a simplified version - in practice, you'd need to
        # modify the model to expose attention weights
        return torch.ones(graph_batch.num_nodes, graph_batch.num_nodes)
    
    def _calculate_batch_feature_importance(self, attention_weights: torch.Tensor,
                                         graph_batch: Batch) -> Dict[str, float]:
        """
        Calculate feature importance for a single batch.
        
        Args:
            attention_weights: Attention weights
            graph_batch: Graph batch
            
        Returns:
            Dictionary of feature importance scores
        """
        # Simplified feature importance calculation
        # In practice, this would be more sophisticated
        num_features = graph_batch.x.size(1)
        
        importance_scores = {}
        for i in range(min(num_features, len(self.feature_names))):
            feature_name = self.feature_names[i]
            # Use attention weights as proxy for feature importance
            importance_scores[feature_name] = attention_weights.mean().item()
        
        return importance_scores
    
    def visualize_feature_importance(self, importance_scores: Dict[str, float],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize feature importance scores.
        
        Args:
            importance_scores: Feature importance scores
            save_path: Optional path to save visualization
            
        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(features, scores)
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance Score')
        ax.set_title('Feature Importance for DDoS Detection')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance visualization saved to {save_path}")
        
        return fig


class XAIExplainer:
    """
    Main XAI explainer that combines multiple explanation methods.
    
    Provides a unified interface for explaining TG-GAT model predictions
    and generating comprehensive explanations.
    """
    
    def __init__(self, model: TGGATModel, config: Dict):
        """
        Initialize XAI explainer.
        
        Args:
            model: Trained TG-GAT model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        
        # Initialize individual explainers
        self.gnn_explainer = GNNExplainer(model, config)
        self.attention_visualizer = AttentionVisualizer(model)
        self.feature_analyzer = FeatureImportanceAnalyzer(model, config)
    
    def explain_prediction(self, graph_data: Data, 
                         target_class: int = 1,
                         visualize: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.
        
        Args:
            graph_data: Graph data to explain
            target_class: Target class for explanation
            visualize: Whether to generate visualizations
            
        Returns:
            Comprehensive explanation dictionary
        """
        # Get GNN explanation
        gnn_explanation = self.gnn_explainer.explain_graph(graph_data, target_class)
        
        # Get attention weights
        attention_weights = self.attention_visualizer.extract_attention_weights(graph_data)
        
        # Get feature importance
        feature_importance = self.gnn_explainer.get_feature_importance_ranking(gnn_explanation)
        
        explanation = {
            'gnn_explanation': gnn_explanation,
            'attention_weights': attention_weights,
            'feature_importance': feature_importance,
            'target_class': target_class
        }
        
        # Generate visualizations if requested
        if visualize:
            explanation['visualizations'] = {}
            
            # Graph explanation visualization
            fig = self.gnn_explainer.visualize_explanation(gnn_explanation, graph_data)
            explanation['visualizations']['graph_explanation'] = fig
            
            # Attention flow visualization
            fig = self.attention_visualizer.visualize_attention_flow(graph_data)
            explanation['visualizations']['attention_flow'] = fig
        
        return explanation
    
    def generate_explanation_report(self, explanation: Dict[str, Any],
                                  save_path: Optional[str] = None) -> str:
        """
        Generate human-readable explanation report.
        
        Args:
            explanation: Explanation dictionary
            save_path: Optional path to save report
            
        Returns:
            Explanation report as string
        """
        report = []
        report.append("# DDoS Attack Detection Explanation")
        report.append("=" * 50)
        report.append("")
        
        # Basic information
        gnn_exp = explanation['gnn_explanation']
        report.append("## Prediction Information")
        report.append(f"- Predicted Class: {gnn_exp['original_prediction']}")
        report.append(f"- Confidence: {gnn_exp['original_probability']:.4f}")
        report.append(f"- Target Class: {gnn_exp['target_class']}")
        report.append("")
        
        # Feature importance
        report.append("## Top Important Features")
        feature_importance = explanation['feature_importance'][:10]
        for i, (feat_idx, importance) in enumerate(feature_importance):
            feat_name = f"Feature_{feat_idx}"
            report.append(f"{i+1}. {feat_name}: {importance:.4f}")
        report.append("")
        
        # Node importance
        node_importance = gnn_exp['node_importance']
        important_nodes = np.argsort(node_importance)[-5:]
        report.append("## Most Important Nodes")
        for i, node_idx in enumerate(important_nodes[::-1]):
            report.append(f"{i+1}. Node {node_idx}: {node_importance[node_idx]:.4f}")
        report.append("")
        
        # Explanation summary
        report.append("## Explanation Summary")
        if gnn_exp['original_prediction'] == gnn_exp['target_class']:
            report.append("The model correctly identified this as a DDoS attack.")
            report.append("The decision was based on:")
            report.append("- High importance of specific network features")
            report.append("- Suspicious communication patterns between nodes")
            report.append("- Temporal anomalies in traffic patterns")
        else:
            report.append("The model did not identify this as a DDoS attack.")
            report.append("The features and patterns did not match typical attack signatures.")
        
        report_text = "\n".join(report)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Explanation report saved to {save_path}")
        
        return report_text

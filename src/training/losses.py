"""
Loss functions for TG-GAT DDoS Detection System.

This module implements custom loss functions optimized for DDoS detection
including focal loss for class imbalance and task-specific losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in DDoS detection.
    
    Implements focal loss as described in Lin et al. (2017) to focus
    training on hard-to-classify examples and handle class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (DDoS attacks)
            gamma: Focusing parameter, higher gamma gives more weight to hard examples
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss tensor
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TGATLoss(nn.Module):
    """
    Custom loss function for TG-GAT model.
    
    Combines multiple loss components to optimize different aspects
    of the DDoS detection task including classification accuracy,
    temporal consistency, and attention regularization.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize TGAT Loss.
        
        Args:
            config: Configuration dictionary
        """
        super(TGATLoss, self).__init__()
        
        self.config = config
        
        # Base classification loss
        loss_config = config['loss']
        if loss_config['type'] == 'focal':
            self.classification_loss = FocalLoss(
                alpha=loss_config['alpha'],
                gamma=loss_config['gamma']
            )
        else:
            self.classification_loss = nn.CrossEntropyLoss()
        
        # Loss weights
        self.classification_weight = 1.0
        self.temporal_weight = 0.1
        self.attention_weight = 0.05
        self.graph_weight = 0.1
        
        # Label smoothing
        self.label_smoothing = loss_config.get('label_smoothing', 0.0)
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute TGAT loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth labels
            
        Returns:
            Combined loss tensor
        """
        # Classification loss
        logits = outputs['logits']
        
        if self.label_smoothing > 0:
            classification_loss = self._label_smoothed_cross_entropy(logits, targets)
        else:
            classification_loss = self.classification_loss(logits, targets)
        
        total_loss = self.classification_weight * classification_loss
        
        # Temporal consistency loss
        if 'temporal_representation' in outputs:
            temporal_loss = self._temporal_consistency_loss(
                outputs['temporal_representation'], targets
            )
            total_loss += self.temporal_weight * temporal_loss
        
        # Attention regularization loss
        if 'attention_weights' in outputs:
            attention_loss = self._attention_regularization_loss(
                outputs['attention_weights']
            )
            total_loss += self.attention_weight * attention_loss
        
        # Graph structure loss
        if 'graph_representation' in outputs:
            graph_loss = self._graph_structure_loss(
                outputs['graph_representation'], targets
            )
            total_loss += self.graph_weight * graph_loss
        
        return total_loss
    
    def _label_smoothed_cross_entropy(self, logits: torch.Tensor, 
                                     targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross entropy loss.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Label-smoothed loss tensor
        """
        num_classes = logits.size(-1)
        
        # Create smooth labels
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Compute cross entropy with smooth labels
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_targets * log_probs, dim=-1).mean()
        
        return loss
    
    def _temporal_consistency_loss(self, temporal_repr: torch.Tensor,
                                  targets: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Encourages similar temporal representations for samples
        of the same class.
        
        Args:
            temporal_repr: Temporal representations [batch_size, hidden_dim]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Temporal consistency loss
        """
        batch_size = temporal_repr.size(0)
        
        # Compute pairwise similarities
        similarities = F.cosine_similarity(
            temporal_repr.unsqueeze(1), 
            temporal_repr.unsqueeze(0), 
            dim=-1
        )
        
        # Create target similarity matrix
        target_matrix = targets.unsqueeze(1) == targets.unsqueeze(0)
        target_matrix = target_matrix.float()
        
        # Remove diagonal elements
        mask = torch.eye(batch_size, device=temporal_repr.device).bool()
        similarities = similarities[~mask]
        target_matrix = target_matrix[~mask]
        
        # Compute consistency loss
        consistency_loss = F.mse_loss(similarities, target_matrix)
        
        return consistency_loss
    
    def _attention_regularization_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention regularization loss.
        
        Encourages sparse and diverse attention patterns.
        
        Args:
            attention_weights: Attention weights [num_nodes, hidden_dim]
            
        Returns:
            Attention regularization loss
        """
        # Sparsity regularization
        sparsity_loss = torch.mean(torch.abs(attention_weights))
        
        # Diversity regularization (encourage different attention patterns)
        if attention_weights.size(0) > 1:
            # Compute correlation matrix
            attention_normalized = F.normalize(attention_weights, p=2, dim=1)
            correlation_matrix = torch.mm(attention_normalized, attention_normalized.t())
            
            # Remove diagonal and take absolute values
            mask = torch.eye(correlation_matrix.size(0), device=correlation_matrix.device).bool()
            off_diagonal = correlation_matrix[~mask]
            
            diversity_loss = torch.mean(torch.abs(off_diagonal))
        else:
            diversity_loss = torch.tensor(0.0, device=attention_weights.device)
        
        return sparsity_loss + diversity_loss
    
    def _graph_structure_loss(self, graph_repr: torch.Tensor,
                             targets: torch.Tensor) -> torch.Tensor:
        """
        Compute graph structure loss.
        
        Encourages meaningful graph representations that capture
        structural differences between classes.
        
        Args:
            graph_repr: Graph representations [batch_size, hidden_dim]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Graph structure loss
        """
        # Compute class-wise centroids
        unique_classes = torch.unique(targets)
        class_centroids = []
        
        for cls in unique_classes:
            class_mask = targets == cls
            if torch.sum(class_mask) > 0:
                class_repr = graph_repr[class_mask]
                centroid = torch.mean(class_repr, dim=0)
                class_centroids.append(centroid)
        
        if len(class_centroids) >= 2:
            # Compute inter-class distances
            centroids_tensor = torch.stack(class_centroids)
            inter_class_distances = torch.pdist(centroids_tensor, p=2)
            
            # Maximize inter-class distances
            structure_loss = -torch.mean(inter_class_distances)
        else:
            structure_loss = torch.tensor(0.0, device=graph_repr.device)
        
        return structure_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for imbalanced DDoS detection.
    
    Applies different weights to positive (DDoS) and negative (Benign)
    samples to handle class imbalance.
    """
    
    def __init__(self, pos_weight: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive samples (DDoS attacks)
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            inputs: Predictions [batch_size, 1] or [batch_size]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Weighted BCE loss tensor
        """
        # Ensure inputs are in the right format
        if inputs.dim() > 1 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        
        # Create weight tensor
        pos_weight_tensor = torch.tensor(self.pos_weight, device=inputs.device)
        weights = torch.where(targets == 1, pos_weight_tensor, torch.tensor(1.0, device=inputs.device))
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        
        # Apply weights
        weighted_loss = bce_loss * weights
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for learning discriminative representations.
    
    Encourages similar samples to be close in embedding space and
    dissimilar samples to be far apart.
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        """
        Initialize Contrastive Loss.
        
        Args:
            margin: Margin for dissimilar samples
            temperature: Temperature for similarity computation
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Sample embeddings [batch_size, embedding_dim]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Contrastive loss tensor
        """
        batch_size = embeddings.size(0)
        
        # Compute similarity matrix
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create label matrix
        label_matrix = targets.unsqueeze(1) == targets.unsqueeze(0)
        label_matrix = label_matrix.float()
        
        # Remove diagonal elements
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        similarity_matrix = similarity_matrix[~mask]
        label_matrix = label_matrix[~mask]
        
        # Compute contrastive loss
        positive_pairs = similarity_matrix[label_matrix == 1]
        negative_pairs = similarity_matrix[label_matrix == 0]
        
        # Positive loss (maximize similarity)
        if len(positive_pairs) > 0:
            positive_loss = -torch.mean(positive_pairs)
        else:
            positive_loss = torch.tensor(0.0, device=embeddings.device)
        
        # Negative loss (minimize similarity with margin)
        if len(negative_pairs) > 0:
            negative_loss = torch.mean(torch.clamp(negative_pairs - self.margin, min=0))
        else:
            negative_loss = torch.tensor(0.0, device=embeddings.device)
        
        return positive_loss + negative_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for joint optimization of multiple objectives.
    
    Combines different loss functions with learnable weights for
    automatic balancing of multiple tasks.
    """
    
    def __init__(self, num_tasks: int, task_weights: Optional[torch.Tensor] = None):
        """
        Initialize Multi-task Loss.
        
        Args:
            num_tasks: Number of tasks
            task_weights: Initial task weights [num_tasks]
        """
        super(MultiTaskLoss, self).__init__()
        
        self.num_tasks = num_tasks
        
        # Learnable task weights (log variance)
        if task_weights is not None:
            self.log_vars = nn.Parameter(torch.log(task_weights))
        else:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, task_losses: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted multi-task loss.
        
        Args:
            task_losses: Individual task losses [num_tasks]
            
        Returns:
            Combined multi-task loss
        """
        # Compute precision weights
        precision_weights = torch.exp(-self.log_vars)
        
        # Compute weighted losses
        weighted_losses = precision_weights * task_losses
        
        # Add uncertainty regularization
        total_loss = torch.sum(weighted_losses) + torch.sum(self.log_vars)
        
        return total_loss

"""
Dual-Stream Multiple Instance Learning (MIL) for Brain Tumor Classification

This module implements a Dual-Stream MIL architecture that:
1. Identifies critical instances (slices) in each bag (patient)
2. Aggregates contextual information from all instances with attention
3. Combines critical and contextual signals for patient-level classification

Architecture:
    Bag of N slices (instances)
      → Instance Encoder (shared 2D CNN)
      → Stream 1: Critical Instance Selection
      → Stream 2: Contextual Attention Aggregation
      → Fusion → Classification Head
      → Patient-level logits

Author: Medical Imaging Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class InstanceEncoder(nn.Module):
    """
    2D CNN encoder for multi-modal slices.
    
    Adapts ResNet18 or EfficientNet-B0 for 4-channel multi-modal input.
    Outputs feature vector per slice.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        feature_dim: int = 512,
        input_size: int = 224,
        logger=None
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.input_size = input_size
        
        if backbone == 'resnet18':
            from torchvision.models import resnet18
            # Load ResNet18 without pretrained weights (multi-modal input)
            resnet = resnet18(pretrained=False)
            
            # Adapt first layer for 4-channel input
            self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            self.avgpool = resnet.avgpool
            
            # Feature dimension for ResNet18
            self.feature_dim = 512
            
        elif backbone == 'efficientnet_b0':
            try:
                from torchvision.models import efficientnet_b0
                efficientnet = efficientnet_b0(pretrained=False)
                
                # Adapt first layer for 4-channel input
                self.features = efficientnet.features
                self.features[0][0] = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
                self.avgpool = efficientnet.avgpool
                self.classifier = None  # We only need features
                
                # Feature dimension for EfficientNet-B0
                self.feature_dim = 1280
                
            except ImportError:
                raise ImportError("EfficientNet requires torchvision >= 0.13.0. Use 'resnet18' instead.")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet18' or 'efficientnet_b0'")
        
        if logger:
            total_params = sum(p.numel() for p in self.parameters())
            logger.info(f"InstanceEncoder ({backbone}): {total_params/1e6:.2f}M parameters")
            logger.info(f"Output feature dimension: {self.feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode multi-modal 2D slice to feature vector.
        
        Args:
            x: Input slice (B, 4, H, W) or (B, N, 4, H, W)
        
        Returns:
            features: (B, feature_dim) or (B, N, feature_dim)
        """
        original_shape = x.shape
        
        # Handle batched slices: (B, N, 4, H, W) -> (B*N, 4, H, W)
        if len(original_shape) == 5:
            B, N, C, H, W = original_shape
            x = x.view(B * N, C, H, W)
            reshape_output = True
        else:
            reshape_output = False
        
        # Resize to standard input size if needed
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        # Forward through backbone
        if self.backbone == 'resnet18':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        else:  # efficientnet_b0
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        
        # Reshape back if needed
        if reshape_output:
            x = x.view(B, N, self.feature_dim)
        
        return x


class CriticalInstanceSelector(nn.Module):
    """
    Stream 1: Identifies the most critical instance in the bag.
    
    Uses a scoring network to assign importance scores to each instance,
    then selects the critical instance using soft selection with temperature.
    
    Soft selection is differentiable and allows gradients to flow through
    the selection mechanism, enabling stable training.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        temperature: float = 1.0
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature  # Temperature for softmax (higher = softer, lower = sharper)
        
        # Scoring network: assigns importance score to each instance
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Score in [0, 1]
        )
    
    def forward(self, instance_features: torch.Tensor, temperature: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select critical instance from bag using soft selection (differentiable).
        
        Args:
            instance_features: (B, N, D) where B=batch, N=instances, D=feature_dim
            temperature: Optional temperature override (if None, uses self.temperature)
        
        Returns:
            critical_feature: (B, D) - weighted combination of instance features
            scores: (B, N) - importance scores for all instances
            critical_idx: (B,) - index of highest-scored instance (for interpretability)
        """
        B, N, D = instance_features.shape
        
        # Compute importance scores for each instance
        scores = self.scorer(instance_features)  # (B, N, 1)
        scores = scores.squeeze(-1)  # (B, N)
        
        # Use provided temperature or default
        temp = temperature if temperature is not None else self.temperature
        
        # Soft selection: weighted combination (differentiable)
        # Higher temperature = softer distribution (more uniform)
        # Lower temperature = sharper distribution (more focused)
        weights = F.softmax(scores / temp, dim=1)  # (B, N)
        critical_feature = torch.sum(weights.unsqueeze(-1) * instance_features, dim=1)  # (B, D)
        
        # Track argmax for interpretability (not used in gradient computation)
        critical_idx = torch.argmax(scores, dim=1)  # (B,)
        
        return critical_feature, scores, critical_idx


class ContextualAggregator(nn.Module):
    """
    Stream 2: Aggregates contextual information from all instances.
    
    Uses attention mechanism conditioned on the critical instance.
    Attention weights indicate which instances support the critical instance.
    """
    
    def __init__(
        self,
        feature_dim: int,
        attention_type: str = 'gated',  # 'gated' or 'cosine'
        hidden_dim: int = 128
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_type = attention_type
        self.hidden_dim = hidden_dim
        
        if attention_type == 'gated':
            # Gated attention: learned similarity
            self.query_proj = nn.Linear(feature_dim, hidden_dim)
            self.key_proj = nn.Linear(feature_dim, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        elif attention_type == 'cosine':
            # Cosine similarity-based attention
            self.query_proj = nn.Linear(feature_dim, feature_dim)
            self.key_proj = nn.Linear(feature_dim, feature_dim)
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")
    
    def forward(
        self,
        instance_features: torch.Tensor,
        critical_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate contextual information with attention.
        
        Args:
            instance_features: (B, N, D) - all instance features
            critical_feature: (B, D) - critical instance feature (query)
        
        Returns:
            context_feature: (B, D) - aggregated contextual feature
            attention_weights: (B, N) - attention weights (for interpretability)
        """
        B, N, D = instance_features.shape
        
        # Project query (critical instance) and keys (all instances)
        query = self.query_proj(critical_feature)  # (B, hidden_dim or D)
        keys = self.key_proj(instance_features)  # (B, N, hidden_dim or D)
        
        if self.attention_type == 'gated':
            # Gated attention: compute similarity and apply gate
            # Expand query to (B, 1, hidden_dim)
            query_expanded = query.unsqueeze(1)  # (B, 1, hidden_dim)
            
            # Compute similarity (dot product)
            similarity = torch.sum(query_expanded * keys, dim=-1)  # (B, N)
            
            # Apply gate
            gate_input = keys  # (B, N, hidden_dim)
            gate_weights = self.gate(gate_input).squeeze(-1)  # (B, N)
            
            # Combine similarity and gate
            attention_scores = similarity * gate_weights  # (B, N)
            attention_weights = F.softmax(attention_scores, dim=1)  # (B, N)
            
        else:  # cosine
            # Cosine similarity
            query_norm = F.normalize(query, p=2, dim=-1)  # (B, D)
            keys_norm = F.normalize(keys, p=2, dim=-1)  # (B, N, D)
            
            # Compute cosine similarity
            similarity = torch.sum(query_norm.unsqueeze(1) * keys_norm, dim=-1)  # (B, N)
            attention_weights = F.softmax(similarity, dim=1)  # (B, N)
        
        # Weighted aggregation
        context_feature = torch.sum(attention_weights.unsqueeze(-1) * instance_features, dim=1)  # (B, D)
        
        return context_feature, attention_weights


class DualStreamMIL(nn.Module):
    """
    Complete Dual-Stream MIL model for patient-level brain tumor classification.
    
    Architecture:
        1. Instance Encoder: Encodes each slice to feature vector
        2. Stream 1: Critical Instance Selector
        3. Stream 2: Contextual Aggregator
        4. Fusion: Combines critical and contextual features
        5. Classification Head: Patient-level prediction
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        instance_encoder_backbone: str = 'resnet18',
        instance_encoder_feature_dim: int = 512,
        instance_encoder_input_size: int = 224,
        attention_type: str = 'gated',
        fusion_method: str = 'concat',  # 'concat', 'weighted', 'cross_attn'
        dropout: float = 0.4,
        use_hidden_layer: bool = True,
        logger=None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.instance_encoder_backbone = instance_encoder_backbone
        self.fusion_method = fusion_method
        
        # Instance encoder
        self.instance_encoder = InstanceEncoder(
            backbone=instance_encoder_backbone,
            feature_dim=instance_encoder_feature_dim,
            input_size=instance_encoder_input_size,
            logger=logger
        )
        
        # Get actual feature dimension from encoder
        feature_dim = self.instance_encoder.feature_dim
        
        # Stream 1: Critical instance selector (always uses soft selection for stability)
        self.critical_selector = CriticalInstanceSelector(
            feature_dim=feature_dim,
            temperature=1.0  # Default temperature (can be adjusted during training)
        )
        
        # Stream 2: Contextual aggregator
        self.contextual_aggregator = ContextualAggregator(
            feature_dim=feature_dim,
            attention_type=attention_type
        )
        
        # Fusion
        if fusion_method == 'concat':
            fused_dim = feature_dim * 2
        elif fusion_method == 'weighted':
            fused_dim = feature_dim
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # Learnable α
        elif fusion_method == 'cross_attn':
            fused_dim = feature_dim
            # Simple cross-attention: context attends to critical
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=4,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")
        
        # Classification head
        if use_hidden_layer:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fused_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fused_dim, num_classes)
            )
        
        if logger:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"DualStreamMIL: {total_params/1e6:.2f}M total params, "
                       f"{trainable_params/1e6:.2f}M trainable")
            logger.info(f"Fusion method: {fusion_method}, Classifier: {'Two-layer' if use_hidden_layer else 'Single-layer'}")
            logger.info(f"Selection mode: soft (differentiable, temperature-based)")
    
    def forward(
        self,
        bag_of_slices: torch.Tensor,
        return_interpretability: bool = False,
        temperature: Optional[float] = None
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Forward pass through Dual-Stream MIL.
        
        Args:
            bag_of_slices: (B, N, 4, H, W) where B=batch, N=instances, 4=modalities, H=height, W=width
            return_interpretability: If True, return interpretability info (scores, attention, critical_idx)
            temperature: Optional temperature for critical instance selection (for annealing)
        
        Returns:
            logits: (B, num_classes) - patient-level classification logits
            interpretability_info (optional): dict with scores, attention_weights, critical_idx, selection_weights
        """
        B, N, C, H, W = bag_of_slices.shape
        
        # 1. Encode all instances
        instance_features = self.instance_encoder(bag_of_slices)  # (B, N, D)
        
        # 2. Stream 1: Critical instance selection (soft selection with temperature)
        # Use provided temperature or default from selector
        temp = temperature if temperature is not None else self.critical_selector.temperature
        critical_feature, instance_scores, critical_idx = self.critical_selector(
            instance_features, temperature=temp
        )
        
        # Compute selection weights for interpretability (same as used in selector)
        selection_weights = F.softmax(instance_scores / temp, dim=1)  # (B, N)
        
        # 3. Stream 2: Contextual aggregation
        context_feature, attention_weights = self.contextual_aggregator(
            instance_features, critical_feature
        )
        
        # 4. Fusion
        if self.fusion_method == 'concat':
            fused_feature = torch.cat([critical_feature, context_feature], dim=1)  # (B, 2D)
        elif self.fusion_method == 'weighted':
            alpha = torch.sigmoid(self.fusion_weight)
            fused_feature = alpha * critical_feature + (1 - alpha) * context_feature
        else:  # cross_attn
            # Cross-attention: context attends to critical
            critical_expanded = critical_feature.unsqueeze(1)  # (B, 1, D)
            context_expanded = context_feature.unsqueeze(1)  # (B, 1, D)
            fused_feature, _ = self.cross_attn(
                query=context_expanded,
                key=critical_expanded,
                value=critical_expanded
            )
            fused_feature = fused_feature.squeeze(1)  # (B, D)
        
        # 5. Classification
        logits = self.classifier(fused_feature)  # (B, num_classes)
        
        if return_interpretability:
            interpretability_info = {
                'instance_scores': instance_scores,  # (B, N)
                'selection_weights': selection_weights,  # (B, N) - soft selection weights
                'attention_weights': attention_weights,  # (B, N) - contextual attention weights
                'critical_idx': critical_idx,  # (B,) - highest-scored instance index
                'critical_feature': critical_feature,  # (B, D)
                'context_feature': context_feature  # (B, D)
            }
            return logits, interpretability_info
        
        return logits


def create_dual_stream_mil(
    num_classes: int = 2,
    instance_encoder_backbone: str = 'resnet18',
    instance_encoder_input_size: int = 224,
    attention_type: str = 'gated',
    fusion_method: str = 'concat',
    dropout: float = 0.4,
    use_hidden_layer: bool = True,
    logger=None
) -> DualStreamMIL:
    """
    Factory function to create DualStreamMIL model.
    
    Args:
        num_classes: Number of output classes (default: 2 for binary)
        instance_encoder_backbone: Backbone for instance encoder ('resnet18' or 'efficientnet_b0')
        instance_encoder_input_size: Input size for instance encoder (default: 224)
        attention_type: 'gated' or 'cosine' attention for contextual aggregation
        fusion_method: 'concat', 'weighted', or 'cross_attn'
        dropout: Dropout rate in classification head
        use_hidden_layer: Whether to use hidden layer in classifier
        logger: Optional logger for model info
    
    Returns:
        DualStreamMIL model instance
    
    Note:
        Critical instance selection always uses soft selection (differentiable) with temperature.
        Temperature can be adjusted during training for curriculum learning.
    """
    model = DualStreamMIL(
        num_classes=num_classes,
        instance_encoder_backbone=instance_encoder_backbone,
        instance_encoder_input_size=instance_encoder_input_size,
        attention_type=attention_type,
        fusion_method=fusion_method,
        dropout=dropout,
        use_hidden_layer=use_hidden_layer,
        logger=logger
    )
    return model


if __name__ == "__main__":
    # Test model
    model = create_dual_stream_mil()
    
    # Test input: batch of 2 patients, each with 64 slices, 4 modalities, 128x128
    test_input = torch.randn(2, 64, 4, 128, 128)
    
    # Forward pass
    logits, interpretability = model(test_input, return_interpretability=True)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Critical instance indices: {interpretability['critical_idx']}")
    print(f"Instance scores shape: {interpretability['instance_scores'].shape}")
    print(f"Attention weights shape: {interpretability['attention_weights'].shape}")


"""
Dual-Stream Multiple Instance Learning (MIL) Model

This module implements a Dual-Stream MIL architecture for brain tumor classification:
- Stream 1: Max-pooling aggregation
- Stream 2: Gated Attention MIL aggregation
- Fusion: Concatenated features from both streams
- Classifier: Binary classification head (HGG vs LGG)

The instance encoder is a ResNet18 adapted for 1-channel 2D slices.

Author: Medical Imaging Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18InstanceEncoder(nn.Module):
    """
    ResNet18-based encoder for 2D slice instances.
    
    Adapted for 1-channel input (grayscale MRI slices).
    """
    
    def __init__(self, pretrained=False):
        super().__init__()
        # Load pretrained ResNet18
        resnet18 = models.resnet18(pretrained=pretrained)
        
        # Replace first conv layer to accept 1 channel instead of 3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        
        # Copy remaining layers
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        
        # Initialize first conv layer weights
        if pretrained:
            # Initialize with mean of RGB channels
            self.conv1.weight.data = resnet18.conv1.weight.data.mean(dim=1, keepdim=True)
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Feature dimension after ResNet18
        self.feature_dim = 512
    
    def forward(self, x):
        """
        Forward pass through ResNet18 encoder.
        
        Args:
            x: Input tensor of shape (batch_size, 1, H, W) or (batch_size, H, W)
        
        Returns:
            Features of shape (batch_size, 512)
        """
        # Handle input shape: add channel dimension if missing
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        return x


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention mechanism for MIL aggregation.
    
    Based on: Ilse et al., "Attention-based Deep Multiple Instance Learning"
    """
    
    def __init__(self, feature_dim, attention_dim=128):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attention_dim, 1)
    
    def forward(self, features):
        """
        Compute attention-weighted aggregation.
        
        Args:
            features: Tensor of shape (batch_size, num_instances, feature_dim)
        
        Returns:
            aggregated: Aggregated features (batch_size, feature_dim)
            attention: Attention weights (batch_size, num_instances)
        """
        # Compute attention
        V = self.attention_V(features)  # (B, N, attention_dim)
        U = self.attention_U(features)  # (B, N, attention_dim)
        attention_scores = self.attention_weights(V * U).squeeze(-1)  # (B, N)
        attention = torch.softmax(attention_scores, dim=1)  # (B, N)
        
        # Weighted aggregation
        aggregated = torch.sum(attention.unsqueeze(-1) * features, dim=1)  # (B, feature_dim)
        
        return aggregated, attention


class DualStreamMIL(nn.Module):
    """
    Dual-Stream Multiple Instance Learning model.
    
    Architecture:
    - Instance Encoder: ResNet18 (1-channel adapted)
    - Stream 1: Max-pooling aggregation
    - Stream 2: Gated Attention MIL aggregation
    - Fusion: Concatenated features
    - Classifier: Binary classification head
    """
    
    def __init__(self, num_classes=2, pretrained_encoder=False, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        
        # Instance encoder
        self.encoder = ResNet18InstanceEncoder(pretrained=pretrained_encoder)
        feature_dim = self.encoder.feature_dim  # 512
        
        # Stream 1: Max pooling (identity - handled in forward)
        # Stream 2: Gated Attention
        self.gated_attention = GatedAttentionMIL(feature_dim, attention_dim=128)
        
        # Fusion: Concatenate both streams
        fused_dim = feature_dim * 2  # max pool + attention
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, bag, return_attention=False):
        """
        Forward pass through Dual-Stream MIL.
        
        Args:
            bag: Tensor of shape (batch_size, num_slices, H, W) or (batch_size, num_slices, 1, H, W)
                 Bag of slice instances
            return_attention: If True, return attention weights
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
            attention: Attention weights (batch_size, num_slices) if return_attention=True
        """
        batch_size, num_slices = bag.shape[0], bag.shape[1]
        
        # Reshape bag for encoder: (B*N, H, W) or (B*N, 1, H, W)
        if bag.dim() == 4:
            # (B, N, H, W) -> (B*N, 1, H, W)
            bag_flat = bag.view(batch_size * num_slices, 1, bag.shape[2], bag.shape[3])
        elif bag.dim() == 5:
            # (B, N, 1, H, W) -> (B*N, 1, H, W)
            bag_flat = bag.view(batch_size * num_slices, bag.shape[2], bag.shape[3], bag.shape[4])
        else:
            raise ValueError(f"Unexpected bag shape: {bag.shape}")
        
        # Encode all instances
        instance_features = self.encoder(bag_flat)  # (B*N, feature_dim)
        
        # Reshape back: (B, N, feature_dim)
        instance_features = instance_features.view(batch_size, num_slices, -1)
        
        # Stream 1: Max pooling
        stream1 = torch.max(instance_features, dim=1)[0]  # (B, feature_dim)
        
        # Stream 2: Gated Attention
        stream2, attention = self.gated_attention(instance_features)  # (B, feature_dim), (B, N)
        
        # Fusion: Concatenate
        fused = torch.cat([stream1, stream2], dim=1)  # (B, 2*feature_dim)
        
        # Classification
        logits = self.classifier(fused)  # (B, num_classes)
        
        if return_attention:
            return logits, attention
        else:
            return logits


# Factory function for convenience
def create_dual_stream_mil(num_classes=2, pretrained_encoder=False, dropout=0.5):
    """
    Create a Dual-Stream MIL model.
    
    Args:
        num_classes: Number of classes (default: 2 for HGG vs LGG)
        pretrained_encoder: Use ImageNet pretrained weights (default: False)
        dropout: Dropout rate (default: 0.5)
    
    Returns:
        DualStreamMIL model
    """
    return DualStreamMIL(
        num_classes=num_classes,
        pretrained_encoder=pretrained_encoder,
        dropout=dropout
    )


# Example usage and testing
if __name__ == "__main__":
    """
    Test Dual-Stream MIL model.
    """
    print("Testing Dual-Stream MIL Model")
    print("=" * 60)
    
    # Create model
    model = create_dual_stream_mil(num_classes=2, pretrained_encoder=False)
    print(f"Model created: {model.__class__.__name__}")
    
    # Test forward pass
    batch_size = 2
    num_slices = 16  # top-k slices
    H, W = 128, 128
    
    # Simulate bag of slices
    bag = torch.randn(batch_size, num_slices, 1, H, W)
    
    print(f"\nInput shape: {bag.shape}")
    
    # Forward pass
    logits, attention = model(bag, return_attention=True)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape: {attention.shape}")
    
    # Verify attention sums to 1
    attention_sum = attention.sum(dim=1)
    print(f"Attention sum (should be ~1.0): {attention_sum}")
    
    print("\n" + "=" * 60)
    print("Model test passed!")


"""
ResNet50-3D Model for Brain Tumor Classification

This module implements ResNet50-3D architecture adapted for 3D medical imaging,
with support for loading MedicalNet pretrained weights.

MedicalNet: https://github.com/Tencent/MedicalNet

Note: We implement ResNet50-3D compatible with MedicalNet's architecture,
as torchvision.models.video doesn't include r3d_50.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def adapt_conv1_weights_for_multimodal(pretrained_conv1_weight: torch.Tensor, target_channels: int, method: str = 'mean') -> torch.Tensor:
    """
    Adapt pretrained single-channel conv1 weights for multi-channel input.
    
    MedicalNet pretrained weights have conv1 with shape (64, 1, 7, 7, 7).
    For multi-modality (4 channels), we need shape (64, 4, 7, 7, 7).
    
    Methods:
    - 'mean': Replicate single channel and average (preserves pretrained features)
    - 'replicate': Copy single channel to all channels (simple replication)
    - 'kaiming': Initialize new channels with Kaiming init (less pretrained benefit)
    
    Args:
        pretrained_conv1_weight: Original conv1 weight tensor (out_channels, 1, k, k, k)
        target_channels: Target number of input channels (e.g., 4 for multi-modality)
        method: Adaptation method ('mean', 'replicate', or 'kaiming')
    
    Returns:
        Adapted conv1 weight tensor (out_channels, target_channels, k, k, k)
    """
    out_channels = pretrained_conv1_weight.shape[0]
    kernel_size = pretrained_conv1_weight.shape[2:]  # (7, 7, 7)
    
    if method == 'mean':
        # Replicate single channel to all channels, then average
        # This preserves the pretrained feature extraction pattern
        # Each modality channel gets the same learned filters
        expanded = pretrained_conv1_weight.repeat(1, target_channels, 1, 1, 1)
        # Average across channels to normalize (each channel gets 1/target_channels of the original)
        adapted_weight = expanded / target_channels
    elif method == 'replicate':
        # Simply replicate the single channel to all channels
        adapted_weight = pretrained_conv1_weight.repeat(1, target_channels, 1, 1, 1)
    elif method == 'kaiming':
        # Initialize new channels with Kaiming, keep first channel from pretrained
        adapted_weight = torch.zeros(out_channels, target_channels, *kernel_size)
        adapted_weight[:, 0:1, :, :, :] = pretrained_conv1_weight
        # Initialize remaining channels with Kaiming
        nn.init.kaiming_normal_(adapted_weight[:, 1:, :, :, :], mode='fan_out', nonlinearity='relu')
    else:
        raise ValueError(f"Unknown adaptation method: {method}")
    
    return adapted_weight


def load_medicalnet_pretrained(model: nn.Module, pretrained_path: Optional[str] = None, logger=None, adapt_multimodal: bool = False):
    """
    Load MedicalNet pretrained weights for ResNet50-3D.
    
    MedicalNet provides pretrained weights trained on 23 diverse medical datasets.
    If pretrained_path is None, the model will be initialized with random weights.
    
    Supports different MedicalNet checkpoint formats:
    - state_dict: Direct state dictionary
    - DataParallel: Checkpoints with 'module.' prefix (automatically stripped)
    - Partial loading: Ignores mismatched classification head weights
    - Multi-modal adaptation: Adapts conv1 weights for 4-channel input
    
    Args:
        model: ResNet50-3D model instance
        pretrained_path: Path to MedicalNet pretrained weights (.pth file)
                        If None, model uses random initialization
        logger: Optional logger for detailed logging
        adapt_multimodal: If True, adapt conv1 weights for multi-channel input (default: False)
    
    Returns:
        model: Model with loaded weights (or original if pretrained_path is None)
    """
    if pretrained_path is None:
        msg = "No pretrained path provided. Using random initialization."
        if logger:
            logger.info(msg)
        else:
            print(f"Warning: {msg}")
        return model
    
    log_func = logger.info if logger else print
    
    try:
        # PyTorch ≥ 2.6 requires weights_only=False for loading pretrained weights
        # This is safe for MedicalNet checkpoints from trusted source
        log_func(f"Loading MedicalNet pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            log_func("Found 'state_dict' key in checkpoint")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            log_func("Found 'model' key in checkpoint")
        else:
            state_dict = checkpoint
            log_func("Using checkpoint as direct state_dict")
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        module_prefix_count = 0
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # Remove 'module.' prefix
                module_prefix_count += 1
            else:
                name = k
            new_state_dict[name] = v
        
        if module_prefix_count > 0:
            log_func(f"Stripped 'module.' prefix from {module_prefix_count} keys (DataParallel format)")
        
        # Adapt conv1 weights for multi-modal input if needed
        if adapt_multimodal:
            # Check if model has multi-channel input
            model_conv1_key = None
            pretrained_conv1_key = None
            
            # Find conv1 keys
            for key in new_state_dict.keys():
                if 'conv1.weight' in key:
                    pretrained_conv1_key = key
                    break
            
            # Check model's conv1 shape
            for name, param in model.named_parameters():
                if 'conv1' in name and 'weight' in name:
                    model_conv1_key = name
                    model_in_channels = param.shape[1]
                    pretrained_in_channels = new_state_dict[pretrained_conv1_key].shape[1] if pretrained_conv1_key else 1
                    
                    if model_in_channels != pretrained_in_channels:
                        log_func(f"Adapting conv1 weights: {pretrained_in_channels} -> {model_in_channels} channels")
                        log_func(f"  Model conv1 shape: {param.shape}")
                        log_func(f"  Pretrained conv1 shape: {new_state_dict[pretrained_conv1_key].shape}")
                        
                        # Adapt weights using mean method (preserves pretrained features)
                        adapted_weight = adapt_conv1_weights_for_multimodal(
                            new_state_dict[pretrained_conv1_key],
                            target_channels=model_in_channels,
                            method='mean'
                        )
                        
                        # Update state dict with adapted weights
                        new_state_dict[pretrained_conv1_key] = adapted_weight
                        log_func(f"  Adapted conv1 shape: {adapted_weight.shape}")
                        log_func("  Method: Mean replication (preserves pretrained feature patterns)")
                        break
        
        # Load weights, allowing for partial loading (e.g., if classification head differs)
        # This safely ignores mismatched classification head weights
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            log_func(f"Missing keys (expected for classification head): {len(missing_keys)} keys")
        if unexpected_keys:
            log_func(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
        
        # Count successfully loaded keys
        loaded_keys = len(new_state_dict) - len(missing_keys)
        total_keys = len(new_state_dict)
        log_func(f"Successfully loaded {loaded_keys}/{total_keys} pretrained weights from MedicalNet")
        log_func("MedicalNet pretrained weights loaded successfully!")
        
    except FileNotFoundError:
        msg = f"Pretrained weights file not found: {pretrained_path}"
        if logger:
            logger.warning(msg)
        else:
            print(f"Warning: {msg}")
        print("Using random initialization instead.")
    except Exception as e:
        msg = f"Could not load pretrained weights from {pretrained_path}: {e}"
        if logger:
            logger.warning(msg)
        else:
            print(f"Warning: {msg}")
        print("Using random initialization instead.")
    
    return model


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock3D(nn.Module):
    """Basic 3D ResNet block"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    """Bottleneck 3D ResNet block"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """3D ResNet architecture compatible with MedicalNet"""
    
    def __init__(self, block, layers, in_channels=1, num_classes=2, dropout=0.5):
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 * block.expansion, num_classes)
        )

        # Initialize weights with improved initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def create_resnet50_3d(
    num_classes: int = 2,
    in_channels: int = 1,
    pretrained_path: Optional[str] = None,
    dropout: float = 0.5,
    logger=None,
    adapt_multimodal: bool = False
) -> nn.Module:
    """
    Create ResNet50-3D model for binary classification (LGG vs HGG).
    
    This implementation is compatible with MedicalNet's ResNet50-3D architecture.
    Supports both single-modality (1 channel) and multi-modality (4 channels) inputs.
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        in_channels: Number of input channels (default: 1 for single modality, 4 for multi-modality)
        pretrained_path: Path to MedicalNet pretrained weights (optional)
        dropout: Dropout rate for classification head (default: 0.5)
        logger: Optional logger for detailed logging
        adapt_multimodal: If True, adapt pretrained conv1 weights for multi-channel input (default: False)
    
    Returns:
        ResNet50-3D model with binary classification head
    """
    # ResNet50 uses Bottleneck3D blocks with [3, 4, 6, 3] layers
    model = ResNet3D(
        block=Bottleneck3D,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # Load MedicalNet pretrained weights if provided
    if pretrained_path is not None:
        # Automatically detect if multi-modal adaptation is needed
        if in_channels > 1:
            adapt_multimodal = True
            if logger:
                logger.info(f"Multi-modal input detected ({in_channels} channels). Will adapt pretrained conv1 weights.")
        
        model = load_medicalnet_pretrained(
            model, 
            pretrained_path, 
            logger=logger,
            adapt_multimodal=adapt_multimodal
        )
    
    return model


class ResNet50_3D(nn.Module):
    """
    ResNet50-3D model wrapper for brain tumor classification.
    
    This class provides a convenient interface for ResNet50-3D with:
    - MedicalNet pretrained weights support
    - Binary classification head
    - Dropout regularization
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained_path: Optional[str] = None,
        dropout: float = 0.5,
        logger=None,
        adapt_multimodal: bool = False
    ):
        super().__init__()
        self.model = create_resnet50_3d(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            dropout=dropout,
            logger=logger,
            adapt_multimodal=adapt_multimodal
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, D, H, W)
               Expected shape: (B, 1, 128, 128, 128) for BraTS 2018
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def get_backbone_params(self):
        """
        Get parameters from the pretrained backbone (all layers except classifier).
        
        Returns:
            Generator of backbone parameters
        """
        for name, param in self.model.named_parameters():
            if 'fc' not in name:  # Exclude classification head
                yield param
    
    def get_classifier_params(self):
        """
        Get parameters from the classification head (fc layer).
        
        Returns:
            Generator of classifier parameters
        """
        for name, param in self.model.named_parameters():
            if 'fc' in name:  # Only classification head
                yield param
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor before classification head
        """
        # Forward through all layers except classification head
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def state_dict(self):
        """Return model state dict."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        """Load model state dict."""
        return self.model.load_state_dict(state_dict, strict=strict)


if __name__ == "__main__":
    """
    Test ResNet50-3D model creation.
    """
    print("Testing ResNet50-3D Model")
    print("=" * 60)
    
    # Create model
    model = ResNet50_3D(
        num_classes=2,
        in_channels=1,
        pretrained_path=None,  # No pretrained weights for testing
        dropout=0.5
    )
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 128, 128, 128)
    
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
        features = model.get_features(input_tensor)
    
    print(f"Output logits shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("ResNet50-3D model test passed!")


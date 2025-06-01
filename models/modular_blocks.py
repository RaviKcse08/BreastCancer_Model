# models/modular_blocks.py

import torch
import torch.nn as nn
from torchvision.models import resnet, densenet

# Example ResNet Basic Block (similar to torchvision's)
class ResNetBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, width_factor=1):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * width_factor, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * width_factor)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels * width_factor, out_channels * width_factor, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * width_factor)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# Example DenseNet Block (simplified, typically uses _DenseBlock and _Transition)
class DenseNetBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers_in_block, width_factor=1):
        super(DenseNetBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_layers_in_block):
            self.layers.append(self._make_dense_layer(current_channels, growth_rate * width_factor))
            current_channels += growth_rate * width_factor

    def _make_dense_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1, bias=False), # Bottleneck
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1)) # Concatenate all previous features
            features.append(new_features)
        return torch.cat(features, 1) # Output the concatenation of all features


# Define a mapping for block types (used by NAS encoding)
BLOCK_MODULES = {
    'ResNetBlock': ResNetBasicBlock,
    'DenseNetBlock': DenseNetBlock,
    # Add more block types as needed
}
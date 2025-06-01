# models/dynamic_nas_model.py

import torch.nn as nn
from models.modular_blocks import BLOCK_MODULES
from config import settings

class DynamicNASModel(nn.Module):
    def __init__(self, architecture_genotype, num_classes, width_factor=1):
        super(DynamicNASModel, self).__init__()
        self.architecture_genotype = architecture_genotype # e.g., [0, 1, 0] for [ResNet, DenseNet, ResNet]
        self.num_classes = num_classes
        self.width_factor = width_factor # How much to scale block widths (from HPO)

        # Initial convolutional layer (common for most vision models)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        current_channels = 64 # Output channels of initial_conv

        self.blocks = nn.ModuleList()
        # You'll need a more sophisticated way to handle channel progression
        # for different block types and their connections. This is simplified.
        # Example: Let's assume output channels double after each main block for simplicity
        # or have a fixed progression.

        for block_idx, block_type_id in enumerate(architecture_genotype):
            block_name = settings.NAS_BLOCK_MAPPING[block_type_id]
            block_class = BLOCK_MODULES[block_name]

            # Adjust channels for next block based on block type
            if block_name == 'ResNetBlock':
                # Example: progressively increase channels
                out_channels = current_channels * 2 if block_idx > 0 else current_channels # Initial expansion
                # Downsample if needed (e.g., if image size reduces)
                downsample = None
                if current_channels != out_channels * self.width_factor: # For skip connection matching
                    downsample = nn.Sequential(
                        nn.Conv2d(current_channels, out_channels * self.width_factor, kernel_size=1, stride=1, bias=False), # Assuming stride 1 for simplicity
                        nn.BatchNorm2d(out_channels * self.width_factor)
                    )
                self.blocks.append(block_class(current_channels, out_channels // self.width_factor, stride=1, downsample=downsample, width_factor=self.width_factor))
                current_channels = out_channels * self.width_factor
            elif block_name == 'DenseNetBlock':
                # For DenseNet, need to define growth_rate and num_layers_in_block
                growth_rate = 32
                num_layers_in_block = 6 # Example: 6 sub-layers within a dense block
                self.blocks.append(block_class(current_channels, growth_rate, num_layers_in_block, width_factor=self.width_factor))
                current_channels += growth_rate * num_layers_in_block * self.width_factor # Features accumulate
            # Add conditions for other block types

        # Final pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(current_channels, num_classes) # Final FC layer

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
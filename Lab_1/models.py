import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------
# 1. Simple MLP
# -----------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, classes, normalization):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if normalization:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if normalization:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], classes))
        self.model = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x):
        return self.model(x)


# -----------------
# 2. ResMLP
# -----------------
class ResMLPBlock(nn.Module):
    def __init__(self, hidden_dim, normalization):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if normalization:
            self.model = nn.Sequential(
                self.fc1,
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                self.fc2,
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
        else:
            self.model = nn.Sequential(
                self.fc1,
                nn.ReLU(),
                self.fc2,
                nn.ReLU(),
            )

    def forward(self, x):
        return self.model(x) + x

class ResMLP(nn.Module):
    def __init__(self, input_size, hidden_dim, num_blocks, classes, normalization):
        super().__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_dim)
        self.blocks = nn.Sequential(*[ResMLPBlock(hidden_dim, normalization) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, classes)

        self.model = nn.Sequential(
            nn.Flatten(),
            self.input_layer,
            nn.ReLU(),
            self.blocks,
            self.fc_out,
        )

    def forward(self, x):
        return self.model(x)


# -----------------
# 3. CNN
# -----------------

class CNN(nn.Module):
    def __init__(self, block_type, layers, use_residual):
        super(CNN, self).__init__()
        self.layers = layers
        self.use_residual = use_residual

        if block_type == "basic":
            self.block = self._make_basic_block
        elif block_type == "bottleneck":
            self.block = self._make_bottleneck_block
        else:
            raise ValueError("Unknown block type")

        self.conv_layers = self._make_conv_layers()
        self.fc = nn.Linear(512, 10)  # CIFAR-10: 10 classes

    def _make_conv_layers(self):
        layers = []
        in_channels = 3
        for i in range(self.layers):
            layers.append(self.block(in_channels))
            in_channels = 64 * 2 ** i
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def _make_basic_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def _make_bottleneck_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
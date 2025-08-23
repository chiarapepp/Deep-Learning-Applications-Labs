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
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_residual=True):
        super(BasicBlock, self).__init__()
        self.use_residual = use_residual
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

        # Skip connection handling
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_residual:
            residual = self.skip_connection(residual)
            out += residual

        out = self.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self, layers, num_classes, use_residual=False, base_channels=32):
        super(CNN, self).__init__()
        self.use_residual = use_residual
        self.base_channels = base_channels
        self.relu = nn.ReLU(inplace=True)

        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Build layers
        self.in_channels = base_channels
        self.layer1 = self._make_layer(BasicBlock, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_channels * 8, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        
        # First block may have stride > 1
        layers.append(block(self.in_channels, out_channels, stride, self.use_residual))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1, use_residual=self.use_residual))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
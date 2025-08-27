import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch import Tensor


'''
1. Simple MLP

Args:
    - input_size (int): The number of input features.
    - hidden_sizes (list of int): The number of units in each hidden layer (example [128, 64]).
    - classes (int): The number of output classes.
    - normalization (bool): Whether to use batch normalization.

'''
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


'''
2. ResMLP

Args:
    - input_size (int): The number of input features.
    - hidden_dim (int): The number of hidden units.
    - num_blocks (int): The number of residual blocks.
    - classes (int): The number of output classes.
    - normalization (bool): Whether to use batch normalization.

'''

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


'''
3. CNN

The exercise allowed using the ResNet building blocks from `torchvision.models.resnet`, so I implemented my 
own ResNet class based on these blocks, introducing a few modifications. 
The main change is the option to disable skip connections, which allows experiments comparing the network 
with and without them. 
Additional methods were added to facilitate tasks such as feature extraction and fine-tuning.

Args:
    - layers (List[int]): Number of blocks in each of the four layers.
    - classes (int): Number of output classes.
    - use_residual (bool): Whether to use residual (skip) connections.
'''

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    # Utility: 3x3 convolution with automatic padding.
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,  # Ensures that that the convolution does not change the spatial dimensions of the input.
        groups=groups,    
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    # Utility: 1x1 convolution with optional stride.
    # Useful for adjusting the number of channels or for projection in residual connections.
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1 # Factor by which channels change (1 for BasicBlock, >1 for Bottleneck (e.g. 4)).

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,         # not used.
            base_width: int = 64,     # not used (is used in Bottleneck).
            dilation: int = 1,
            use_residual: bool = True
    ) -> None:
        super().__init__()
        self.use_residual = use_residual

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # If not None, transforms the input (identity) to make it compatible with the output of the block
        self.downsample = downsample 
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_residual:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        out = self.relu(out)
        return out


class CNN(nn.Module):
    def __init__(
            self,
            layers: List[int],
            classes: int = 10,
            use_residual: bool = True
    ) -> None:
        super().__init__()

        self.use_residual = use_residual
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], use_residual=self.use_residual)
        self.layer2 = self._make_layer(128, layers[1], stride=2, use_residual=self.use_residual)
        self.layer3 = self._make_layer(256, layers[2], stride=2, use_residual=self.use_residual)
        self.layer4 = self._make_layer(512, layers[3], stride=2, use_residual=self.use_residual)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
            self,
            planes: int,
            blocks: int,
            stride: int = 1,
            use_residual: bool = True
    ) -> nn.Sequential:

        downsample = None
        self.use_residual = use_residual

        if stride != 1 or self.inplanes != planes :
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(
            BasicBlock(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, self.dilation,
                use_residual=self.use_residual
            )
        )
        self.inplanes = planes 
        for _ in range(1, blocks):
            layers.append(
                # Every block here has stride=1 and downsample=None
                BasicBlock(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    use_residual=self.use_residual
                )
            )

        return nn.Sequential(*layers)
    
    # Extract features from the network (before the final classification layer).
    # Returns the 512-dimensional feature vector after global average pooling.
    def get_features(self, x: Tensor) -> Tensor:
            
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
        
        return x  

    def forward(self, x: Tensor) -> Tensor:

        features = self.get_features(x)
        x = self.fc(features)

        return x

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
# 3. Simple CNN
# -----------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # CIFAR-10: 32x32 → after 2 pools = 8x8
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32→16
        x = self.pool(F.relu(self.conv2(x)))  # 16→8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def get_model(name, dataset="MNIST", **kwargs):
    """
    Factory function to get model by name.
    Args:
        name (str): 'mlp', 'resmlp', 'cnn'
        dataset (str): 'MNIST', 'CIFAR10', 'CIFAR100' (influences num_classes)
    """
    classes = 10 if dataset in ["MNIST", "CIFAR10"] else 100

    if name.lower() == "mlp":
        return MLP(classes=classes, **kwargs)
    elif name.lower() == "resmlp":
        return ResMLP(classes=classes, **kwargs)
    elif name.lower() == "cnn":
        return SimpleCNN(classes=classes)
    else:
        raise ValueError(f"Model {name} not recognized.")
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------
# 1. MLP per MNIST
# -----------------
class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=[256, 128], num_classes=10):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)


# -----------------
# 2. ResMLP
# -----------------
class ResMLPBlock(nn.Module):
    def __init__(self, dim):
        super(ResMLPBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out


class ResMLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_dim=256, num_blocks=2, num_classes=10):
        super(ResMLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_dim)
        self.blocks = nn.Sequential(*[ResMLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.fc_out(x)
        return x


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


# -----------------
# 4. Utility: funzione costruttore modelli
# -----------------
def get_model(name, dataset="MNIST", **kwargs):
    """
    Factory function to get model by name.
    Args:
        name (str): 'mlp', 'resmlp', 'cnn'
        dataset (str): 'MNIST', 'CIFAR10', 'CIFAR100' (influences num_classes)
    """
    num_classes = 10 if dataset in ["MNIST", "CIFAR10"] else 100

    if name.lower() == "mlp":
        return MLP(num_classes=num_classes, **kwargs)
    elif name.lower() == "resmlp":
        return ResMLP(num_classes=num_classes, **kwargs)
    elif name.lower() == "cnn":
        return SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Model {name} not recognized.")
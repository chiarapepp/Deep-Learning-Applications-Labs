import torch
import torch.nn as nn
import torch.nn.functional as F

"""
PolicyNetwork and ValueNetwork, defined to be independent of the environment.

Args:
    env: The Gym environment used to determine input/output dimensions.
    num_layers (int): The number of hidden layers in the network.
    hidden_dim (int): Number of neurons in each hidden layer.
"""

# Oss env.observation_space.shape[0] is the dimension of the observation space.
# Oss env.action_space.n is the number of possible actions.

class PolicyNetwork(nn.Module):
    def __init__(self, env, num_layers=1, hidden_dim=128):
        super().__init__()
        hidden_layers = [nn.Linear(env.observation_space.shape[0], hidden_dim), nn.ReLU()]
        hidden_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (num_layers - 1)
        self.hidden = nn.Sequential(*hidden_layers)
        self.out = nn.Linear(hidden_dim, env.action_space.n)

    def forward(self, state, temperature=1.0):
        state = self.hidden(state)
        return F.softmax(self.out(state) / temperature, dim=-1) # Probabilities over the actions


class ValueNetwork(nn.Module):
    def __init__(self, env, num_layers=1, hidden_dim=128):
        super().__init__()

        hidden_layers = [nn.Linear(env.observation_space.shape[0], hidden_dim), nn.ReLU()]
        hidden_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (num_layers - 1)
        self.hidden = nn.Sequential(*hidden_layers)
        self.out = nn.Linear(hidden_dim, 1) # Single output for state value

    def forward(self, state):
        state = self.hidden(state)
        return self.out(state).squeeze(-1)

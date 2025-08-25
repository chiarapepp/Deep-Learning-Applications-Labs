import gymnasium as gym
import torch
from reinforce import reinforce
from models import PolicyNetwork
from utils import save_checkpoint, load_checkpoint

# Ambiente con e senza render
env_render = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1')

# Reset: restituisce una tupla (obs, info)
obs, info = env.reset()
print("Initial observation:", obs)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)




obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

policy = PolicyNetwork(obs_dim, act_dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

all_rewards, eval_rewards, eval_lengths = reinforce(
    env, policy, optimizer, num_episodes=1000, gamma=0.99,
    eval_every=50, eval_episodes=10
)

import matplotlib.pyplot as plt

plt.plot(all_rewards, label="Training rewards")
plt.plot(range(0, 1000, 50), eval_rewards, label="Eval avg rewards")
plt.legend()
plt.show()

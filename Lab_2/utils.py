import torch
from torch.distributions import Categorical
import numpy as np
import os
import imageio

'''
Utility function for model checkpointing.
'''
def save_checkpoint(name, model, opt, dir):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "opt_state_dict": opt.state_dict(),
        },
        os.path.join(dir, f"checkpoint-{name}.pt"),
    )

"""
Utility function to load a model checkpoint.

fname: The name of the checkpoint file.
model: The model to load the state_dict into.
opt: The optimizer to load the state_dict into (optional).

"""
def load_checkpoint(model, fname, opt=None):
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint["model_state_dict"])
    if opt:
        opt.load_state_dict(checkpoint["opt_state_dict"])
    return model

'''
This function takes in an environment, observation, policy, a temperature parameter and a 
flag for deterministic or stochastic action selection (sample from pi(a|obs)).

It returns the selected action, the log probability of that action (needed for policy gradient)
and the entropy of the action distribution.
'''
def select_action(env, obs, policy, temperature=1.0, deterministic=False):
    probs = policy(obs, temperature=temperature)

    if deterministic:
        action = torch.argmax(probs)
        log_prob = torch.log(probs[action])
        entropy = 0
    else:
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
    return (action.item(), log_prob.reshape(1), entropy)

 
"""
Function that computes the discounted total reward for a sequence of rewards.
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

Parameters:
    rewards (list or numpy array): List or array of rewards.
    gamma (float): Discount factor.

"""
def compute_returns(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns

"""
Function that given an environment and a policy, run it up to the maximum number of steps.

Parameters:
    env: The environment to run the episode in.
    policy: The policy used to select actions.
    max_steps (int): The maximum number of steps to run in the episode. Default is 1000.
    temperature (float): The temperature parameter for action selection. Default is 1.0.
    deterministic (bool): Whether to select actions deterministically or stochastically. Default is False.

Returns:
    observations (list): A list of observations throughout the episode.
    actions (list): A list of actions taken during the episode.
    log_probs (torch.Tensor): A tensor containing the log probabilities of the actions taken.
    rewards (list): A list of rewards received at each step.
    entropies (torch.Tensor): A tensor containing the entropies of the action distributions.
"""
def run_episode(env, policy, max_steps=1000, temperature=1.0, deterministic=False):
    
    observations = []
    actions = []
    log_probs = []
    rewards = []
    entropies = []
    
    # Reset the environment and start the episode.
    (obs, info) = env.reset()
    for i in range(max_steps):
        # Get the current observation, run the policy and select an action.
        obs = torch.tensor(obs, dtype=torch.float32)
        (action, log_prob, entropy) = select_action(env, obs, policy, temperature, deterministic)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        entropies.append(entropy)

        # Advance the episode by executing the selected action.
        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        # term : whether the episode was terminated (in the cartpole documentation this is said to happen 
        # when the pole is no longer upright (angle > +-12 degrees or cart position > +- 2.4)).
        # trunc : whether the episode was truncated (max_steps reached).
        if term or trunc:
            break
    return (
        observations,
        actions,
        torch.cat(log_probs),
        rewards,
        torch.stack(entropies),
    )

# Observation, cartpole truncation is set to 500 steps, while lunarlander is set to 1000, so that's why
# max_steps is set to 1000 by default.

"""
Function that evaluates a given policy in a specified environment.

Runs the policy for a given number of episodes.
Returns:
    avg_reward (float): Average total reward over all episodes.
    avg_length (float): Average length of episodes.
    std_reward (float): Standard deviation of total rewards over all episodes.
    std_length (float): Standard deviation of episode lengths over all episodes.
"""
def evaluate_policy(
    env, policy, episodes=5, max_steps=1000, temperature=1.0, deterministic=False
):
    policy.eval()
    total_rewards = []
    lengths = []

    for _ in range(episodes):
        obs, _ = env.reset()
        rewards = 0
        for t in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _, _ = select_action(
                    env,
                    obs_tensor,
                    policy,
                    temperature=temperature,
                    deterministic=deterministic,
                )
            obs, reward, term, trunc, _ = env.step(action)
            rewards += reward
            if term or trunc:
                break
        total_rewards.append(rewards)
        lengths.append(t + 1)

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)
    policy.train()
    return avg_reward, avg_length, std_reward, std_length

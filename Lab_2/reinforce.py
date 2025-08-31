import torch
import wandb
from utils import save_checkpoint, run_episode, compute_returns, evaluate_policy

"""
Implementation of the REINFORCE policy gradient algorithm.
Checkpoints best model at each iteration to the wandb run directory.

Args:
    policy: The policy network to be trained.
    env: The environment in which the agent operates.
    run: An object that handles logging and running episodes.
    value_network: The value network to be trained (if using as a baseline).
    gamma: The discount factor for future rewards.
    lr: Learning rate for the optimizer.
    baseline: The type of baseline to use ["none", "std", "value"].
    num_episodes: The number of episodes to train the policy.
    eval_interval: Evaluate the learned policy every N (eval_interval) iterations.
    eval_episodes: Number of episodes to evaluate the policy (some integer M).
    norm_advantages: If True, normalize the provided baseline into zero mean and 1 variance.
    clip_gradients: If True, clip the norm of the gradients of the policy and value networks.
    deterministic: If True, evaluate the learned policy with a deterministic policy sampler every eval_interval iterations.
    temperature: Softmax temperature for stochastic policy sampling.
    t_schedule: Temperature scheduler. Can be Linear or Exponential. Note that T_min=0.1 and decay_factor=0.999.
    entropy_coeff: Coefficient for entropy regularization. Defaults to 0.01.


Returns:
    running_rewards: A list of running rewards over episodes.
"""

def reinforce(
    policy,
    env,
    run,
    value_network=None,
    gamma=0.99,
    lr=1e-3,
    baseline="std",
    num_episodes=10,
    eval_interval=50,
    eval_episodes=20,
    norm_advantages=False, 
    clip_gradients=False,
    deterministic=False,
    temperature=1.0,
    t_schedule=None,
    entropy_coeff=0.01,
):

    # Initial temperature and minimum temperature for the scheduler.
    T_start = temperature
    T_min = 0.05

    if baseline not in ["none", "std", "value"]:
        raise ValueError(f"Unknown baseline {baseline}")
    if baseline == 'value':
        if value_network is None:
            raise ValueError("Value baseline selected, but value_network is None")
        value_network.train()
        value_opt = torch.optim.Adam(value_network.parameters(), lr=lr)

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    
    running_rewards = [0.0]
    eval_rewards = []
    eval_lengths = []
    det_rewards = []
    det_lengths = []

    # Track best evaluation return.
    best_eval_return = float("-inf")

    # Main training loop.
    policy.train()

    for episode in range(num_episodes):

        log = {}

        # Compute temperature based on the selected scheduler.
        if t_schedule is not None:
            if t_schedule == "linear":
                decay_rate = (T_start - T_min) / num_episodes
                T = max(T_min, T_start - decay_rate * episode)
            elif t_schedule == "exponential":
                # Decay factor of the exp scheduler is hard coded to 0.999.
                T = max(T_min, T_start * (0.999**episode))
        else:
            T = T_start
        log["temperature"] = T 

        # Run an episode of the environment, collect everything needed for policy update.
        (observations, actions, log_probs, rewards, entropies) = run_episode(
            env, policy
        )

        # Compute the discounted reward for every step of the episode.
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)
        obs_tensor = torch.stack(observations)

        # Keep a running exponential average of total discounted rewards for the whole episode.
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        log["episode_length"] = len(returns)
        log["return"] = returns[0]

        if baseline == "none":
            base_returns = returns
        elif baseline == "std":
            base_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        elif baseline == "value":
            values = value_network(obs_tensor)
            base_returns = returns - values.detach()

            # Value loss (MSE between predicted value and return).
            value_opt.zero_grad()
            value_loss = torch.nn.functional.mse_loss(values, returns)
            value_loss.backward()
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(value_network.parameters(), max_norm=1.0)
            value_opt.step()
            log["value_loss"] = value_loss.item()
            if norm_advantages:
                base_returns = (base_returns - base_returns.mean()) / (base_returns.std() + 1e-8)

        # Make an optimization step on the policy network.
        opt.zero_grad()
        policy_loss = (-log_probs * base_returns - entropy_coeff * entropies).mean()   # helps with stability 
        policy_loss.backward()
        if clip_gradients:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()

        # Log the current loss and finalize the log for this episode.
        log["policy_loss"] = policy_loss.item()

        # Print running reward and (optionally) render an episode after every 100 policy updates
        if not episode % 100:
            print(f"Running reward @ episode {episode}: {running_rewards[-1]}")

        # Evaluate the policy for eval_episodes every eval_interval episodes
        if episode % eval_interval == 0:
            avg_reward, avg_length, std_reward, std_length = evaluate_policy(
                env, policy, episodes=eval_episodes, temperature=T, deterministic=False
            )
            eval_rewards.append(avg_reward)
            eval_lengths.append(avg_length)
            log["eval_avg_reward"] = avg_reward
            log["eval_avg_length"] = avg_length
            print(
                f"[EVAL] Episode {episode} — Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}"
            )
            if deterministic:
                avg_det_reward, avg_det_length, std_det_reward, std_det_length = evaluate_policy(
                    env,
                    policy,
                    episodes=eval_episodes,
                    temperature=T,
                    deterministic=True,
                )
                log["avg_det_reward"] = avg_det_reward
                log["avg_det_length"] = avg_det_length
                log["std_det_reward"] = std_det_reward
                log["std_det_length"] = std_det_length

                det_rewards.append(avg_det_reward)
                det_lengths.append(avg_det_length)
                print(
                    f"[DET-EVAL] Episode {episode} — Avg Reward: {avg_det_reward:.2f}, Avg Length: {avg_det_length:.2f}"
                )

            # # Save checkpoint if current evaluation outperforms all previous evaluations
            if avg_reward > best_eval_return:
                best_eval_return = avg_reward
                save_checkpoint("best_eval_policy",  policy, opt, wandb.run.dir)
                if baseline == "value":
                    save_checkpoint("best_eval_value", value_network, value_opt, wandb.run.dir)

        run.log(log)

    # Return the running rewards.
    policy.eval()
    if value_network:
        value_network.eval()
    return running_rewards, eval_rewards, eval_lengths
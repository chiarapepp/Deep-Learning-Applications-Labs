import argparse
import wandb
import gymnasium as gym
import torch
from reinforce import reinforce
from models import PolicyNetwork, ValueNetwork
from utils import save_checkpoint, load_checkpoint, run_episode


def parse_args():
    """The argument parser for the main training script."""
    parser = argparse.ArgumentParser(
        description="A script implementing REINFORCE on the Cartpole and LunarLander environments."
    )
    parser.add_argument("--baseline", type=str, default="none", help="Baseline to use (none, std)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--visualize", action="store_true", help="Visualize final agent")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of hidden layers in the policy and value networks")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Width of the layers in the policy and value networks")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluate the policy every --eval-interval iterations")
    parser.add_argument("--eval_episodes", type=int, default=20, help="Evaluate the policy for --eval-episodes episodes")
    parser.add_argument("--normalize", action="store_true", help="If true, normalize G_t - b_t to zero mean and unit variance")
    parser.add_argument("--clip_grad", action="store_true", help="If true, clip gradients to unit norm for both the policy and the value networks")
    parser.add_argument("--det", action="store_true", help="Enable deterministic policy evaluation every --eval-interval iterations")
    parser.add_argument("--T", type=float, default=1.0, help="Softmax temperature for the policy. If a temperature scheduler is used, this will be the starting temperature")
    parser.add_argument("--t_schedule", choices=["linear", "exponential"], help="Choose between a linear or exponential temperature scheduler")
    parser.add_argument("--env", default="cartpole", choices=["cartpole", "lunarlander"], help="Choose between the Cartpole and the LunarLander environment")
    parser.set_defaults(visualize=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    seed = 10
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run = wandb.init(
        project="DLA_Lab_2",
        config={
            "environment": args.env,
            "learning_rate": args.lr,
            "baseline": args.baseline,
            "gamma": args.gamma,
            "num_episodes": args.episodes,
            "num_layers": args.num_layers,
            "hidden_dim": args.hidden_dim,
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes,
            "normalize": args.normalize,
            "clip_grad": args.clip_grad,
            "deterministic_eval": args.det,
            "temperature": args.T,
            "t_schedule": args.t_schedule,
        },
        name=f"REINFORCE_{args.env}_baseline={args.baseline}_{args.det}_T={args.T}_t_schedule={args.t_schedule}_gamma={args.gamma}",
    )

    # Instantiate the environment (no visualization)
    if args.env == "cartpole":
        env = gym.make("CartPole-v1")
    elif args.env == "lunarlander":
        env = gym.make("LunarLander-v3")

    # Make a policy network
    policy = PolicyNetwork(env, num_layers=args.num_layers, hidden_dim=args.hidden_dim)
    if args.baseline == "value":
        value_network = ValueNetwork(env, num_layers=args.num_layers, hidden_dim=args.hidden_dim)
    else:
        value_network = None
    # Train the agent.
    reinforce(
        policy,
        env,
        run,
        value_network,
        gamma=args.gamma,
        lr=args.lr,
        baseline=args.baseline,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        norm_advantages=args.normalize,
        clip_gradients=args.clip_grad,
        deterministic=args.det,
        temperature=args.T,
        t_schedule=args.t_schedule,
    )

    if args.visualize:
        if args.env == "cartpole":
            env_render = gym.make("CartPole-v1", render_mode="human")
        elif args.env == "lunarlander":
            env_render = gym.make("LunarLander-v3", render_mode="human")
        for _ in range(10):
            run_episode(env_render, policy)

        # Close the visualization environment.
        env_render.close()

    # Close the Cartpole environment and finish the wandb run.
    env.close()
    run.finish()
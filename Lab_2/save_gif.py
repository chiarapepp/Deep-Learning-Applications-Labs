import torch
import gymnasium as gym
import os
import argparse
from networks import PolicyNetwork 
import imageio
from utils import run_episode, load_checkpoint




def make_gif(env, policy, checkpoint, gif_path, temperature=1.0, deterministic=False, episodes=1, maxlen=500):

    frames = []
    policy = load_checkpoint(policy, checkpoint)

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < maxlen:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_probs = policy(obs_t)

                if deterministic:
                    action = torch.argmax(action_probs).item()
                else:
                    action = (torch.distributions.Categorical(action_probs / temperature)
                    .sample()
                    .item()
                    )

            frame = env.render()
            if frame is not None:
                frames.append(frame)
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            step += 1

    env.close()

    if frames:
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Saved GIF to {gif_path}")
    else:
        print("No frames captured; check render_mode and compatibility.")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    
    parser.add_argument("--num_layers", type=int, default=1, help="Number of hidden layers in the policy and value networks")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Width of the layers in the policy and value networks")
    
    parser.add_argument("--det", action="store_true", help="Enable deterministic policy evaluation every --eval-interval iterations")
    parser.add_argument("--T", type=float, default=1.0, help="Softmax temperature for the policy. If a temperature scheduler is used, this will be the starting temperature")
    
    parser.add_argument("--t_schedule", choices=["linear", "exponential"], help="Choose between a linear or exponential temperature scheduler")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Coefficient for entropy regularization")
    
    parser.add_argument("--env", default="cartpole", choices=["cartpole", "lunarlander"], help="Choose between the Cartpole and the LunarLander environment")
    parser.add_argument("--checkpoint", type=str, default="wandb/latest-run/files/checkpoint-best_eval_policy.pt")
    parser.add_argument("--gif-path", type=str, default="cartpole.gif", help="Path to save the gif.")

    args = parser.parse_args()
    
    render_mode = "rgb_array" 

    if args.env == "cartpole":
        env = gym.make("CartPole-v1", render_mode=render_mode)
    elif args.env == "lunarlander":
        env = gym.make("LunarLander-v3", render_mode=render_mode)

    # Define your policy architecture (must match what was trained)
    policy = PolicyNetwork(env, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    policy.eval()

    # Load checkpoint
    policy = load_checkpoint(policy, args.checkpoint)

    print(f"Recording GIF to {args.gif_path}...")
    make_gif(env, policy, args.checkpoint, args.gif_path, temperature=args.T, deterministic=args.det)

if __name__ == "__main__":
    main()
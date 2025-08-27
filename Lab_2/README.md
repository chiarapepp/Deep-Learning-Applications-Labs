# Lab 3 — Deep Reinforcement Learning

This lab implements the REINFORCE algorithm to train an agent in the **CartPole-v1** and **LunarLander-v3** environments using PyTorch. 
The project supports various enhancements including value baselines, entropy regularization, gradient clipping, and temperature scheduling.

- **Training (main script)**:
  - Trains a policy using the REINFORCE algorithm.
  - Supports optional baselines: none, std or learned (value network).
  - Allows advantage normalization, gradient clipping, and temperature scheduling.
  - Evaluates and logs performance to wandb.

- **Networks**:
  - `PolicyNet`: Feedforward policy network for action selection.
  - `ValueNet`: Optional baseline network for advantage estimation.

- **Evaluation & Visualization**:
  - Stochastic and Deterministic Policy sampling.
  - `visualize.py`: Loads a trained policy checkpoint and runs it in a visual environment for inspection.
  - Register `.gif` of agents in the Cartpole/LunarLander visual environment 

- **Utilities**:
  - Modular structure to allow plug-and-play of environments, policies, and configurations.
  - Command-line arguments for different training and evaluation setups.
  
## 📁 Folder Structure

```
├── main.py           # Entry point for training
├── reinforce.py      # Core REINFORCE training loop
├── visualize.py      # Visualize a trained agent and render a gif 
├── networks.py       # Policy and value network definitions
├── utils.py          # Utility functions for training and evaluation
└── README.md         # This file
```

---

## 🚀  Usage

Use `main.py` to train an agent on a given environment.

### 🔧 Example: Train an agent on LunarLander-v3 

```bash
python main.py --env lunarlander --baseline std --episodes 1000 --depth 2 --width 256 --normalize --clip-grad --det --T 1.0 --t-schedule exp
```

### 🎮 Example: Visualize a pretrained agent trying to do its best

```bash
python visualize.py --checkpoint path/to/policy_weights
```
### 🎮 Example: Save a gif of an agent landing on the moon

```bash
python visualize.py --checkpoint path/to/policy_weights --env lunarlander --gif --deterministic --gif-path landing_loop.gif
```
## 📥 Supported Arguments

###  `main.py`

This script is the main training entry point for running the REINFORCE algorithm using PyTorch on CartPole-v1 or LunarLander-v3 environments. 

- Manages experiment configuration and logging

- Sets up environments and models

- Trains a REINFORCE agent using the specified configuration

- Optionally visualizes the trained agent

| Argument              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `--env`               | Environment to train on: `cartpole` or `lunarlander`                        |
| `--baseline`          | Baseline to use: `none`, `std`, or `learned`                                |
| `--episodes`          | Number of episodes to train for                                             |
| `--depth`             | Number of hidden layers in policy/value networks                            |
| `--width`             | Width of each hidden layer                                                  |
| `--lr`                | Learning rate (default: `1e-3`)                                              |
| `--gamma`             | Discount factor (default: `0.99`)                                           |
| `--normalize`         | Normalize the advantage term                                                |
| `--clip-grad`         | Clip gradients to unit norm                                                 |
| `--det`               | Perform deterministic evaluations                                           |
| `--T`                 | Initial temperature for action sampling                                     |
| `--t-schedule`        | Temperature schedule: `lin`, `exp`, or leave unset for static temperature   |
| `--eval-interval`     | How often to evaluate the policy                                            |
| `--eval-episodes`     | Number of episodes to run during each evaluation                            |
| `--visualize`         | Visualize the final trained policy after training                           |


### `visualize.py`


This script is used to **load and visualize a trained policy network** in either the CartPole or LunarLander environments. 
The policy is loaded from a specified checkpoint and used to run a number of episodes, rendered in a window for visual inspection.
The script allows the user to optionally render a `.gif` of a previously trained agent acting inside the environment.

| Argument         | Type    | Default                                               | Description                                                                 |
|------------------|---------|-------------------------------------------------------|-----------------------------------------------------------------------------|
| `--checkpoint`   | `str`   | `"wandb/latest-run/files/checkpoint-BEST_EVAL_POLICY.pt"` | Path to the policy checkpoint file.                                         |
| `--episodes`     | `int`   | `10`                                                  | Number of episodes to run (used only if not recording a GIF).               |
| `--temperature`  | `float` | `1.0`                                                 | Softmax temperature for the policy sampling.                                |
| `--deterministic`| `flag`  | `False`                                               | If set, uses greedy action selection instead of sampling.                   |
| `--env`          | `str`   | _(required)_                                          | Environment to run: `cartpole` or `lunarlander`.                            |
| `--width`        | `int`   | `256`                                                 | Width of the hidden layers in the policy network.                           |
| `--depth`        | `int`   | `2`                                                   | Number of hidden layers in the policy network.                              |
| `--gif`          | `flag`  | `False`                                               | If set, records a single episode as a GIF instead of running multiple episodes interactively. |
| `--gif-path`     | `str`   | `"lander.gif"`                                        | Output path for the recorded GIF if `--gif` is set.                         |

---

## 📊 Evaluation & Logging

Evaluation is done every `--eval-interval` steps with `--eval-episodes` episodes.

Logged metrics include:
- Episode return and length
- Evaluation reward (stochastic and deterministic)
- Losses: policy and value loss (if using learned baseline)
- Temperature schedule

Model checkpoints are saved to:
```
wandb/latest-run/files/checkpoint-BEST_EVAL_POLICY.pt
```

---

## 📊 Results Summary

#### 🏗️ Cartpole-v1: Qualitative Results

<p align="center">
  <img src="images/cartpole_loop.gif" alt="Demo" />
</p>

#### 🏗️ Cartpole-v1: Stochastic and Deterministic Average Evaluation Rewards

<p align="center">
  <img src="images/cartpole_eval_avg_stochastic_reward.png" width="45%" alt="cnn OOD evaluation">
  <img src="images/cartpole_avg_det_eval_reward.png" width="45%" alt="cartpolereward">
</p>

🔗 More results available at [Cartpole](https://wandb.ai/jaysenoner/DLA2025-Cartpole?nw=nwuserjaysenoner1999)


Observations:

- (Personal) This was the hardest lab for me since I haven't got any prior experience in Reinforcement Learning. However, it was also the funniest one.
- Reinforce without baseline (`none_base_eval`, yellow line) consistently performs poorly in both training and evaluation, struggling to achieve high rewards and exhibiting high instability. This clearly demonstrates the critical need for a baseline in REINFORCE
- This is further confirmed by the other experiment without baseline but with a deeper and wider policy network (`none_w256_d2`, blue line), which shows a catastrophic collapse after initial learning, highlighting potential instability during training without baseline and proper hyperparameter tuning
- Various learned baseline approaches (`linearT`, `expT`, `clip+norm`, `norm`) and the standardized reward baseline (`std_base_deteval`,) all show strong performance, generally converging to high rewards by around 400-600 steps.
- There are minor differences in convergence speed and stability, but overall, most baseline strategies are effective for this simpler environment
- Overall, better performances are reached by using a learned baseline with a temperature scaling (`learned_base_linearT`, green line  or `learned_base_expT`, orange line)

#### 🌕 LunarLander-v3: Qualitative Results

<p align="center">
  <img src="images/landing_loop.gif" alt="Demo" />
</p>


#### 🌕 LunarLander-v3: Stochastic and Deterministic Average Evaluation Rewards

<p align="center">
  <img src="images/lander_eval_stochastic_avg_reward.png" width="45%" alt="cnn OOD evaluation">
  <img src="images/lander_avg_det_eval_reward.png" width="45%" alt="cartpolereward">
</p>

🔗 More results available at [Wandb - LunarLander](https://wandb.ai/jaysenoner/DLA2025-LunarLander)

Observations:

- To solve the LunarLander environment with REINFORCE I needed to implement a lot of tricks in order to stabilize training such as normalizing advantages, entropy regularization and temperature scaling
- LunarLander is a more challenging environment, and this is reflected in the reward curves: Initial rewards are highly negative, and the learning process shows significant variance across different runs
- I managed to achieve a peak average evaluation reward of approx `277` (violet curve) by training an agent with REINFORCE for `7000` episodes, using gradient clipping, advantage normalization, exponential temperature scheduling (with a starting temperature of `2`), entropy regularization and a deeper and wider network with respect to our baseline (`width = 256` instead of `128`, `depth = 2` instead of `1`)
- For more complex tasks like LunarLander, REINFORCE exhibits higher variance and greater sensitivity to hyperparameters. Achieving good performance is possible but not guaranteed, often requiring careful tuning and potentially many trials. The results highlight the challenges of applying vanilla REINFORCE to harder problems without more advanced techniques for variance reduction or exploration. 

## 📚 References

- [Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.](https://link.springer.com/article/10.1007/BF00992696) Machine learning, 8(3-4), 229-256
- [Greensmith, E., Bartlett, P. L., & Baxter, J. (2004). Variance reduction techniques for gradient estimates in reinforcement learning.](https://www.jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf) Journal of Machine Learning Research, 5(Nov), 1471-1530

---

## 🧑‍💻 Author

Jay Senoner — [@jaysenoner99](https://github.com/jaysenoner99)
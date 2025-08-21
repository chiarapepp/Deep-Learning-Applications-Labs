# Deep Learning Applications Laboratory Assignments

This repository contains the implementation of three laboratory assignments developed for the *Deep Learning Applications* course, part of the MSc in Artificial Intelligence at the **University of Florence**.

Each lab is organized in a separate folder and includes all code required to reproduce the experiments and results. While each lab includes a dedicated `README.md` with experiment-specific details and visualizations, full training logs and results are available through public **Weights & Biases** (W&B) project pages, linked below.

---

## 📁 Lab Overviews

### **Lab 1 – Working with Deep Models**

This lab explores training deep neural networks using MLPs and CNNs, with a particular focus on **residual connections**. The main goal is to reproduce (on a smaller scale) the findings from the paper:

> [**Deep Residual Learning for Image Recognition**](https://arxiv.org/abs/1512.03385) – Kaiming He et al., CVPR 2016

Additionally, the lab includes an implementation of **Class Activation Maps (CAMs)** based on:

> [**Learning Deep Features for Discriminative Localization**](http://cnnlocalization.csail.mit.edu/#:~:text=A%20class%20activation%20map%20for,decision%20made%20by%20the%20CNN.) – Zhou et al., CVPR 2016

🔗 [View results on Weights & Biases](https://wandb.ai/jaysenoner/lab_1_DLA?nw=nwuserjaysenoner1999)

---

### **Lab 2 Deep Reinforcement Learning**

This lab features a modular and configurable implementation of the **REINFORCE algorithm**, based on:

> [**Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning**](https://link.springer.com/article/10.1007/BF00992696#citeas) – Williams, 1992

It is applied to two environments from [Gymnasium](https://gymnasium.farama.org/):

- [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

🔗 [CartPole Results](https://wandb.ai/jaysenoner/DLA2025-Cartpole?nw=nwuserjaysenoner1999)  
🔗 [Lunar Lander Results](https://wandb.ai/jaysenoner/DLA2025-LunarLander?nw=nwuserjaysenoner1999)

---

## ⚙️ Setup Instructions

To get started, clone the repository and install the required Python packages.

```bash
# Clone the repository
git clone https://github.com/jaysenoner99/DLA.git
cd DLA
```

```bash
# Create and activate a virtual environment (example using venv)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

Each lab may contain additional setup notes or specific dependencies, which are documented in the corresponding subfolder’s `README.md`.


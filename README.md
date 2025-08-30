# Deep Learning Applications Laboratory Assignments

This repository contains the implementation of three laboratory assignments developed as part of the exam for the **Deep Learning Applications** course, MSc in Artificial Intelligence at the University of Florence.

Each lab is organized in a dedicated folder and includes all the code required to reproduce the experiments and results. Inside each folder, a `README.md` file provides details specific for each experiment, visualizations, and explanations of the design choices made, along with references and sources.

To complement the code and documentation, full training logs and results are available through public Weights & Biases (W&B) project pages.

---

## Lab Overviews

### **Laboratory 1 - Deep Neural Networks: MLPs, ResMLPs, and CNNs**

This lab explores the impact of residual connections on deep network training, reproducing key findings from:

> [**Deep Residual Learning for Image Recognition**](https://arxiv.org/abs/1512.03385) – Kaiming He et al., CVPR 2016

Additionally, the lab includes a fine-tuning experiment with different layer freezing strategies and an SVM baseline comparison using extracted features.

🔗 [Lab_1 Results](https://wandb.ai/chiara-peppicelli-university-of-florence/DLA_Lab_1?nw=nwuserchiarapeppicelli)

---

### Laboratory 2 — Deep Reinforcement Learning : REINFORCE on CartPole & LunarLander

This lab features a comprehensive implementation of the **REINFORCE algorithm**, based on:

> [**Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning**](https://link.springer.com/article/10.1007/BF00992696#citeas) – Williams, 1992

It is applied to two environments from *Gymnasium*:

- [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

🔗 [Lab_2 Results](https://wandb.ai/chiara-peppicelli-university-of-florence/DLA_Lab_2?nw=nwuserchiarapeppicelli)  

---

### Laboratory 3 - Working with Transformers in the HuggingFace Ecosystem

This lab provides a comprehensive exploration of transformer fine-tuning within the HuggingFace ecosystem. It investigates both **full fine-tuning** and **LoRA (Low-Rank Adaptation)** approaches, highlighting their differences in terms of efficiency and performance.

The implementation and methodology are primarily inspired by the [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/).

🔗 [Lab_3 Results](https://wandb.ai/chiara-peppicelli-university-of-florence/DLA_Lab_3?nw=nwuserchiarapeppicelli)  

---
## ⚙️ Setup Instructions

To get started, clone the repository and install the required Python packages.

1. Clone the repository
```bash
git clone https://github.com/chiarapepp/Deep-Learning-Applications-Labs.git
cd DLA
```
2. Create and activate a virtual environment
```bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
Each lab may contain additional setup notes or specific dependencies, which are documented in the corresponding subfolder’s `README.md`.


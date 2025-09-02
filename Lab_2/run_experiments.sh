#!/bin/bash

# ========================================================================================
# Deep Learning Applications - Laboratory 2
# ========================================================================================

# Common argument sets
CARTPOLE_ARGS="--env cartpole --episodes 1000 --eval_interval 50 --eval_episodes 20"
LUNARLANDER_ARGS="--env lunarlander --episodes 2000 --eval_interval 100 --eval_episodes 20"
LUNARLANDER_EXTENDED="--env lunarlander --episodes 5000 --eval_interval 100 --eval_episodes 20"

# ====================================================
# REINFORCE with No Baseline on CartPole 
# ====================================================

# REINFORCE on CartPole (no baseline)
python main.py $CARTPOLE_ARGS --baseline none --run_name "REINFORCE_CartPole_no_baseline"

# Bigger architecture
python main.py $CARTPOLE_ARGS --baseline none --num_layers 2 --hidden_dim 256 --run_name "REINFORCE_CartPole_no_baseline_architecture_2_256"

# ====================================================
# REINFORCE with STD Baseline on CartPole 
# ====================================================

# REINFORCE with standardization baseline on CartPole
python main.py $CARTPOLE_ARGS --baseline std --run_name "REINFORCE_CartPole_std_baseline"

# Comparison with different learning rates on CartPole
python main.py $CARTPOLE_ARGS --baseline std --lr 1e-2 --run_name "REINFORCE_CartPole_std_baseline_lr=1e-2"
python main.py $CARTPOLE_ARGS --baseline std --lr 1e-4 --run_name "REINFORCE_CartPole_std_baseline_lr=1e-4"

# Comparison with different gamma values on CartPole
python main.py $CARTPOLE_ARGS --baseline std --gamma 0.95 --run_name "REINFORCE_CartPole_std_baseline_gamma=0.95"
python main.py $CARTPOLE_ARGS --baseline std --gamma 0.999 --run_name "REINFORCE_CartPole_std_baseline_gamma=0.999"

# Deeper architecture
python main.py $CARTPOLE_ARGS --baseline std --num_layers 2 --hidden_dim 128 --run_name "REINFORCE_CartPole_std_baseline_architecture_2_128"

# ===========================================
# REINFORCE with Value Baseline on CartPole
# ===========================================

# REINFORCE with value baseline on CartPole
python main.py $CARTPOLE_ARGS --baseline value --run_name "REINFORCE_CartPole_value_baseline"

# Value baseline with advantage normalization
python main.py $CARTPOLE_ARGS --baseline value --normalize --run_name "REINFORCE_CartPole_value_baseline_normalize" 

# Value baseline with gradient clipping
python main.py $CARTPOLE_ARGS --baseline value --clip_grad --run_name "REINFORCE_CartPole_value_baseline_clip_grad"

# Value baseline with both stabilization techniques
python main.py $CARTPOLE_ARGS --baseline value --normalize --clip_grad --run_name "REINFORCE_CartPole_value_baseline_normalize_clip_grad"

# Comparison of different architectures 
python main.py $CARTPOLE_ARGS --baseline value --num_layers 2 --hidden_dim 128 --run_name "REINFORCE_CartPole_value_baseline_architecture_2_128"
python main.py $CARTPOLE_ARGS --baseline value --num_layers 2 --hidden_dim 256 --run_name "REINFORCE_CartPole_value_baseline_architecture_2_256"

# Comparison with different gamma values 
python main.py $CARTPOLE_ARGS --baseline value --gamma 0.95 --run_name "REINFORCE_CartPole_value_baseline_gamma=0.95"
python main.py $CARTPOLE_ARGS --baseline value --gamma 0.90 --run_name "REINFORCE_CartPole_value_baseline_gamma=0.90"

# Experiments with different fixed temperatures
python main.py $CARTPOLE_ARGS --baseline value --T 0.5 --run_name "REINFORCE_CartPole_value_baseline_T=0.5"
python main.py $CARTPOLE_ARGS --baseline value --T 2.0 --run_name "REINFORCE_CartPole_value_baseline_T=2.0"

# Linear temperature scheduling
python main.py $CARTPOLE_ARGS --baseline value --T 2.0 --t_schedule linear --run_name "REINFORCE_CartPole_value_baseline_T=2.0_t_schedule=linear"

# Exponential temperature scheduling
python main.py $CARTPOLE_ARGS --baseline value --T 2.0 --t_schedule exponential --run_name "REINFORCE_CartPole_value_baseline_T=2.0_t_schedule=exponential"

# Entropy regularization experiments
python main.py $CARTPOLE_ARGS --baseline value --entropy_coeff 0.0 --run_name "REINFORCE_CartPole_value_baseline_entropy_coeff=0.0"
python main.py $CARTPOLE_ARGS --baseline value --entropy_coeff 0.05 --run_name "REINFORCE_CartPole_value_baseline_entropy_coeff=0.05"

# ========================================
# Deterministic evaluation on CartPole
# ========================================

# Once the main experiments are done I decided to try out some deterministic evaluations.

# No baseline
python main.py $CARTPOLE_ARGS --baseline none --det --run_name "REINFORCE_CartPole_no_baseline_deterministic"

# Baseline std
python main.py $CARTPOLE_ARGS --baseline std --det --run_name "REINFORCE_CartPole_std_baseline_deterministic"

# Baseline value
python main.py $CARTPOLE_ARGS --baseline value --det --run_name "REINFORCE_CartPole_value_baseline_deterministic"

# ========================================
# Lunar Lander Environment
# ========================================

# REINFORCE on LunarLander (no baseline)
python main.py $LUNARLANDER_ARGS --baseline none --run_name "REINFORCE_LunarLander_no_baseline"

# REINFORCE with standardization baseline on LunarLander
python main.py $LUNARLANDER_ARGS --baseline std --run_name "REINFORCE_LunarLander_std_baseline"

# REINFORCE with value baseline on LunarLander
python main.py $LUNARLANDER_ARGS --baseline value --run_name "REINFORCE_LunarLander_value_baseline"

# LunarLander with all stabilization techniques
python main.py $LUNARLANDER_ARGS --baseline value --normalize --clip_grad --run_name "REINFORCE_LunarLander_value_baseline_normalize_clip_grad"

# LunarLander with smaller learning rate
python main.py $LUNARLANDER_ARGS --baseline value --lr 5e-4 --normalize --clip_grad --run_name "REINFORCE_LunarLander_value_baseline_lr=5e-4_normalize_clip_grad"

# Comparison with different gamma values
python main.py $LUNARLANDER_ARGS --baseline value --gamma 0.90 --run_name "REINFORCE_LunarLander_value_baseline_gamma=0.9"
python main.py $LUNARLANDER_ARGS --baseline value --gamma 0.95 --run_name "REINFORCE_LunarLander_value_baseline_gamma=0.95"
python main.py $LUNARLANDER_ARGS --baseline value --gamma 0.999 --run_name "REINFORCE_LunarLander_value_baseline_gamma=0.999"

# ===================================================
# Deterministic evaluation Lunar Lander
# ===================================================

# # No baseline
python main.py $LUNARLANDER_ARGS --baseline none --det --run_name "REINFORCE_LunarLander_no_baseline_deterministic"

# Baseline std
python main.py $LUNARLANDER_ARGS --baseline std --det --run_name "REINFORCE_LunarLander_std_baseline_deterministic"

# Baseline value
python main.py $LUNARLANDER_ARGS --baseline value --det --run_name "REINFORCE_LunarLander_value_baseline_deterministic"

# ========================================
# Lunar Lander longer
# ========================================

python main.py $LUNARLANDER_EXTENDED --baseline value --normalize --clip_grad --run_name "REINFORCE_LunarLander_value_baseline_normalize_clip_grad"
python main.py $LUNARLANDER_EXTENDED --baseline value --num_layers 2 --hidden_dim 256 --normalize --clip_grad --run_name "REINFORCE_LunarLander_value_baseline_norm_clip_2_256"
python main.py $LUNARLANDER_EXTENDED --baseline value --normalize --clip_grad --det --run_name "REINFORCE_LunarLander_value_baseline_deterministic"

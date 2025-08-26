#!/bin/bash

# run_experiments.sh - Comprehensive experiment runner for DLA Lab 2
# Make executable with: chmod +x run_experiments.sh

set -e  # Exit on any error

echo "Starting DLA Lab 2 Experiments"
echo "=================================="

# Configuration
EPISODES=1000
EVAL_INTERVAL=50
EVAL_EPISODES=20
LR=1e-3

# Function to run a single experiment
run_experiment() {
    local name="$1"
    local args="$2"
    echo "Running experiment: $name"
    echo "Args: $args"
    python main.py $args
    echo "Completed: $name"
    echo ""
}

# =============================================================================
echo "EXERCISE 1: Basic REINFORCE Experiments"
echo "----------------------------------------"

# Cartpole with different baselines
run_experiment "Cartpole - No Baseline" \
    "--env cartpole --baseline none --episodes $EPISODES --lr $LR"

run_experiment "Cartpole - Standardization Baseline" \
    "--env cartpole --baseline std --episodes $EPISODES --lr $LR"

# =============================================================================
echo "EXERCISE 2: Value Baseline Experiments"
echo "----------------------------------------"

run_experiment "Cartpole - Value Baseline" \
    "--env cartpole --baseline value --episodes $EPISODES --lr $LR"

run_experiment "Cartpole - Value Baseline + Normalization" \
    "--env cartpole --baseline value --normalize --episodes $EPISODES --lr $LR"

# =============================================================================
echo "EXERCISE 3: LunarLander Experiments"
echo "------------------------------------"

run_experiment "LunarLander - No Baseline" \
    "--env lunarlander --baseline none --episodes 2000 --lr $LR"

run_experiment "LunarLander - Value Baseline" \
    "--env lunarlander --baseline value --normalize --episodes 2000 --lr $LR"

# =============================================================================
echo "Hyperparameter Studies"
echo "-------------------------"

# Learning rate study
for lr in 1e-4 1e-3 1e-2; do
    run_experiment "Cartpole - LR Study (lr=$lr)" \
        "--env cartpole --baseline value --lr $lr --episodes $EPISODES"
done

# Gamma study
for gamma in 0.95 0.99 0.995; do
    run_experiment "Cartpole - Gamma Study (gamma=$gamma)" \
        "--env cartpole --baseline value --gamma $gamma --episodes $EPISODES --lr $LR"
done

# Network architecture study
run_experiment "Cartpole - Deeper Network (2 layers)" \
    "--env cartpole --baseline value --num_layers 2 --episodes $EPISODES --lr $LR"

run_experiment "Cartpole - Wider Network (256 units)" \
    "--env cartpole --baseline value --hidden_dim 256 --episodes $EPISODES --lr $LR"

# =============================================================================
echo "Advanced Features Study"
echo "-------------------------"

# Temperature scheduling
run_experiment "Cartpole - Linear Temperature Schedule" \
    "--env cartpole --baseline value --T 2.0 --t_schedule linear --episodes $EPISODES --lr $LR"

run_experiment "Cartpole - Exponential Temperature Schedule" \
    "--env cartpole --baseline value --T 2.0 --t_schedule exponential --episodes $EPISODES --lr $LR"

# Gradient clipping
run_experiment "Cartpole - With Gradient Clipping" \
    "--env cartpole --baseline value --clip-grad --episodes $EPISODES --lr $LR"

# Deterministic evaluation
run_experiment "Cartpole - Deterministic Evaluation" \
    "--env cartpole --baseline value --det --episodes $EPISODES --lr $LR"

# =============================================================================
echo "Ablation Studies"
echo "------------------"

# Full featured vs minimal
run_experiment "Cartpole - Full Featured" \
    "--env cartpole --baseline value --normalize --clip-grad --det --T 1.5 --t_schedule exponential --episodes $EPISODES --lr $LR"

run_experiment "Cartpole - Minimal" \
    "--env cartpole --baseline none --episodes $EPISODES --lr $LR"

# =============================================================================
echo "Final Best Model Training"
echo "----------------------------"

# Train best models for longer
run_experiment "Cartpole - Best Model (Long Training)" \
    "--env cartpole --baseline value --normalize --clip-grad --det --T 1.2 --t_schedule exponential --episodes 2000 --lr 5e-4 --visualize"

run_experiment "LunarLander - Best Model (Long Training)" \
    "--env lunarlander --baseline value --normalize --clip-grad --det --T 1.0 --episodes 3000 --lr 1e-3 --visualize"


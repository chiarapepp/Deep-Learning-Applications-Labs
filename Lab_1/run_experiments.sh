#!/bin/bash

# =============================================================================
# Deep Learning Applications - Laboratory 1
# =============================================================================
# 
# This script runs:
# - 72 MLP experiments (3 depths × 3 widths × 8 configurations)
# - 12 CNN experiments (3 architectures × 4 configurations) 
# - 24 Fine-tuning experiments (3 freeze strategies × 8 configurations)
#
# =============================================================================

# Create necessary directory
mkdir -p Models 

# =============================================================================
# EXERCISE 1.1 & 1.2: MLP EXPERIMENTS
# =============================================================================
# Common arguments
MLP_ARGS="--dataset MNIST --lr 1e-3 --use_wandb"
# -----------------------------------------------------------------------------
# MLP Experiments - Fixed width 128
# -----------------------------------------------------------------------------
echo " Running MLP experiments with WIDTH=128 "
echo "  └── Testing depths: 40, 20, 10 with all configurations"
echo ""

# Depth 40, Width 128 


python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --normalization --depth 40 --width 128 # norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --normalization --depth 40 --width 128 # norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --normalization --depth 40 --width 128 # norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --normalization --depth 40 --width 128 # norm, no scheduler

python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --depth 40 --width 128 # no norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --depth 40 --width 128 # no norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --depth 40 --width 128  # no norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --depth 40 --width 128   # no norm, no scheduler

# Depth 20, Width 128 

python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --normalization --depth 20 --width 128 # norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --normalization --depth 20 --width 128 # norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --normalization --depth 20 --width 128 # norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --normalization --depth 20 --width 128 # norm, no scheduler

python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --depth 20 --width 128 # no norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --depth 20 --width 128 # no norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --depth 20 --width 128 # no norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --depth 20 --width 128 # no norm, no scheduler

# Depth 10, Width 128 

python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --normalization --depth 10 --width 128 # norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --normalization --depth 10 --width 128 # norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --normalization --depth 10 --width 128 # norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --normalization --depth 10 --width 128 # norm, no scheduler

python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --depth 10 --width 128 # no norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --depth 10 --width 128 # no norm, no scheduler
 
python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --depth 10 --width 128 # no norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --depth 10 --width 128 # no norm, no scheduler

# -----------------------------------------------------------------------------
# MLP Experiments - Fixed width 64 
# -----------------------------------------------------------------------------
echo ""
echo " Running MLP experiments with Width=64"
echo "  └── Testing depths: 40, 20, 10 with all configurations"
echo ""

# Depth 40, Width 64 

# Depth 40, Width 64 
python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --normalization --depth 40 --width 64 # norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --normalization --depth 40 --width 64 # norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --normalization --depth 40 --width 64 # norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --normalization --depth 40 --width 64 # norm, no scheduler

python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --depth 40 --width 64 # no norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --depth 40 --width 64 # no norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --depth 40 --width 64 # no norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --depth 40 --width 64 # no norm, no scheduler

# Depth 20, Width 64 
python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --normalization --depth 20 --width 64 # norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --normalization --depth 20 --width 64 # norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --normalization --depth 20 --width 64 # norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --normalization --depth 20 --width 64 # norm, no scheduler

python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --depth 20 --width 64 # no norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --depth 20 --width 64 # no norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --depth 20 --width 64 # no norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --depth 20 --width 64 # no norm, no scheduler

# Depth 10, Width 64 
python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --normalization --depth 10 --width 64 # norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --normalization --depth 10 --width 64 # norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --normalization --depth 10 --width 64 # norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --normalization --depth 10 --width 64 # norm, no scheduler

python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --depth 10 --width 64 # no norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --depth 10 --width 64 # no norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --depth 10 --width 64 # no norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --depth 10 --width 64 # no norm, no scheduler

# -----------------------------------------------------------------------------
# MLP Experiments - Fixed width 32 
# -----------------------------------------------------------------------------
echo ""
echo " Running MLP experiments with WIDTH=32 (24 experiments)..."
echo "  └── Testing depths: 40, 20, 10 with all configurations"
echo ""

# Depth 40, Width 32 
python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --normalization --depth 40 --width 32 # norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --normalization --depth 40 --width 32 # norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --normalization --depth 40 --width 32 # norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --normalization --depth 40 --width 32 # norm, no scheduler

python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --depth 40 --width 32 # no norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --depth 40 --width 32 # no norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --depth 40 --width 32 # no norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --depth 40 --width 32 # no norm, no scheduler

# Depth 20, Width 32  
python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --normalization --depth 20 --width 32 # norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --normalization --depth 20 --width 32 # norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --normalization --depth 20 --width 32 # norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --normalization --depth 20 --width 32 # norm, no scheduler

python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --depth 20 --width 32 # no norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --depth 20 --width 32 # no norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --depth 20 --width 32 # no norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --depth 20 --width 32 # no norm, no scheduler

# Depth 10, Width 32 
python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --normalization --depth 10 --width 32  # norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --normalization --depth 10 --width 32 # norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --normalization --depth 10 --width 32 # norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --normalization --depth 10 --width 32 # norm, no scheduler

python main_ex1.py --model resmlp $MLP_ARGS --use_scheduler --depth 10 --width 32 # no norm, scheduler
python main_ex1.py --model resmlp $MLP_ARGS --depth 10 --width 32 # no norm, no scheduler

python main_ex1.py --model mlp $MLP_ARGS --use_scheduler --depth 10 --width 32 # no norm, scheduler
python main_ex1.py --model mlp $MLP_ARGS --depth 10 --width 32 # no norm, no scheduler

echo "✅ MLP experiments completed!"
echo ""

# =============================================================================
# EXERCISE 1.3: CNN EXPERIMENTS 
# =============================================================================
# Common arguments
CNN_ARGS="--dataset CIFAR10 --lr 1e-3 --use_wandb"
# -----------------------------------------------------------------------------
# CNN ResNet-18 style [2,2,2,2] 
# -----------------------------------------------------------------------------
echo " Running CNN ResNet-18 style [2,2,2,2]"
echo "  └── With/without residual connections and scheduler"
echo ""

python main_ex1.py --model cnn $CNN_ARGS --use_scheduler --use_residual --layers 2 2 2 2     # scheduler, residual
python main_ex1.py --model cnn $CNN_ARGS --use_residual --layers 2 2 2 2      # no scheduler, residual
python main_ex1.py --model cnn $CNN_ARGS --use_scheduler --layers 2 2 2 2     # scheduler, no residual
python main_ex1.py --model cnn $CNN_ARGS --layers 2 2 2 2     # no scheduler, no residual

# -----------------------------------------------------------------------------
# CNN ResNet-34 style [3,4,6,3] 
# -----------------------------------------------------------------------------
echo "🔹 Running CNN ResNet-34 style [3,4,6,3]"
echo "  └── With/without residual connections and scheduler"
echo ""
python main_ex1.py --model cnn $CNN_ARGS --use_scheduler --use_residual --layers 3 4 6 3    # scheduler, residual
python main_ex1.py --model cnn $CNN_ARGS --use_residual --layers 3 4 6 3     # no scheduler, residual
python main_ex1.py --model cnn $CNN_ARGS --use_scheduler --layers 3 4 6 3    # scheduler, no residual
python main_ex1.py --model cnn $CNN_ARGS --layers 3 4 6 3    # no scheduler, no residual

# -----------------------------------------------------------------------------
# CNN ResNet-50 style [5,6,8,5]
# -----------------------------------------------------------------------------
echo "🔹 Running CNN ResNet-50 style [5,6,8,5] (just to try a more deep network even though ResNet-50 uses Bottleneck blocks)"
echo "  └── With/without residual connections and scheduler"
echo ""
python main_ex1.py --model cnn $CNN_ARGS --use_scheduler --use_residual --layers 5 6 8 5  # scheduler, residual
python main_ex1.py --model cnn $CNN_ARGS --use_residual --layers 5 6 8 5      # no scheduler, residual
python main_ex1.py --model cnn $CNN_ARGS --use_scheduler --layers 5 6 8 5    # scheduler, no residual
python main_ex1.py --model cnn $CNN_ARGS --layers 5 6 8 5         # no scheduler, no residual

echo "✅ CNN experiments completed!"
echo ""


# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                     ALL EXPERIMENTS COMPLETED!                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

echo "Results saved in:"
echo "  • Model weights: Models/ directory"
echo "  • W&B Dashboard: Check your Weights & Biases project"
'''
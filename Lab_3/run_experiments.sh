#!/bin/bash

# ========================================================================================
# Deep Learning Applications - Laboratory 3
# ========================================================================================
# 
# This script runs:
# - Token Preprocessing
# - 2 Fine-tuning experiments of DistilBERT with different lr
# - 8 Fine-tuning experiments with LoRA (2 learning rates × 2 LoRA alphas × 2 LoRA ranks)
#
# ========================================================================================

COMMON_ARGS="--batch_size 16 --epochs 5 --use_wandb"

# ======================================================================
# EXERCISE 2: Tokenization, Model Setup, and Fine-tuning of DistilBERT
# ======================================================================

# Token Preprocessing
python main.py --step e21 $COMMON_ARGS

# Fine-tuning DistilBERT (Full Model) 
python main.py --step e23  --lr 2e-5 $COMMON_ARGS --output_dir runs/distilbert_lr2e-5
python main.py --step e23 --lr 2e-4 $COMMON_ARGS --output_dir runs/distilbert_lr2e-4

# ==============================================
# EXERCISE 3: Efficient Fine-tuning (LoRA)
# ==============================================

# lr 2e-5
python main.py --step e31 --lr 2e-5 --lora_alpha 32 --lora_rank 8 $COMMON_ARGS --output_dir runs/lora_lr2e-5_alpha32_rank8
python main.py --step e31 --lr 2e-5 --lora_alpha 32 --lora_rank 16 $COMMON_ARGS --output_dir runs/lora_lr2e-5_alpha32_rank16

python main.py --step e31 --lr 2e-5 --lora_alpha 64 --lora_rank 8 $COMMON_ARGS --output_dir runs/lora_lr2e-5_alpha64_rank8
python main.py --step e31 --lr 2e-5 --lora_alpha 64 --lora_rank 16 $COMMON_ARGS --output_dir runs/lora_lr2e-5_alpha64_rank16

# lr 2e-4
python main.py --step e31 --lr 2e-4 --lora_alpha 32 --lora_rank 8 $COMMON_ARGS --output_dir runs/lora_lr2e-4_alpha32_rank8
python main.py --step e31 --lr 2e-4 --lora_alpha 32 --lora_rank 16 $COMMON_ARGS --output_dir runs/lora_lr2e-4_alpha32_rank16

python main.py --step e31 --lr 2e-4 --lora_alpha 64 --lora_rank 8 $COMMON_ARGS --output_dir runs/lora_lr2e-4_alpha64_rank8
python main.py --step e31 --lr 2e-4 --lora_alpha 64 --lora_rank 16 $COMMON_ARGS --output_dir runs/lora_lr2e-4_alpha64_rank16

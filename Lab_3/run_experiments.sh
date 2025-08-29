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
python main.py --step e21 

# Fine-tuning DistilBERT (Full Model) use in tokenizer_dataset padding=False so that we can use dynamic padding during training
python main.py --step e23  --lr 2e-5 $COMMON_ARGS --output_dir runs/distilbert_lr2e-5
python main.py --step e23 --lr 2e-4 $COMMON_ARGS --output_dir runs/distilbert_lr2e-4

# Fine-tuning DistilBERT (Full Model) use in tokenizer_dataset padding="max_length" (512)
python main.py --step e23  --lr 2e-5 $COMMON_ARGS --output_dir runs/distilbert_lr2e-5_padding512 --run_name distilbert_finetuning_lr:2e-5_padding --use_fixed_padding
python main.py --step e23 --lr 2e-4 $COMMON_ARGS --output_dir runs/distilbert_lr2e-4_padding512 --run_name distilbert_finetuning_lr:2e-4_padding --use_fixed_padding


# ==============================================
# EXERCISE 3: Efficient Fine-tuning (LoRA)
# ==============================================

# Lora used target_modules = q_lin k_lin v_lin out_lin 

# lr 2e-5, padding=False
python main.py --step e31 --lr 2e-5 --lora_alpha 32 --lora_rank 8 $COMMON_ARGS --output_dir runs/lora_lr2e-5_alpha32_rank8
python main.py --step e31 --lr 2e-5 --lora_alpha 64 --lora_rank 16 $COMMON_ARGS --output_dir runs/lora_lr2e-5_alpha64_rank16

# lr 2e-5, padding="max_length" (512)
python main.py --step e31 --lr 2e-5 --lora_alpha 32 --lora_rank 8 $COMMON_ARGS --output_dir runs/lora_lr2e-5_alpha32_rank8_padding --run_name finetuning_with_lora_r:8_a:32_lr:2e-5_padding --use_fixed_padding
python main.py --step e31 --lr 2e-5 --lora_alpha 64 --lora_rank 16 $COMMON_ARGS --output_dir runs/lora_lr2e-5_alpha64_rank16_padding --run_name finetuning_with_lora_r:16_a:64_lr:2e-5_padding --use_fixed_padding

# lr 2e-4, padding=False
python main.py --step e31 --lr 2e-4 --lora_alpha 32 --lora_rank 8 $COMMON_ARGS --output_dir runs/lora_lr2e-4_alpha32_rank8
python main.py --step e31 --lr 2e-4 --lora_alpha 64 --lora_rank 16 $COMMON_ARGS --output_dir runs/lora_lr2e-4_alpha64_rank16

# lr 2e-4, padding="max_length" (512)
python main.py --step e31 --lr 2e-4 --lora_alpha 32 --lora_rank 8 $COMMON_ARGS --output_dir runs/lora_lr2e-4_alpha32_rank8_padding --run_name finetuning_with_lora_r:8_a:32_lr:2e-4_padding --use_fixed_padding
python main.py --step e31 --lr 2e-4 --lora_alpha 64 --lora_rank 16 $COMMON_ARGS --output_dir runs/lora_lr2e-4_alpha64_rank16_padding --run_name finetuning_with_lora_r:16_a:64_lr:2e-4_padding --use_fixed_padding

# ==================================================================================================================

# Lora different target modules used, target_modules = q_lin k_lin v_lin out_lin lin1 lin2

python main.py --step e31 --lr 2e-5 --lora_alpha 32 --lora_rank 8 $COMMON_ARGS --output_dir runs/lora_lr2e-5_alpha32_rank8_tm --run_name finetuning_with_lora_r:8_a:32_lr:2e-5_tm --target_modules q_lin k_lin v_lin out_lin lin1 lin2
python main.py --step e31 --lr 2e-5 --lora_alpha 64 --lora_rank 16 $COMMON_ARGS --output_dir runs/lora_lr2e-5_alpha64_rank16_tm --run_name finetuning_with_lora_r:16_a:64_lr:2e-5_tm --target_modules q_lin k_lin v_lin out_lin lin1 lin2

python main.py --step e31 --lr 2e-4 --lora_alpha 32 --lora_rank 8 $COMMON_ARGS --output_dir runs/lora_lr2e-4_alpha32_rank8_tm --run_name finetuning_with_lora_r:8_a:32_lr:2e-4_tm --target_modules q_lin k_lin v_lin out_lin lin1 lin2
python main.py --step e31 --lr 2e-4 --lora_alpha 64 --lora_rank 16 $COMMON_ARGS --output_dir runs/lora_lr2e-4_alpha64_rank16_tm --run_name finetuning_with_lora_r:16_a:64_lr:2e-4_tm --target_modules q_lin k_lin v_lin out_lin lin1 lin2
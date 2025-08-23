#!/bin/bash

# Parametri
BATCH_SIZE=16
EPOCHS=5

# Learning rates
LRS=("2e-5" "2e-4")

# LoRA parametri
LORA_ALPHAS=(32 64)
LORA_RANKS=(8 16)

echo "=== Esecuzione modello FULL ==="
for LR in "${LRS[@]}"; do
    echo "python main.py --batch_size $BATCH_SIZE --lr $LR --epochs $EPOCHS --step e23"
    python main.py --batch_size $BATCH_SIZE --lr $LR --epochs $EPOCHS --step e23
done

echo "=== Esecuzione modello LoRA ==="
for LR in "${LRS[@]}"; do
    for ALPHA in "${LORA_ALPHAS[@]}"; do
        for RANK in "${LORA_RANKS[@]}"; do
            echo "python main.py --batch_size $BATCH_SIZE --lr $LR --epochs $EPOCHS --step e31 --lora_alpha $ALPHA --lora_rank $RANK"
            python main.py --batch_size $BATCH_SIZE --lr $LR --epochs $EPOCHS --step e31 --lora_alpha $ALPHA --lora_rank $RANK
        done
    done
done

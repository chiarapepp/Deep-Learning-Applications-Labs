import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.svm import SVC
import numpy as np
from peft import LoraConfig, get_peft_model

def count_svm_params(embedding_dim=768, num_classes=2):
    """
    Baseline SVM: linear layer on top of CLS embeddings
    """
    # SVM linear layer has weights + bias
    weight_params = embedding_dim * num_classes
    bias_params = num_classes
    total_params = weight_params + bias_params
    print("SVM Baseline:")
    print(f"  Trainable params: {total_params}\n")

def count_full_finetune_params():
    """
    Full fine-tuning of DistilBERT
    """
    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Full Fine-tuning DistilBERT:")
    print(f"  Total params: {total_params}")
    print(f"  Trainable params: {trainable_params}\n")

def count_lora_params(rank=8, alpha=16, target_modules=None):
    """
    LoRA fine-tuning
    """
    if target_modules is None:
        target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2)

    # Define LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    lora_model = get_peft_model(model, lora_config)
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    print(f"LoRA DistilBERT (r={rank}, alpha={alpha}, targets={target_modules}):")
    print(f"  Total params: {total_params}")
    print(f"  Trainable params: {trainable_params}\n")

if __name__ == "__main__":
    count_svm_params()
    count_full_finetune_params()
    # LoRA rank=8 alpha=32
    count_lora_params(rank=8, alpha=32, target_modules=["q_lin","k_lin","v_lin","out_lin"])
    # LoRA rank=16 alpha=64
    count_lora_params(rank=16, alpha=64, target_modules=["q_lin","k_lin","v_lin","out_lin"])
    # LoRA rank=8 alpha=32
    count_lora_params(rank=8, alpha=32, target_modules=["q_lin","k_lin","v_lin","out_lin","lin1","lin2"])
    # LoRA rank=16 alpha=64
    count_lora_params(rank=16, alpha=64, target_modules=["q_lin","k_lin","v_lin","out_lin","lin1","lin2"])


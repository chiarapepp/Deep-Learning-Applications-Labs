import os
from typing import Optional
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
from utils import compute_metrics, config
from exercise2 import tokenize_dataset
import wandb


# --------------------------------
# EXERCISE 3.1: LoRA Fine-tuning
# --------------------------------

"""
Function that fine-tunes a DistilBERT model for binary sequence classification
using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

Args:
    lora_rank (int): Rank of the LoRA adaptation matrices. Default is 8.
    lora_alpha (int): Alpha scaling factor for LoRA layers. Default is 16.
    lr (float): Learning rate for the optimizer. Default is 2e-5.
    epochs (int): Number of training epochs. Default is 3.
    batch_size (int): Batch size for both training and evaluation. Default is 16.
    output_dir (str): Directory to save checkpoints, logs, and the best model. Default is "runs/distilbert_lora".
    use_wandb (bool): Whether to log metrics and training info to Weights & Biases. Default is False.
    target_modules (list): List of target modules for LoRA adaptation. Default is None.

Returns:
    - trainer.train() returns the Hugging Face `TrainOutput` object containing
        training loss, global step, and metrics.
    - test_results: dictionary with evaluation metrics on the test set if available.
"""

def fine_tune_with_lora(
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 16,
    output_dir: str = "runs/distilbert_lora",
    use_wandb: bool = False,
    run_name: str = None,
    target_modules: list = None,
    use_fixed_padding: Optional[bool] = None
):

    tokenized_ds = tokenize_dataset(use_fixed_padding=use_fixed_padding)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1}
    )
    for name, param in base_model.named_parameters():
        print(f"parameter: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")

    if target_modules is None:
        target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
    else:
        target_modules = target_modules

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,  # Rank of adaptation
        lora_alpha=lora_alpha,  # Alpha parameter for LoRA scaling
        lora_dropout=0.1,  # Dropout probability for LoRA layers
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],  # Target DistilBERT attention modules
        bias="none",
    )
    
    model = get_peft_model(base_model, lora_config)
    for name, param in model.named_parameters():
            if 'lora' in name:
                print(f"LoRA parameter: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")
    print(f"LoRA parameters: {sum(p.numel() for n, p in model.named_parameters() if 'lora' in n):,}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    if run_name is None:
        run_name = f"finetuning_with_lora_r:{lora_rank}_a:{lora_alpha}_lr:{lr}"
    else:
        run_name = run_name

    if use_wandb:
        wandb.init(
            project="DLA_Lab_3",
            name=run_name,
            config={
                "model": config.model_name,
                "dataset": config.dataset_name,
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
            }
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print(f"Starting LoRA fine-tuning...")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    
    train_result = trainer.train()

    if "test" in tokenized_ds:
        test_results = trainer.evaluate(eval_dataset=tokenized_ds["test"])
        print("\nTest Results:")
        for key, value in test_results.items():
            if key.startswith("eval_"):
                print(f"  {key[5:].capitalize()}: {value:.4f}")
    
        if use_wandb:
            wandb.log({f"test/{k.replace('eval_', '')}": v for k, v in test_results.items() if k.startswith("eval_")})


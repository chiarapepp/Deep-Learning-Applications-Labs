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
Function that fine-tunes a DistilBERT model for binary sequence 
classification using LoRA (Parameter-Efficient Fine-tuning).
"""

def fine_tune_with_lora(
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 16,
    output_dir: str = "runs/distilbert_lora",
    use_wandb: bool = False
):
    
    if os.path.exists(config.tokenized_path):
        tokenized_ds = load_from_disk(config.tokenized_path)
    else:
        tokenized_ds = tokenize_dataset()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1}
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,  # Rank of adaptation
        lora_alpha=lora_alpha,  # Alpha parameter for LoRA scaling
        lora_dropout=0.1,  # Dropout probability for LoRA layers
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],  # Target DistilBERT attention modules
        bias="none",
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="wandb" if use_wandb else "none",
        seed=config.seed,
    )
    
    if use_wandb:
        wandb.init(
            project="DLA_Lab_3",
            name=f"distilbert_finetuning_with_lora",
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
    

    trainer.train()
    if "test" in tokenized_ds:
        test_results = trainer.evaluate(eval_dataset=tokenized_ds["test"])
        print("\nTest Results:")
        for key, value in test_results.items():
            if key.startswith("eval_"):
                print(f"  {key[5:].capitalize()}: {value:.4f}")
    
        if use_wandb:
            wandb.log({f"test/{k.replace('eval_', '')}": v for k, v in test_results.items() if k.startswith("eval_")})


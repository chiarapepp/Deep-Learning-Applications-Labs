import os
from typing import Optional, List, Dict
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from utils import compute_metrics, config
from tqdm.auto import tqdm
import wandb


# -------------------------------------
# EXERCISE 2.1: Dataset Tokenization
# -------------------------------------

class TQDMProgressCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.pbar = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.pbar = tqdm(total=state.max_steps, desc="Training", unit="step")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.pbar.update(1)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.pbar.close()


"""
Function that tokenize the dataset and save it to cache
"""

def tokenize_dataset(subset: Optional[int] = None) -> DatasetDict:
    
    ds = load_dataset(config.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    def tokenize_function(batch):
        return tokenizer(
            batch["text"], 
            truncation=True, 
            padding=False,  # Will pad dynamically during training (DataCollatorWithPadding)
            max_length=512
        )
    
    print("Tokenizing dataset...")
    tokenized_ds = ds.map(
        tokenize_function, 
        batched=True, 
        desc="Tokenizing"
    )

    print("\nVerifying tokenized dataset:")
    sample = tokenized_ds["train"][0]
    required_keys = ["text", "label", "input_ids", "attention_mask"]
    
    for key in required_keys:
        if key in sample:
            if key == "text":
                print(f"  ✓ {key}: {sample[key][:50]}...")
            elif key in ["input_ids", "attention_mask"]:
                print(f"  ✓ {key}: length {len(sample[key])}")
            else:
                print(f"  ✓ {key}: {sample[key]}")
        else:
            print(f"  ✗ Missing {key}")
    

    os.makedirs(config.cache_dir, exist_ok=True)
    tokenized_ds.save_to_disk(config.tokenized_path)
    print(f"\nTokenized dataset saved to: {config.tokenized_path}")
    
    return tokenized_ds

# ------------------------------------------
# EXERCISE 2.3: Fine-tuning with Trainer
# ------------------------------------------

"""
Function that fine-tunes a DistilBERT model for binary sequence 
classification using Hugging Face Trainer.
"""

def fine_tune_model(
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 16,
    output_dir: str = "runs/distilbert_finetuned",
    use_wandb: bool = False,
):
      
    if os.path.exists(config.tokenized_path):
        tokenized_ds = load_from_disk(config.tokenized_path)
        print(f"Loaded tokenized dataset from cache")
    else:
        print("No cached tokenized dataset found. Tokenizing now...")
        tokenized_ds = tokenize_dataset()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # ------------------------------------------
    # EXERCISE 2.2: Setup Model for Sequence Classification
    # ------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1}
    )
    
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
        metric_for_best_model="accuracy",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        report_to="wandb" if use_wandb else "none",
        seed=config.seed,
    )
    
    if use_wandb:
        wandb.init(
            project="DLA_Lab_3",
            name=f"distilbert_finetuning",
            config={
                "model": config.model_name,
                "dataset": config.dataset_name,
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
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
        callbacks=[TQDMProgressCallback()],
    )
    
    print(f"Starting fine-tuning...")

    train_result = trainer.train()

    if "test" in tokenized_ds:
        test_results = trainer.evaluate(eval_dataset=tokenized_ds["test"])
        print("\nTest Set Results:")
        for key, value in test_results.items():
            if key.startswith("eval_"):
                metric = key.replace("eval_", "")
                print(f"  {metric.capitalize()}: {value:.4f}")

        if use_wandb:
            wandb.log({f"test/{k.replace('eval_', '')}": v for k, v in test_results.items() if k.startswith("eval_")})
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

"""
This class implements a progress bar using TQDM for training loops.
It updates the progress bar at each training step and closes it at the end of training.
"""

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

def tokenize_dataset(use_fixed_padding: Optional[bool] = False) -> DatasetDict:

    ds = load_dataset(config.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    chosen_padding = "max_length" if use_fixed_padding else False

    def tokenize_function(batch):
        return tokenizer(
            batch["text"], 
            truncation=True, 
            padding=chosen_padding,  # If false, will pad dynamically during training (DataCollatorWithPadding), uUsing padding = max_length makes the training longer
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
Function that fine-tunes a DistilBERT model for binary sequence classification
using Hugging Face Trainer.

Args:
    lr (float): Learning rate for the optimizer. Default is 2e-5.
    epochs (int): Number of training epochs. Default is 5.
    batch_size (int): Batch size for both training and evaluation. Default is 16.
    output_dir (str): Directory to save checkpoints, logs, and the best model. Default is "runs/distilbert_finetuned".
    use_wandb (bool): Whether to log metrics to Weights & Biases. Default is False.
    use_fixed_padding (bool): Whether to use fixed padding (max_length) for input sequences. Default is False.

Returns:
    - train_result: the Hugging Face `TrainOutput` object containing training loss, global step, and metrics.
    - test_results: dictionary with evaluation metrics on the test set if available.

"""

def fine_tune_model(
    lr: float = 2e-5,
    epochs: int = 5,
    batch_size: int = 16,
    output_dir: str = "runs/distilbert_finetuned",
    use_wandb: bool = False,
    run_name: str = None,
    use_fixed_padding: Optional[bool] = None
):
    
    tokenized_ds = tokenize_dataset(use_fixed_padding=use_fixed_padding)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # ------------------------------------------
    # EXERCISE 2.2: Setup Model for Sequence Classification
    # ------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,    # Sentiment analysis: Positive/Negative
        id2label={0: "negative", 1: "positive"},   # Map label IDs to label names
        label2id={"negative": 0, "positive": 1}    # Map label names to label IDs
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if run_name is None:
        run_name = f"distilbert_finetuning_lr:{lr}"
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
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{output_dir}/logs",
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
        compute_metrics=compute_metrics,   # Compute metrics function
        callbacks=[TQDMProgressCallback()],
    )
    
    print(f"Starting fine-tuning...")

    trainer.train()
    trainer.save_model(f"{output_dir}/model_final")

    # Hugging face Trainer add the prefix eval_ automatically
    if "test" in tokenized_ds:
        test_results = trainer.evaluate(eval_dataset=tokenized_ds["test"])
        print("\nTest Set Results:")
        for key, value in test_results.items():
            if key.startswith("eval_"):
                metric = key.replace("eval_", "")
                print(f"  {metric.capitalize()}: {value:.4f}")

        if use_wandb:
            wandb.log({f"test/{k.replace('eval_', '')}": v for k, v in test_results.items() if k.startswith("eval_")})
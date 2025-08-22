"""
Usage examples:
  # inspect dataset
  python main.py --step e11

  # run the SVM baseline (fast: use --subset 200 or more)
  python main.py --step e13 --subset 500

  # tokenize dataset and save to cache
  python main.py --step e21

  # train with HF Trainer and log to wandb
  export WANDB_API_KEY="<your-key>"  # or run `wandb login`
  python main.py --step e23 --use_wandb --wandb_project hf_lab_chiara --output_dir runs/distilbert_full

Notes:
- Adjust batch sizes / epochs to your hardware.
- Install required libraries before running:
    pip install -U datasets transformers accelerate scikit-learn wandb peft torch torchvision

"""

from __future__ import annotations
import os
import argparse
import random
from dataclasses import asdict
from typing import Optional, Tuple, Dict

import numpy as np

from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline,
)

# sklearn
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Optional imports guarded at runtime
try:
    import torch
except Exception:
    torch = None

# Constants / default config
MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_NAME = "rotten_tomatoes"
SEED = 42
CACHE_DIR = ".cache_hf_lab"
TOKENIZED_PATH = os.path.join(CACHE_DIR, "tokenized_rotten")
DEFAULT_WANDB_PROJECT = "hf_lab"


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# -----------------------------
# Exercise 1.1 — Dataset inspect
# -----------------------------

def load_dataset_splits(subset: Optional[int] = None) -> DatasetDict:
    ds = load_dataset(DATASET_NAME)
    if subset is not None:
        ds = DatasetDict({k: v.select(range(min(len(v), subset))) for k, v in ds.items()})
    return ds


def explore_dataset(ds: DatasetDict):
    print("Dataset splits and sizes:")
    for k, v in ds.items():
        print(f"  {k}: {len(v)} samples")
    print("\nSample item from 'train':")
    print(ds["train"][0])


# ---------------------------------
# Exercise 1.2 — Probe model + tok
# ---------------------------------

def probe_model_and_tokenizer(sample_texts: Optional[list[str]] = None):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModel.from_pretrained(MODEL_NAME)
    if sample_texts is None:
        sample_texts = [
            "I loved this movie, it was brilliant!",
            "This was boring and too long.",
        ]
    enc = tok(sample_texts, padding=True, truncation=True, return_tensors="pt")
    mdl.eval()
    with torch.no_grad():
        out = mdl(**enc)
    last = out.last_hidden_state
    print("tokenizer produced keys:", list(enc.keys()))
    print("last_hidden_state shape:", tuple(last.shape))
    print("[CLS] (first token) embedding shape:", tuple(last[:, 0, :].shape))


# ------------------------------------------------
# Exercise 1.3 — Feature extraction + SVM baseline
# ------------------------------------------------

def extract_cls_embeddings(texts: list[str], tokenizer, model, batch_size: int = 32) -> np.ndarray:
    model.eval()
    all_cls = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        if torch is not None and next(model.parameters()).is_cuda:
            enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_cls.append(cls)
    return np.concatenate(all_cls, axis=0)


def svm_baseline(subset: Optional[int] = None, use_pipeline: bool = False):
    ds = load_dataset_splits(subset)
    tr_texts = ds["train"]["text"]
    tr_y = np.array(ds["train"]["label"])
    va_texts = ds["validation"]["text"]
    va_y = np.array(ds["validation"]["label"])
    te_texts = ds["test"]["text"]
    te_y = np.array(ds["test"]["label"])

    if use_pipeline:
        fe = pipeline("feature-extraction", model=MODEL_NAME, tokenizer=MODEL_NAME)
        def fe_batch(txts):
            outs = fe(txts, padding=True)
            # outs: list of list(seq_len x hidden); take first token
            cls = [np.array(o)[0] for o in outs]
            return np.stack(cls, axis=0)
        Xtr = fe_batch(tr_texts)
        Xva = fe_batch(va_texts)
        Xte = fe_batch(te_texts)
    else:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        mdl = AutoModel.from_pretrained(MODEL_NAME)
        Xtr = extract_cls_embeddings(tr_texts, tok, mdl)
        Xva = extract_cls_embeddings(va_texts, tok, mdl)
        Xte = extract_cls_embeddings(te_texts, tok, mdl)

    clf = LinearSVC()
    clf.fit(Xtr, tr_y)

    def eval_split(X, y, name="set"):
        p = clf.predict(X)
        acc = accuracy_score(y, p)
        prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="binary")
        print(f"{name} — acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    eval_split(Xva, va_y, "validation")
    eval_split(Xte, te_y, "test")


# -------------------------------------
# Exercise 2.1 — Tokenize with .map
# -------------------------------------

def tokenize_and_cache(subset: Optional[int] = None):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset_splits(subset)

    def fn(batch):
        return tok(batch["text"], truncation=True)

    tokenized = ds.map(fn, batched=True)
    # sanity check keys
    sample = tokenized["train"][0]
    assert "input_ids" in sample and "attention_mask" in sample
    os.makedirs(CACHE_DIR, exist_ok=True)
    tokenized.save_to_disk(TOKENIZED_PATH)
    print("Saved tokenized dataset to:", TOKENIZED_PATH)
    return tokenized


# ---------------------------------
# Exercise 2.2 — Build classifier
# ---------------------------------

def build_sequence_classifier():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    return model


# ---------------------------------
# Exercise 2.3 — Trainer + wandb
# ---------------------------------

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def train_with_trainer(
    tokenized_ds: Optional[DatasetDict] = None,
    subset: Optional[int] = None,
    output_dir: str = "runs/distilbert_full",
    use_wandb: bool = False,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
):
    # load tokenized dataset from disk if not provided
    if tokenized_ds is None:
        if os.path.isdir(TOKENIZED_PATH):
            tokenized_ds = load_from_disk(TOKENIZED_PATH)
        else:
            tokenized_ds = tokenize_and_cache(subset)

    # keep only required columns
    cols = [c for c in tokenized_ds["train"].column_names if c in ("input_ids", "attention_mask", "label")]
    tokenized_ds = DatasetDict({k: v.remove_columns([c for c in v.column_names if c not in cols]) for k, v in tokenized_ds.items()})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer)

    model = build_sequence_classifier()

    # Prepare TrainingArguments
    report_to = "wandb" if use_wandb else "none"
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=report_to,
        logging_steps=50,
        seed=SEED,
    )

    # Optional: initialize wandb manually to set project and config
    if use_wandb:
        try:
            import wandb
            run_name = os.path.basename(output_dir)
            wandb.init(project=wandb_project, name=run_name, config=asdict(args))
        except Exception as e:
            print("Warning: could not initialize wandb:", e)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate(tokenized_ds["test"])  # evaluate on test
    print("Test results:")
    for k, v in results.items():
        if k.startswith("eval_"):
            print(f"  {k[5:]}: {v:.4f}")

    # finish wandb run if active
    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass


# ---------------------------
# Exercise 3.1 — LoRA (PEFT)
# ---------------------------

def train_lora(subset: Optional[int] = None):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as e:
        print("PEFT/LoRA not installed. Install with `pip install peft` to use this function.")
        return

    tokenized_ds = load_from_disk(TOKENIZED_PATH) if os.path.isdir(TOKENIZED_PATH) else tokenize_and_cache(subset)
    cols = [c for c in tokenized_ds["train"].column_names if c in ("input_ids", "attention_mask", "label")]
    tokenized_ds = DatasetDict({k: v.remove_columns([c for c in v.column_names if c not in cols]) for k, v in tokenized_ds.items()})

    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    peft_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_lin", "v_lin"], bias="none")
    model = get_peft_model(base, peft_cfg)
    model.print_trainable_parameters()

    # Then you can re-use the train_with_trainer style with this model by passing it into a Trainer.
    print("LoRA model ready. Use Trainer to fine-tune as usual (example omitted here).")


# ------------------
# CLI entrypoint
# ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["e11", "e12", "e13", "e21", "e22", "e23", "e31"], help="Which exercise to run")
    parser.add_argument("--subset", type=int, default=None, help="Use only first N examples per split for quick debugging")
    parser.add_argument("--use_wandb", action="store_true", help="Log training to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--output_dir", type=str, default="runs/distilbert_full")
    args = parser.parse_args()

    set_seed(SEED)

    if args.step == "e11":
        ds = load_dataset_splits(args.subset)
        explore_dataset(ds)
    elif args.step == "e12":
        probe_model_and_tokenizer()
    elif args.step == "e13":
        svm_baseline(subset=args.subset)
    elif args.step == "e21":
        tokenize_and_cache(args.subset)
    elif args.step == "e22":
        m = build_sequence_classifier()
        print(m)
    elif args.step == "e23":
        train_with_trainer(subset=args.subset, output_dir=args.output_dir, use_wandb=args.use_wandb, wandb_project=args.wandb_project)
    elif args.step == "e31":
        train_lora(subset=args.subset)


if __name__ == "__main__":
    main()

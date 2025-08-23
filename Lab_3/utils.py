import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "distilbert/distilbert-base-uncased"
    dataset_name: str = "rotten_tomatoes"
    seed: int = 10
    cache_dir: str = ".cache_lab3"
    tokenized_path: str = ".cache_lab3/tokenized_dataset"

config = Config()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
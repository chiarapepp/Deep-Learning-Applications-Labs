from datasets import load_dataset, DatasetDict
import torch
from tqdm import tqdm
import numpy as np
import os
from typing import Optional, List, Dict
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import compute_metrics, config
import wandb

#--------------------------------------
# EXERCISE 1.1: Dataset Exploration
# -------------------------------------

def load_and_explore_dataset(subset: Optional[int] = None) -> DatasetDict:
    """Load dataset and explore its structure"""
    print(f"\nLoading dataset: {config.dataset_name}")
    
    ds = load_dataset(config.dataset_name)
    
    if subset is not None:
        print(f"Using subset of {subset} samples per split")
        ds = DatasetDict({
            split: data.select(range(min(len(data), subset)))   # Limit to subset size  
            for split, data in ds.items()
        })
    
    print("\nDataset splits and sizes:")
    for split, data in ds.items():
        print(f"  {split}: {len(data):,} samples")
        
        # Label distribution
        labels = data['label']
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        print(f"    Positive: {pos_count:,} ({pos_count/len(labels):.1%})")
        print(f"    Negative: {neg_count:,} ({neg_count/len(labels):.1%})")
    
    print(f"\nSample from training set:")
    sample = ds["train"][0]
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 100:  # Truncate long strings
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")
    
    return ds

# -----------------------------------------------
# EXERCISE 1.2: Model and Tokenizer Exploration
# -----------------------------------------------

def explore_model_and_tokenizer(sample_texts: Optional[List[str]] = None, use_dataset: bool = True):
    print("\nExploration of pre-trained model and tokenizer outputs")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)
    
    if use_dataset:
        ds = load_dataset(config.dataset_name)
        sample_texts = [ds["train"][0]["text"], ds["train"][1]["text"], ds["train"][2]["text"]]
    elif sample_texts is None:
        sample_texts = [
            "This movie was absolutely fantastic! Great acting and plot.",
            "Boring and predictable. Waste of time and money.",
            "An average film with some good moments but nothing special."
        ]
    

    print("Sample texts:")
    for i, text in enumerate(sample_texts):
        print(f"  {i+1}. {text}")
    
    # Tokenize
    encoded = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")
    print(f"\nTokenizer output keys: {list(encoded.keys())}")
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    print(f"Input IDs (first sample): {encoded['input_ids'][0][:10].tolist()}")

    # Show some token information
    print(f"\nExample tokenization of first text:")
    tokens = tokenizer.tokenize(sample_texts[0])
    print(f"  Tokens: {tokens[:10]}...")  # Show first 10 tokens
    print(f"  Total tokens: {len(tokens)}")

    # Get model outputs
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded)
    

    # From Huggingface documentation
    # last_hidden_state (tf.Tensor of shape (batch_size, sequence_length, hidden_size)) 
    # — Sequence of hidden-states at the output of the last layer of the model.

    print(f"\nModel output keys: {list(outputs.keys())}")
    print(f"CLS token embeddings shape: {outputs.last_hidden_state[:, 0, :].shape}")
    print(f"Hidden size: {outputs.last_hidden_state.shape[-1]}")


# -----------------------------------------------
# EXERCISE 1.3: SVM Baseline
# -----------------------------------------------

def extract_cls_features(texts: List[str], tokenizer, model, batch_size: int = 32) -> np.ndarray:
    """Extract CLS token features from texts"""
    model.eval()
    all_features = []
    
    num_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"Extracting features from {len(texts)} texts...")
    
    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        encoded = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**encoded)
            
        # Extract CLS token (first token) from last hidden state
        cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_features.append(cls_features)
            
    return np.concatenate(all_features, axis=0)

def run_svm_baseline(use_wandb: bool = False):
    ds = load_dataset(config.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)
    
    print("\nExtracting features...")
    X_train = extract_cls_features(ds["train"]["text"], tokenizer, model)
    y_train = np.array(ds["train"]["label"])
    
    X_val = extract_cls_features(ds["validation"]["text"], tokenizer, model)  
    y_val = np.array(ds["validation"]["label"])
    
    X_test = None
    y_test = None
    if "test" in ds:
        X_test = extract_cls_features(ds["test"]["text"], tokenizer, model)
        y_test = np.array(ds["test"]["label"])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    if "test" in ds:
        X_test_scaled = scaler.transform(X_test)
    
    if use_wandb:
        wandb.init(
            project="DLA_Lab_3",
            name=f"SVM_baseline_subset",
            config={
                "model": config.model_name,
                "dataset": config.dataset_name,
                "scaler": "StandardScaler",
                "classifier": "LinearSVC",
            }
        )

    print("Training SVM...")
    clf = LinearSVC(random_state=config.seed, max_iter=2000)
    clf.fit(X_train_scaled, y_train)
    
    def evaluate_split(X, y, split_name):
        predictions = clf.predict(X)
        acc = accuracy_score(y, predictions)
        prec, rec, f1, _ = precision_recall_fscore_support(y, predictions, average='binary')
        
        print(f"\n{split_name} Results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        if use_wandb:
            wandb.log({
                f"{split_name}/accuracy": acc,
                f"{split_name}/precision": prec,
                f"{split_name}/recall": rec,
                f"{split_name}/f1": f1
            })

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    
    train_metrics = evaluate_split(X_train_scaled, y_train, "Training")
    val_metrics = evaluate_split(X_val_scaled, y_val, "Validation")
    if "test" in ds:
        test_metrics = evaluate_split(X_test_scaled, y_test, "Test")
    
    return val_metrics.get("accuracy", 0)
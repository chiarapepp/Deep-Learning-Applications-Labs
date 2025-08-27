import wandb
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from models import CNN

"""
Computes and plots the gradient norms of each layer's weights and biases 
for a single minibatch.

This function helps monitor the gradient flow in the network to detect 
issues such as vanishing or exploding gradients. 
"""

def gradient_norm(model, dataloader, device, args):

    data, labels = next(iter(dataloader))
    data = data.to(device)
    labels = labels.to(device)

    model.zero_grad()
    logits = model(data)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    grad_weights = {}
    grad_biases = {}
    for name, param in model.named_parameters():
        print(name)
        if param.grad is not None:
            if "weight" in name:
                grad_weights[name] = param.grad.norm().item()
            elif "bias" in name:
                grad_biases[name] = param.grad.norm().item()

    sorted_weight_layers = sorted(grad_weights.keys())
    sorted_bias_layers = sorted(grad_biases.keys())

    weight_norms = [grad_weights[layer] for layer in sorted_weight_layers]
    bias_norms = [grad_biases[layer] for layer in sorted_bias_layers]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_weight_layers)), weight_norms, color="b", label="Weights", alpha=1)
    plt.bar(range(len(sorted_bias_layers)), bias_norms, color="r", label="Biases", alpha=0.6)

    plt.xlabel("Layers")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm per Layer")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    if args.use_wandb:
        wandb.log({"Gradient Norm Plot": wandb.Image(plt)})

"""
Extracts feature representations from the model for all samples in the dataloader.

This function uses the model's (CNN) feature extractor to obtain latent embeddings 
(512-dimensional vectors) that can later be used to train a separate classifier.
"""

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            feat = model.get_features(x)  # 512-d feature vector
            features.append(feat.cpu().numpy())
            labels.append(y.numpy())
    return np.vstack(features), np.hstack(labels)

# Observe: the outputs are numpy arrays

"""
Function that trains a Linear SVM from sklearn on extracted features 
and evaluate accuracy. Will be used as baseline.
"""

def evaluate_with_svm(train_features, train_labels, val_features, val_labels, test_features=None, test_labels=None):

    # Ensure numpy arrays (in case of lists or improperly converted tensors arrive)
    train_features, train_labels = np.array(train_features), np.array(train_labels)
    val_features, val_labels = np.array(val_features), np.array(val_labels)
    if test_features is not None and test_labels is not None:
        test_features, test_labels = np.array(test_features), np.array(test_labels)

    # Normalize features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    if test_features is not None:
        test_features_scaled = scaler.transform(test_features)

    clf = SVC(kernel='linear')
    clf.fit(train_features_scaled, train_labels)

    val_preds = clf.predict(val_features_scaled)
    val_acc = accuracy_score(val_labels, val_preds)

    results = {"val_acc": val_acc}

    # Optional test accuracy
    if test_features is not None and test_labels is not None:
        test_preds = clf.predict(test_features_scaled)
        test_acc = accuracy_score(test_labels, test_preds)
        results["test_acc"] = test_acc

    return results
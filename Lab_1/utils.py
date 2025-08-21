import wandb
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

"""
Function that trains the model for one epoch.
Computes loss, backpropagation, gradient norm, and logs metrics to Weights & Biases if enabled.
"""

def train_one_epoch(model, dataloader, optimizer, criterion, epoch, args):
    model.train()

    total_loss, total_num = 0.0, 0
    train_bar = tqdm(dataloader, desc=f"[Train Epoch {epoch}]")

    for data, labels in train_bar:
        data = data.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_bar.set_postfix(minibatch_loss=f"{loss.item():.4f}")

        if args.use_wandb:
            wandb.log({"Train Loss": loss.item()})


"""
Function that evaluates the model on a test or validation dataset.
Performs inference, computes cross-entropy loss, and calculates top-1 and top-5 accuracy.
"""

def test(model, dataloader, device):
    model.eval()
    predictions = []
    gts = []
    losses = []

    top5_correct = 0
    total_samples = 0

    with torch.no_grad():
        test_bar = tqdm(dataloader, desc=f"[Test/Validation]")

        for data, labels in test_bar:
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss = F.cross_entropy(logits, labels)
            prob = F.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            # Top-5 accuracy calculation
            top5 = torch.topk(prob, k=5, dim=1).indices
            top5_correct += sum([labels[i].item() in top5[i] for i in range(labels.size(0))])
            total_samples += labels.size(0)

            gts.append(labels.cpu().numpy())
            predictions.append(pred.cpu().numpy())
            losses.append(loss.item())
            test_bar.set_postfix(minibatch_loss=f"{loss.item():.4f}")

    top5_accuracy = top5_correct / total_samples
    top1_accuracy = accuracy_score(np.hstack(gts), np.hstack(predictions))
    avg_loss = np.mean(losses)

    return top1_accuracy, top5_accuracy, avg_loss




"""
Computes and visualizes the gradient norms of each layer's weights and biases 
for a single minibatch.

This is useful for diagnosing vanishing or exploding gradients during training.
Gradients are computed via backpropagation on one batch from the provided dataloader.
A bar plot of the L2 norms of gradients is generated and logged to Weights & Biases (wandb).
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

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(sorted_weight_layers)), weight_norms, color="#4c72b0", label="Weights", alpha=1)
    plt.bar(range(len(sorted_bias_layers)), bias_norms, color="#dd8452", label="Biases", alpha=0.8)

    plt.xlabel("Layers")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm / Layer")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    if args.use_wandb:
        wandb.log({"Gradient Norm Plot": wandb.Image(plt)})

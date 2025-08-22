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

def train_epoch(model, dataloader, optimizer, device, epoch, epochs):
    model.train()
    model = model.to(device)
    losses = []
    
    train_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]", leave=False)
    
    for data, labels in train_bar:
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # Shows the current training loss
        train_bar.set_postfix(minibatch_loss=f"{loss.item():.4f}")

    return np.mean(losses)


def train(model, run_name, optimizer, train_dataloader, val_dataloader, args):
    train_bar = tqdm(range(args.epochs), desc=f"[Training epochs]")
    for epoch in train_bar:
        train_loss = train_epoch(model, train_dataloader, optimizer, args.device, epoch, args.epochs)
        accuracy, _, val_loss = evaluate_model(model, val_dataloader, args.device)

        if args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            scheduler.step()

        if args.use_wandb:
            wandb.log({"Train-" + run_name + "-" + args.dataset: {"Loss": train_loss, "epoch": epoch}, "Validation-" + run_name + "-" + args.dataset: {"Loss": val_loss, "Accuracy":accuracy, "epoch": epoch}})
        train_bar.set_postfix(epoch_loss=f"{train_loss:.4f}")



"""
Function that evaluates the model on a test or validation dataset.
Performs inference, computes cross-entropy loss, and calculates top-1 and top-5 accuracy.
"""

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    ground_truths = []
    losses = []

    top5_correct = 0
    total_samples = 0

    with torch.no_grad():
        test_bar = tqdm(dataloader, desc=f"[Test/Validation]")

        for data, labels in test_bar:
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss = F.cross_entropy(logits, labels)  # Classification Loss
            prob = F.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            # Top-5 accuracy calculation
            top5 = torch.topk(prob, k=5, dim=1).indices
            top5_correct += sum([labels[i].item() in top5[i] for i in range(labels.size(0))])
            total_samples += labels.size(0)

            ground_truths.append(labels.cpu().numpy())
            predictions.append(pred.cpu().numpy())
            losses.append(loss.item())

            test_bar.set_postfix(minibatch_loss=f"{loss.item():.4f}")

    top5_accuracy = top5_correct / total_samples
    top1_accuracy = accuracy_score(np.hstack(ground_truths), np.hstack(predictions))
    avg_loss = np.mean(losses)

    return top1_accuracy, top5_accuracy, avg_loss


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
    plt.bar(range(len(sorted_bias_layers)), bias_norms, color="r", label="Biases", alpha=0.8)

    plt.xlabel("Layers")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm per Layer")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    if args.use_wandb:
        wandb.log({"Gradient Norm Plot": wandb.Image(plt)})

    plt.show()  
    plt.close()
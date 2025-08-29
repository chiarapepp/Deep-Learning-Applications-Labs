import argparse
import datetime
import os
import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import MLP, ResMLP, CNN
from dataloaders import get_dataloaders
from utils import gradient_norm, extract_features, evaluate_with_svm
from train_eval import train, evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Laboratory 1 : Fine-tuning a pre-trained CNN on a new classification task " \
        "with optional linear evaluation."
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=75, help="Number of train epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of the dataloaders")
    parser.add_argument("--freeze_layers", type=str, default="layer1,layer2", 
                    help="Comma-separated list of layer names to freeze")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for the optimizer")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam"], default="SGD", help="Choose optimizer from SGD or ADAM")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument('--use_scheduler', action='store_true',
                       help='Use cosine learning rate scheduler')
    
    parser.add_argument("--path", type=str, default=None, help="Path to the pretrained model of the CNN on CIFAR10")
    
    # CNN-specific arguments
    # [2, 2, 2, 2] like a ResNet18, [3, 4, 6, 3] like a resnet34, [5, 6, 8, 5] like a resnet50
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 2, 2, 2],
                        help="Number of layer pattern for the CNN Model")
    parser.add_argument("--use_residual",action='store_true', help="Use skip connection")

    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation split ratio')
    
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--svm_baseline', action='store_true', help="Compute SVM as baseline")

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    DATASET = "CIFAR100"

    torch.manual_seed(10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(10)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layers_to_freeze = args.freeze_layers.split(",")

    train_dataloader, val_dataloader, test_dataloader, classes, input_size = get_dataloaders(DATASET, args.batch_size, num_workers = args.num_workers, val_ratio=args.val_ratio)

    if args.path is None:
        raise ValueError("Please provide path to pretrained CIFAR10 model with --path")

    # Load the pretrained CNN (CIFAR10)
    model_cifar10 = CNN(args.layers, classes=10, use_residual=True).to(device)
    model_cifar10.load_state_dict(torch.load(args.path, map_location=device))

    # Feature extraction  
    train_features, train_labels = extract_features(model_cifar10, train_dataloader, device)
    validation_features, validation_labels = extract_features(model_cifar10, val_dataloader, device)
    test_features, test_labels = extract_features(model_cifar10, test_dataloader, device)

    print("Feature extraction complete.")

    if args.svm_baseline:
        print("Evaluating baseline with Linear SVM on extracted features")
        
        # Get the accuracy on the validation/test set using a Linear SVM on the extracted features.
        results = evaluate_with_svm(train_features, train_labels, validation_features, validation_labels , test_features, test_labels)
        val_accuracy = results["val_acc"]
        test_accuracy = results["test_acc"]
        print(f"Baseline Linear SVM - Val Acc: {val_accuracy}") 
        print(f"Baseline Linear SVM - Test Acc: {test_accuracy}")

        os.makedirs("Results", exist_ok=True)
        baseline_file = os.path.join("Results", "baseline_svm.txt")
        with open(baseline_file, "a") as f:  
            f.write(f"Run {datetime.datetime.now()}: Val Acc = {val_accuracy:.4f}, Test Acc = {test_accuracy:.4f}\n")

    # Fine-tuning the CNN on CIFAR100
    model = CNN(args.layers, classes=100, use_residual=True).to(device)
    pretrained_dict = {k: v for k, v in model_cifar10.state_dict().items() if "fc" not in k}  # load weights except for the final layer.
    model.load_state_dict(pretrained_dict, strict=False)

    if hasattr(model, "fc"):
        model.fc = torch.nn.Linear(model.fc.in_features, 100)     # new classifier head.
        torch.nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(model.fc.bias)
    else:
        raise AttributeError("The CNN model does not have an attribute 'fc'")

    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layers_to_freeze):
            param.requires_grad = False
            print(f"Freezing layer: {name}")

    run_name = f"finetune_cnn_sched{int(args.use_scheduler)}_freeze_{args.freeze_layers}_lr{args.lr}_opt{args.optimizer}"

    if args.use_wandb:
        wandb.init(
            project='DLA_Lab_1',
            name=run_name,
            config=args
        )

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    train(model, optimizer, train_dataloader, val_dataloader, device, args)

    top1, top5, test_loss = evaluate_model(model, test_dataloader, device)

    if args.use_wandb:
        wandb.log({
            "test_accuracy": top1,
            "test_top5_accuracy": top5
        })


    print(f"Final Test Results:")
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    os.makedirs("Models", exist_ok=True)
    torch.save(model.state_dict(), "Models/"+run_name+".pth")

if __name__ == "__main__":
    main()
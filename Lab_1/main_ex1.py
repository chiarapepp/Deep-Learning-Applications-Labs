import argparse
from html import parser
import os
import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import MLP, ResMLP, CNN
from dataloaders import get_dataloaders
from utils import gradient_norm
from train_eval import train, evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Laboratory 1 : Parameters to train a Model (MLP/ResMLP/CNN)"
    )

    # Model arguments
    parser.add_argument('--model', type=str, default='mlp', 
                       choices=['mlp', 'resmlp', 'cnn'],
                       help='Model type to train')
    parser.add_argument('--dataset', type=str, default='MNIST',
                       choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                       help='Dataset to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')

    parser.add_argument('--normalization', action='store_true',
                       help='Use batch normalization')
    parser.add_argument('--use_scheduler', action='store_true',
                       help='Use cosine learning rate scheduler')
    
    # MLP specific arguments
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=None,
                       help='Custom hidden layer sizes for MLP (list of integers), if not provided MLP uses width and depth parameters')
    parser.add_argument('--width', type=int, default=128,
                       help='Width of the MLP or the ResMLP')
    parser.add_argument('--depth', type=int, default=2,
                       help='Depth of the MLP or the ResMLP')
    
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

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    torch.manual_seed(10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataloader, val_dataloader, test_dataloader, classes, input_size = get_dataloaders(args.dataset, args.batch_size, num_workers = args.num_workers, val_ratio=args.val_ratio)

    if args.model == "mlp":
        if args.hidden_sizes:  
            hidden = args.hidden_sizes
            run_name = f"mlp_custom_{hidden}_n{int(args.normalization)}_sched{int(args.use_scheduler)}"
        else:
            hidden = [args.width] * args.depth
            run_name = f"mlp_w{args.width}_d{args.depth}_n{int(args.normalization)}_sched{int(args.use_scheduler)}"

        model = MLP(input_size, hidden, classes, args.normalization)

    elif args.model == "resmlp":
        model = ResMLP(input_size, args.width, args.depth, classes, args.normalization)
        run_name = f"resmlp_w{args.width}_d{args.depth}_n{int(args.normalization)}_sched{int(args.use_scheduler)}"

    elif args.model == "cnn":
        model = CNN(args.layers, classes, args.use_residual)
        layers_str = "-".join(map(str, args.layers))
        run_name = f"cnn_skip{int(args.use_residual)}_L{layers_str}_sched{int(args.use_scheduler)}"

    if args.use_wandb:
        wandb.init(
            project='DLA_Lab_1',
            name=run_name,
            config=args
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    train(model, optimizer, train_dataloader, val_dataloader, device, args)

    top1, top5, test_loss = evaluate_model(model, test_dataloader, device)
    if args.use_wandb:
        wandb.log({
        "test_accuracy": top1,
        "test_top5_accuracy": top5
        }, step=args.epochs)

    print(f"Final Test Results:")
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    print("\nComputing gradient norms...")
    gradient_norm(model, train_dataloader, device, args)

    os.makedirs("Models", exist_ok=True)
    torch.save(model.state_dict(), "Models/"+run_name+".pth")

if __name__ == "__main__":
    main()
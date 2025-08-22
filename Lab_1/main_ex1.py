import argparse
import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import MLP, ResMLP
from dataloaders import get_dataloaders
from utils import train, evaluate_model, gradient_norm
import torch.nn as nn

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
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    
    # Model-specific arguments
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 64],
                       help='Hidden layer sizes for MLP (list of integers)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for ResMLP')
    parser.add_argument('--num_blocks', type=int, default=3,
                       help='Number of residual blocks for ResMLP')
    parser.add_argument('--normalization', action='store_true',
                       help='Use batch normalization')
    
    # Data arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation split ratio')
    
    parser.add_argument('--use_scheduler', action='store_true',
                       help='Use cosine learning rate scheduler')
    '''
    parser.add_argument('--scheduler_type', type=str, default='step',
                       choices=['step', 'cosine'],
                       help='Type of scheduler')
    '''

    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=10,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    torch.manual_seed(args.seed)

    train_dataloader, val_dataloader, test_dataloader, classes, input_size = get_dataloaders(args.dataset, args.batch_size, num_workers = args.num_workers, val_ratio=args.val_ratio)

    if args.model == "mlp":
        model = MLP(input_size, args.hidden_sizes, classes, args.normalization)
        run_name = f"mlp_w{args.hidden_sizes}_d{len(args.hidden_sizes)}"
    elif args.model == "resmlp":
        model = ResMLP(input_size, args.hidden_dim, args.num_blocks, classes, args.normalization)
        run_name = f"resmlp_w{args.hidden_dim}_b{args.num_blocks}"
    '''    
    elif args.model == "cnn":
        layers = args.layers
        model = CustomCNN(block_type="basic", layers=layers, use_skip=args.skip)
        run_name = f"cnn_skip{args.skip}_layers{layers}"
    '''    
    if args.use_wandb:
        run_name = f"{args.model}_{args.dataset}_{args.epochs}epochs"
        wandb.init(
            project='DLA_Lab_1',
            name=run_name,
            config=args
        )

    model = model.to(args.device)
    data, _ = next(iter(train_dataloader))
    data = data.to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    train(model, run_name, optimizer, train_dataloader, val_dataloader, args)

    top1, top5, _ = evaluate_model(model, test_dataloader, args.device)
    if args.use_wandb:
        wandb.log({"Test-" + run_name + "-" + args.dataset: {"Accuracy": top1, "Top-5 Accuracy": top5}})
    print("Test accuracy: {}".format(top1))

    gradient_norm(model, train_dataloader, args.device, args)

    torch.save(model.state_dict(), "Models/"+run_name+".pth")

if __name__ == "__main__":
    main()
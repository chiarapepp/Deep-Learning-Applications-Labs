import argparse
import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import MLP, ResMLP
from data import get_dataloaders
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a MLP/ResMLP/CNN on MNIST/CIFAR10/CIFAR100"
    )

    parser.add_argument("--epochs", type=int, default=50, help="Number of train epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of the dataloaders")

    parser.add_argument(
        "--schedule",
        default=[35],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )
    parser.add_argument("--normalization", default=True, help="If True use normalization layers (this choice is only available for the MLP model)")

    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

    device = "cuda" if torch.cuda.is_available else "cpu"
    parser.add_argument(
        "--device",
        default=device,
        type=str,
        metavar="D",
        help="The device where the training process will be executed. Autoset to cuda if cuda is available on the local machine",
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        type=str,
        metavar="dataset",
        help="Training/testing dataset",
    )
    parser.add_argument(
        "--val-split",
        default=0.1,
        type=float,
        metavar="v",
        help="Proportion of the training data to be reserved to the validation split.",
    )
    parser.add_argument(
        "--seed",
        default=69,
        type=int,
        metavar="seed",
        help="Seed of the experiments, to be set for reproducibility",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "resmlp", "cnn"],
        default="mlp",
        help="Choose model architecture: mlp, resmlp, or cnn",
    )
    parser.add_argument(
        "--width",
        default=16,
        type=int,
        metavar="w",
        help="Width of the neural network(hidden dimension)",
    )
    parser.add_argument(
        "--depth",
        default=2,
        type=int,
        metavar="depth",
        help="Depth of the neural network(number of layers)",
    )
    parser.add_argument("--skip", action="store_true", help="Whether to use skip connnections inside the BasicBlock of a CNN")
    parser.add_argument("--layers", default=[2, 2, 2, 2], nargs="*", type=int, help="Layer configuration for the custom CNN")

    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging. If not provided, wandb will be disabled.")

    args = parser.parse_args()
    return args


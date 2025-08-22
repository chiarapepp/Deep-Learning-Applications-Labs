import argparse
import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import MLP, ResMLP
from Lab_1.dataloaders import get_dataloaders
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Model (MLP/ResMLP/CNN)"
    )

    args = parser.parse_args()
    return args


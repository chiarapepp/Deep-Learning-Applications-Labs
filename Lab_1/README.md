# Laboratory 1 - Deep Neural Networks: MLPs, ResMLPs, and CNNs

This lab focuses on training deep models—**MLPs**, **Residual MLPs**, and **CNNs**—on standard image classification datasets (MNIST, CIFAR10). The key objectives are:

- Reproducing results (at a smaller scale) from:
  - 📄 [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) (He et al., CVPR 2016)
- Understanding the effect of residual connections on model performance.
- Investigating training dynamics and gradient flow.

---

## Project Structure

```
Lab_1/
├── main_ex1.py          # Main script for Exercise 1 (MLP/CNN experiments)
├── main_ex2.py          # Main script for Exercise 2 (Fine-tuning experiments)
├── models.py            # Model implementations (MLP, ResMLP, CNN)
├── dataloaders.py       # Data loading utilities for MNIST, CIFAR-10, CIFAR-100
├── train_eval.py        # Training and evaluation functions
├── utils.py             # Utility functions (gradient analysis, feature extraction)
├── run_experiments.sh   # Complete experiment runner script
├── Models/              # Directory for saved model weights
└── README.md            # This file
```


## 📋 Project Overview

The laboratory is divided into two main exercises:

**Exercise 1**: Verification of ResNet findings on MLPs and CNNs
- **1.1**: Baseline MLP implementation
- **1.2**: MLP with residual connections (ResMLP)
- **1.3**: CNN with/without residual connections

**Exercise 2**: Fine-tuning and transfer learning
- **2.1**: Pre-trained model fine-tuning from CIFAR-10 to CIFAR-100


## 🧠 Model Implementations

### 1. MLP (Multilayer Perceptron)
- Configurable depth and width
- Optional batch normalization
- Standard feedforward architecture

### 2. ResMLP (Residual MLP)
- Residual blocks with skip connections
- Identical capacity to MLP for fair comparison
- Demonstrates residual learning benefits on MLPs

### 3. CNN (Convolutional Neural Network)
- ResNet-style architecture with BasicBlocks
- Configurable layer patterns: [2,2,2,2], [3,4,6,3], [5,6,8,5]
- Optional residual connections (can be disabled for comparison)
- Feature extraction capabilities for transfer learning

## 📊 Datasets

- **MNIST**: 28×28 grayscale digit classification (10 classes)
- **CIFAR-10**: 32×32 color image classification (10 classes)
- **CIFAR-100**: 32×32 color image classification (100 classes)

## 🚀 Usage

### Quick Start

1. **Install dependencies**:
```bash
pip install torch torchvision tqdm matplotlib scikit-learn wandb numpy
```

2. **Login to Weights & Biases** (optional but recommended):
```bash
wandb login
```

3. **Run all experiments**:
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### Individual Experiments

#### Exercise 1.1 & 1.2: MLP Experiments
```bash
# Standard MLP on MNIST
python main_ex1.py --model mlp --dataset MNIST --depth 10 --width 128 --epochs 50

# ResMLP with residual connections
python main_ex1.py --model resmlp --dataset MNIST --depth 10 --width 128 --epochs 50

# With batch normalization and scheduler
python main_ex1.py --model resmlp --dataset MNIST --depth 20 --width 128 --normalization --use_scheduler --epochs 50
```

#### Exercise 1.3: CNN Experiments
```bash
# CNN with residual connections (ResNet-18 style)
python main_ex1.py --model cnn --dataset CIFAR10 --layers 2 2 2 2 --use_residual --epochs 75

# CNN without residual connections
python main_ex1.py --model cnn --dataset CIFAR10 --layers 2 2 2 2 --epochs 75

# Deeper CNN (ResNet-34 style)
python main_ex1.py --model cnn --dataset CIFAR10 --layers 3 4 6 3 --use_residual --use_scheduler --epochs 75
```

#### Exercise 2.1: Fine-tuning Experiments
```bash
# Linear evaluation (freeze all layers)
python main_ex2.py --path Models/your_pretrained_model.pth --freeze_layers "layer1,layer2,layer3,layer4" --optimizer SGD --lr 1e-3 --epochs 75

# Fine-tuning (unfreeze last layers)
python main_ex2.py --path Models/your_pretrained_model.pth --freeze_layers "layer1,layer2" --optimizer Adam --lr 1e-3 --use_scheduler --epochs 75
```

### Command Line Arguments

#### Common Arguments
- `--epochs`: Number of training epochs (default: 50 for MLPs, 75 for CNNs)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--num_workers`: Data loading workers (default: 4)
- `--use_wandb`: Enable Weights & Biases logging
- `--use_scheduler`: Use cosine annealing scheduler

#### MLP/ResMLP Specific
- `--model`: Choose 'mlp' or 'resmlp'
- `--depth`: Number of hidden layers
- `--width`: Hidden layer size
- `--normalization`: Enable batch normalization

#### CNN Specific
- `--layers`: Layer pattern (e.g., 2 2 2 2 for ResNet-18)
- `--use_residual`: Enable residual connections

#### Fine-tuning Specific
- `--path`: Path to pre-trained model
- `--freeze_layers`: Comma-separated layer names to freeze
- `--optimizer`: 'SGD' or 'Adam'

## 🔬 Key Experiments and Expected Results

### Exercise 1: Verifying ResNet Claims

1. **Depth vs Performance**: Deeper networks without residual connections may perform worse
2. **Residual Benefits**: Networks with residual connections should train more effectively
3. **Gradient Flow**: Gradient analysis shows better flow in residual networks

### Exercise 2: Transfer Learning

1. **Feature Quality**: SVM baseline on extracted features
2. **Fine-tuning Strategies**: Comparison of different unfreezing strategies
3. **Optimizer Impact**: SGD vs Adam for fine-tuning tasks

## 📈 Monitoring and Results

### Weights & Biases Integration
- Automatic logging of training/validation metrics
- Gradient norm visualization
- Model comparison dashboards
- Hyperparameter tracking

### Local Logging
- Individual experiment logs in `logs/` directory
- Model checkpoints saved in `Models/` directory
- Gradient analysis plots

## 🔧 Implementation Details

### Key Features
- **Modular Design**: Easy to extend and modify
- **Gradient Analysis**: Built-in gradient norm computation
- **Feature Extraction**: CNN feature extraction for transfer learning
- **Flexible Architecture**: Configurable model depths and widths
- **Reproducibility**: Fixed random seeds for consistent results

### Technical Highlights
- Custom ResNet implementation with optional skip connections
- Efficient data loading with proper normalization
- Top-1 and Top-5 accuracy computation
- Learning rate scheduling support
- Memory-efficient feature extraction

## 📚 Key Findings

The experiments demonstrate:

1. **Residual connections enable training of deeper networks** without degradation
2. **Gradient flow is improved** in networks with skip connections
3. **Transfer learning benefits** from pre-trained feature representations
4. **Fine-tuning strategies** significantly impact performance on new tasks




## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 201
- [](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L204)
## ⚙️ How to Run

You can run training from the command line using `main.py`. The script is fully configurable via command-line arguments. 

When using the MLP model, there are two main ways to define its architecture:

Custom architecture with `--hidden_sizes`: You can provide a list of integers specifying the size of each hidden layer. This allows you to experiment freely with any layer configuration you like.

Standard architecture with `--width` and `--depth`: If you want a setup that can be directly compared with ResMLP for a coherent evaluation of model structures, use the width (number of neurons per hidden layer) and depth (number of hidden layers) parameters. This ensures both MLP and ResMLP have a comparable number of layers and units. 

Additionally, you can enable batch normalization after each hidden layer using the normalization flag.

### 🔧 Example: Train a ResMLP on MNIST

```bash
python main.py --model resmlp --dataset mnist --depth 4 --width 64 --bn --use-wandb
```

### 🔧 Example: Train a CNN with skip connections on CIFAR10

```bash
python main.py --model cnn --dataset cifar10 --skip --layers 2 2 2 2 --use-wandb
```

---

## 🧠 `main.py` - Supported Arguments

This script provides an entry point to train MLP, ResMLP(MLP with Residual connections), CNNs and ResNet models. 
The implementation of the CNN models relies on the 'BasicBlock' definition in [torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L59)

| Argument         | Description |
|------------------|-------------|
| `--model`        | Model type to use: `mlp`, `resmlp`, `cnn` |
| `--dataset`      | Dataset to use: `mnist`, `cifar10`, `cifar100` |
| `--lr`           | Learning rate (default: 0.001) |
| `--epochs`       | Number of training epochs (default: 50) |
| `--batch-size`   | Batch size (default: 256) |
| `--depth`        | Number of hidden layers (MLP/ResMLP only) |
| `--width`        | Width of hidden layers (MLP/ResMLP only) |
| `--bn`           | Use BatchNorm (only MLP/ResMLP) |
| `--skip`         | Enable skip connections in CNN |
| `--layers`       | CNN block configuration (default: `[2 2 2 2]`) |
| `--schedule`     | LR drop milestones (e.g., `--schedule 35`) |
| `--cos`          | Use cosine LR scheduling |
| `--val-split`    | Fraction of training data used for validation (default: 0.1) |
| `--seed`         | Set random seed for reproducibility |
| `--device`       | Device to use: `cpu` or `cuda` |
| `--use-wandb`    | Enable Weights & Biases logging |


## 🧠 `main_cam.py` – Supported Arguments

This script generates Class Activation Maps (CAMs) using a pretrained ResNet18 on the **Imagenette** dataset.

| Argument        | Aliases                          | Type   | Default | Description |
|----------------|----------------------------------|--------|---------|-------------|
| `--class_index` | `--cls_index`, `--class_idx`, `--cls` | `int`  | `0`     | Class index of the input image. Choose from:<br> `tench (0)`, `English springer (1)`, `cassette player (2)`, `chainsaw (3)`, `church (4)`, `French horn (5)`, `garbage truck (6)`, `gas pump (7)`, `golf ball (8)`, `parachute (9)` |
| `--sample_index` | `--sample_idx`, `--sample`       | `int`  | `5`     | Index of the image sample to visualize within the selected class. |
| `--url`         | –                                | `str`  | `""`    | URL of a custom input image from the Imagenette dataset. If provided, it overrides `--class_index` and `--sample_index`. |

> 📌 Note: If you provide a URL, the `class_index` and `sample_index` are ignored.

---

## 📊 Visualizations & Logging

If `--use-wandb` is enabled, training metrics and model summaries are logged to:

🔗 [W&B Project – Lab 1](https://wandb.ai/jaysenoner/lab_1_DLA?nw=nwuserjaysenoner1999)

You can view:

- Learning curves
- Validation accuracy
- Test Top-1 and Top-5 accuracy
- Parameter counts
- Model summaries

---

## 🧪 Results Summary

Key findings include:

- Residual connections significantly improve training stability and final accuracy, especially for deeper networks.
- Residual connections allow the gradients to backpropagate to earlier layers of the network with a stronger signal
- CNNs with skip connections demonstrate higher accuracy on CIFAR datasets. Moreover, CNNs with skip connections are less prone to overfitting.
- This trend is exxagerated when the depth of the convolutional network is increased.
- Class Activation Maps allow localization of discriminative image regions without additional supervision.


More material that supports those findings can be found inside the `wandb` project. 

### 📈 Gradient Flow on Network Layers

#### MLP vs ResMLP on MNIST

<p align="center">
  <img src="images/gradient_norm_mlp.png" alt="MLP_norm" width="45%" style="margin-right:10px;"/>
  <img src="images/gradient_norm_resmlp.png" alt="ResMLP_norm" width="45%"/>
</p>

#### CNN vs ResNet on CIFAR10

<p align="center">
  <img src="images/cnn_noskip_gradient_norm.png" alt="CNN_norm" width="45%" style="margin-right:10px;"/>
  <img src="images/cnn_skip_gradient_norm.png" alt="ResNet_norm" width="45%"/>
</p>


### 🔍 Class Activation Maps (CAMs on Imagenette)

<p align="center">
  <img src="images/cam_church.jpg" alt="CNN_norm" width="45%" style="margin-right:10px;"/>
  <img src="images/cam_french_horn.jpg" alt="ResNet_norm" width="45%"/>
</p>

<p align="center">
  <img src="images/cam_gas_pump.jpg" alt="CNN_norm" width="45%" style="margin-right:10px;"/>
  <img src="images/imagenette_CAM_result.jpg" alt="ResNet_norm" width="45%"/>
</p>








-----------------------------------
'''
# =============================================================================
# EXERCISE 2.1: FINE-TUNING EXPERIMENTS 
# =============================================================================

# -----------------------------------------------------------------------------
# Linear Evaluation - Freeze All Layers 
# -----------------------------------------------------------------------------
echo "🔹 Linear Evaluation - Freeze ALL layers"
echo "  └── Testing SGD vs Adam with different learning rates"
echo ""
python main_ex2.py --lr 1e-3 --optimizer SGD --use_scheduler --freeze_layers "layer1,layer2,layer3,layer4" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-3 --optimizer SGD --freeze_layers "layer1,layer2,layer3,layer4" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer SGD --use_scheduler --freeze_layers "layer1,layer2,layer3,layer4" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer SGD --freeze_layers "layer1,layer2,layer3,layer4" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-3 --optimizer Adam --use_scheduler --freeze_layers "layer1,layer2,layer3,layer4" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-3 --optimizer Adam --freeze_layers "layer1,layer2,layer3,layer4" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer Adam --use_scheduler --freeze_layers "layer1,layer2,layer3,layer4" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer Adam --freeze_layers "layer1,layer2,layer3,layer4" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2

# -----------------------------------------------------------------------------
# Partial Fine-tuning - Freeze Early Layers (8 experiments)
# -----------------------------------------------------------------------------
echo "🔹 Partial Fine-tuning - Freeze layer1,layer2 (8 experiments)..."
echo "  └── Unfreeze layer3,layer4 for adaptation"
echo ""
python main_ex2.py --lr 1e-3 --optimizer SGD --use_scheduler --freeze_layers "layer1,layer2" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-3 --optimizer SGD --freeze_layers "layer1,layer2" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer SGD --use_scheduler --freeze_layers "layer1,layer2" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer SGD --freeze_layers "layer1,layer2" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-3 --optimizer Adam --use_scheduler --freeze_layers "layer1,layer2" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-3 --optimizer Adam --freeze_layers "layer1,layer2" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer Adam --use_scheduler --freeze_layers "layer1,layer2" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer Adam --freeze_layers "layer1,layer2" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2

# -----------------------------------------------------------------------------
# Full Fine-tuning - Freeze Only First Layer (8 experiments)
# -----------------------------------------------------------------------------
echo "🔹 Full Fine-tuning - Freeze only layer1 (8 experiments)..."
echo "  └── Unfreeze layer2,layer3,layer4 for full adaptation"
echo ""
python main_ex2.py --lr 1e-3 --optimizer SGD --use_scheduler --freeze_layers "layer1" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-3 --optimizer SGD --freeze_layers "layer1" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer SGD --use_scheduler --freeze_layers "layer1" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer SGD --freeze_layers "layer1" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-3 --optimizer Adam --use_scheduler --freeze_layers "layer1" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-3 --optimizer Adam --freeze_layers "layer1" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer Adam --use_scheduler --freeze_layers "layer1" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2
python main_ex2.py --lr 1e-2 --optimizer Adam --freeze_layers "layer1" --path "Models/cnn_skip_True_layers[2, 2, 2, 2].pth" --layers 2 2 2 2

echo "✅ Fine-tuning experiments completed! (108/108)"
echo ""


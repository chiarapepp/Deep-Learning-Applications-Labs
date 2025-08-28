# Laboratory 1 - Deep Neural Networks: MLPs, ResMLPs, and CNNs

## Overview
This laboratory explores deep neural network architectures and their training dynamics, with a focus on training deep models—MLPs, Residual MLPs, and CNNs—on standard image classification datasets (MNIST, CIFAR10). The key objectives are:

- Reproducing results (at a smaller scale) from the paper:[*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385)
- Understanding the effect of residual connections on model performance.
- Investigating training dynamics and gradient flow.
- Exploring transfer learning through fine-tuning techniques.


### Project Structure

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

### Implemented Models
- 1. **MLP (Multilayer Perceptron)**
  - Configurable hidden layer sizes.
  - Optional batch normalization.
  - Standard feedforward architecture.
- 2. **ResMLP (Residual MLP)**
  - MLP with residual connections between blocks.
  - Each block contains two linear layers with skip connections.
  - Configurable depth and width.
  - Demonstrates residual learning benefits on MLPs.
- 3. **CNN (Convolutional Neural Network)**
  - ResNet-style architecture with BasicBlocks
  - Configurable layer patterns: [2,2,2,2], [3,4,6,3], [5,6,8,5]
  - Optional residual connections (can be disabled for comparison)
  - Feature extraction capabilities for transfer learning

### Datasets

- **MNIST**: 28×28 grayscale digit classification (10 classes)
- **CIFAR-10**: 32×32 color image classification (10 classes)
- **CIFAR-100**: 32×32 color image classification (100 classes)

## Usage

### Quick Start

1. **Install dependencies**:
```bash
pip install torch torchvision tqdm matplotlib scikit-learn wandb numpy
```

2. **Login to Weights & Biases** (optional but recommended):
```bash
wandb login
```


## Exercise 1: Verification of ResNet findings on MLPs and CNNs
Train and evaluate MLPs and CNNs on MNIST/CIFAR-10 with various depths, widths, normalization, residual connections, and schedulers.

### MLP
The training can be done from the command line using main_ex1.py. The script is fully configurable via command-line arguments.

```bash
# Standard MLP on MNIST
python main_ex1.py --model mlp --dataset MNIST --depth 10 --width 128 --epochs 50

# ResMLP with residual connections
python main_ex1.py --model resmlp --dataset MNIST --depth 10 --width 128 --epochs 50

# With batch normalization and scheduler
python main_ex1.py --model resmlp --dataset MNIST --depth 20 --width 128 --normalization --use_scheduler --epochs 50
```
When using the MLP (not ResMLP) model, there are two main ways to define its architecture: 

Custom architecture with `--hidden_sizes`: You can provide a list of integers specifying the size of each hidden layer. This allows you to experiment freely with any layer configuration you like.

Standard architecture with `--width` and `--depth`: If you want a setup that can be directly compared with ResMLP for a coherent evaluation of model structures, use the width (number of neurons per hidden layer) and depth (number of hidden layers) parameters. This ensures both MLP and ResMLP have a comparable number of layers and units. 

All the arguments possibilmente usabili sono di seguito. 

#### Common Arguments
- `--epochs`: Number of training epochs (default: 50 for MLPs, 75 for CNNs).
- `--batch_size`: Batch size (default: 128).
- `--lr`: Learning rate (default: 0.001).
- `--num_workers`: Data loading workers (default: 4).
- `--use_wandb`: Enable Weights & Biases logging.
- `--use_scheduler`: Use cosine annealing scheduler.

#### MLP/ResMLP Specific
- `--model`: Choose 'mlp' or 'resmlp'.
- `--hidden_size`: Only for 'mlp', custom architecture (list of integers specifying the size of each hidden layer). 
- `--depth`: Number of hidden layers.
- `--width`: Hidden layer size.
- `--normalization`: Enable batch normalization.

### CNN 
The implementation of the CNN models relies on the 'BasicBlock' definition in [torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L59).
To train the CNN, it's possible to use 

```bash
# CNN with residual connections (ResNet-18 style)
python main_ex1.py --model cnn --dataset CIFAR10 --layers 2 2 2 2 --use_residual --epochs 75

# CNN without residual connections
python main_ex1.py --model cnn --dataset CIFAR10 --layers 2 2 2 2 --epochs 75

# Deeper CNN (ResNet-34 style)
python main_ex1.py --model cnn --dataset CIFAR10 --layers 3 4 6 3 --use_residual --use_scheduler --epochs 75
```

Oltre common arguments visti sopra ci sono tali arguments specifici per la CNN.

#### CNN Specific
- `--model`: 'cnn' (obbligatorio).
- `--layers`: Layer pattern (e.g., 2 2 2 2 for ResNet-18).
- `--use_residual`: Enable residual connections.

## Exercise 2: Pre-trained model fine-tuning from CIFAR-10 to CIFAR-100



#### Fine-tuning Experiments
```bash
# Linear evaluation (freeze all layers)
python main_ex2.py --path Models/your_pretrained_model.pth --freeze_layers "layer1,layer2,layer3,layer4" --optimizer SGD --lr 1e-3 --epochs 75

# Fine-tuning (unfreeze last layers)
python main_ex2.py --path Models/your_pretrained_model.pth --freeze_layers "layer1,layer2" --optimizer Adam --lr 1e-3 --use_scheduler --epochs 75
```

#### Fine-tuning Specific
- `--path`: Path to pre-trained model
- `--freeze_layers`: Comma-separated layer names to freeze
- `--optimizer`: 'SGD' or 'Adam'



3. **Run all experiments**:
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```






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



## 📚 Key Findings

The experiments demonstrate:

1. **Residual connections enable training of deeper networks** without degradation
2. **Gradient flow is improved** in networks with skip connections
3. **Transfer learning benefits** from pre-trained feature representations
4. **Fine-tuning strategies** significantly impact performance on new tasks




## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 201
- [](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L204)



 
## Visualizations & Logging

If `--use_wandb` is enabled, training metrics and model summaries are logged to:

🔗 [W&B Project – DLA_Lab_1](https://wandb.ai/chiara-peppicelli-university-of-florence/DLA_Lab_1?nw=nwuserchiarapeppicelli)

You can view:

- TO ADD

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







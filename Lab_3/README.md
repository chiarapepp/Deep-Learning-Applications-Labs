# Lab 3: Working with Transformers in the HuggingFace Ecosystem

## Overview

This laboratory explores the HuggingFace ecosystem for adapting pre-trained transformer models to new tasks. We'll work through sentiment analysis using DistilBERT, progressing from basic feature extraction to full fine-tuning and parameter-efficient methods.

This lab focuses on adapting pre-trained transformers from the HuggingFace ecosystem to new tasks. 
In this laboratory, the Rotten Tomatoes movie review dataset is employed for sentiment analysis. The activities carried out include:

- Exploration of the dataset and probing of pre-trained models.
- Construction of a stable baseline using a pre-trained DistilBERT model.
- Tokenization of the dataset and fine-tuning of the model with HuggingFace Trainer.
- Application of parameter-efficient fine-tuning using LoRA.


## Setup and Installation

### Prerequisites
Should have been already installed in the requirements.txt. 
However should have be: 
```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install scikit-learn numpy pandas
pip install wandb peft  
```

### Running Experiments

All experiments are managed through a single script main.py. The `--step` argument selects the exercise to run.

```bash
# Exercise 1
# Inspect dataset and explore pre-trained model (Exercise 1.1 and 1.2)
python main.py --step e11
python main.py --step e12

# Run SVM baseline (Exercise 1.3)
python main.py --step e13 

# Tokenize dataset for fine-tuning (Exercise 2.1)
python main.py --step e21

# Fine-tune DistilBERT with Trainer (Exercise 2.3)
python main.py --step e23 --epochs 3 --batch_size 16 --use_wandb

# Fine-tune DistilBERT using LoRA (Exercise 3.1)
python main.py --step e31 --lora_rank 8 --lora_alpha 16
```
#### Arguments 

- `--subset`: It's possible to use a smaller subset of the dataset for faster testing.
- `--lr`: Learning rate for fine-tuning (default: 2e-5).
- `--epochs`: Number of training epochs (default: 3).
- `--batch_size`: Training batch size (default: 16).
- `--lora_alpha`: LoRA alpha parameter (default: 32).
- `--lora_rank`: LoRA rank parameter (default: 8).
- `--use_wandb`: Enable Weights & Biases logging.
- `--output_dir`: Directory for saving models and logs.


## Observations and Results

### Exercise 1: Dataset and Model Exploration

#### 1.1 Dataset Exploration
**Goal**: Understand the Rotten Tomatoes movie review dataset structure.

**What we do**:
- Load the dataset using `datasets.load_dataset("rotten_tomatoes")`
- Explore splits (train/validation/test) and their sizes
- Analyze label distribution (positive/negative reviews)
- Examine sample data structure

**Key findings**:
- 5,331 positive and 5,331 negative movie reviews
- Balanced dataset with 50/50 split
- Text samples are preprocessed sentences



#### 1.2 Model and Tokenizer Investigation
**Goal**: Understand how DistilBERT processes text inputs.

**What we do**:
- Load DistilBERT model and tokenizer using `AutoModel` and `AutoTokenizer`
- Tokenize sample texts to see input format
- Pass tokens through the model to examine outputs
- Understand the `[CLS]` token and hidden states structure

**Key insights**:
- DistilBERT produces contextualized embeddings for each token
- The `[CLS]` token (first token) represents the entire sequence
- Output shape: `(batch_size, sequence_length, hidden_size)`
- Hidden size is 768 for DistilBERT-base


#### 1.3 SVM Baseline
**Goal**: Establish a performance baseline using traditional ML on transformer features.

**Approach**:
1. Extract `[CLS]` token embeddings from all text samples
2. Use these 768-dimensional features to train a linear SVM
3. Evaluate on validation/test sets

**Why this matters**:
- Provides a stable baseline for comparison
- Shows the power of pre-trained representations even without fine-tuning
- Fast to train and evaluate

**Expected results**:
- Validation accuracy: ~80-85%
- Test accuracy: ~78-82%
- Much faster than full fine-tuning



### Exercise 2: Fine-tuning DistilBERT

#### 2.1 Dataset Tokenization
**Goal**: Prepare the dataset for transformer training by tokenizing all text.

**Process**:
- Apply tokenizer to all dataset splits using `Dataset.map()`
- Add `input_ids` and `attention_mask` to each sample
- Maintain original `text` and `label` fields
- Cache tokenized data for reuse

**Technical details**:
- Use `truncation=True` to handle long sequences
- Don't pad here - padding handled dynamically during training
- Batched processing for efficiency



#### 2.2 Model Setup
**Goal**: Configure DistilBERT for sequence classification.

**Key concept**: 
- `AutoModelForSequenceClassification` automatically adds a classification head
- Random initialization of the classification layer
- Pre-trained weights for the transformer backbone

#### 2.3 Fine-tuning with Trainer
**Goal**: Train the entire model end-to-end for optimal performance.

**Training setup**:
- Learning rate: 2e-5 (typical for transformers)
- Batch size: 16-32 (adjust for your hardware)
- Epochs: 3-5 (avoid overfitting)
- Evaluation every epoch
- Save best model based on accuracy

**Expected improvements**:
- Should achieve 85-90% accuracy
- 5-10% improvement over SVM baseline
- Takes significantly longer to train



### Exercise 3: Advanced Techniques

#### 3.1 LoRA (Parameter-Efficient Fine-tuning)
**Goal**: Achieve similar performance with much fewer trainable parameters.

**LoRA concept**:
- Only fine-tune low-rank adaptation matrices
- Typically 0.1-1% of original parameters
- Faster training, less memory usage
- Often achieves 95%+ of full fine-tuning performance

**Configuration**:
- Rank (r): 8-16 (controls adaptation capacity)
- Alpha: 16-32 (scaling factor)
- Target modules: attention layers (`q_lin`, `v_lin`, etc.)

**Benefits**:
- 10x faster training
- Much less GPU memory required
- Easy to switch between different adaptations

## 📊 Expected Results Comparison

| Method | Parameters Trained | Training Time | Validation Acc | Test Acc |
|--------|-------------------|---------------|----------------|----------|
| SVM Baseline | ~1M (SVM only) | 2-5 minutes | ~82% | ~79% |
| Full Fine-tuning | ~67M (all) | 15-30 minutes | ~88% | ~85% |
| LoRA Fine-tuning | ~0.3M (adapters) | 5-10 minutes | ~87% | ~84% |

## 🔧 Hyperparameter Tuning Tips

### Learning Rate
- **Full fine-tuning**: 1e-5 to 5e-5
- **LoRA**: Can be higher, 1e-4 to 5e-4
- Use learning rate scheduler for better convergence

### Batch Size
- **Small GPU**: 8-16
- **Large GPU**: 32-64
- Gradient accumulation if memory constrained

### LoRA Parameters
- **Rank**: Start with 8, increase to 16 if underfitting
- **Alpha**: Usually 2x the rank
- **Dropout**: 0.1 for regularization

## 🚨 Common Issues and Solutions

### Memory Issues
```python
# Reduce batch size
per_device_train_batch_size=8

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use gradient accumulation
gradient_accumulation_steps=4
```

### Slow Training
```python
# Use smaller subset for debugging
python main.py --step e23 --subset 1000

# Enable mixed precision
fp16=True  # in TrainingArguments

# Reduce sequence length
max_length=256  # instead of 512
```

### Poor Performance
- Check learning rate (might be too high/low)
- Verify data preprocessing
- Ensure balanced evaluation
- Try different random seeds

## 📈 Advanced Extensions

### Different Datasets
```python
# Try other sentiment datasets
dataset = load_dataset("stanfordnlp/sst2")  # Stanford Sentiment
dataset = load_dataset("imdb")              # IMDB Reviews
```

### Other Models
```python
# Try different transformer architectures
model = "roberta-base"
model = "bert-base-uncased"
model = "albert-base-v2"
```

### Multi-class Classification
```python
# Modify for multi-class problems
num_labels=5  # for 5-star rating prediction
```

## 🧪 Experimental Ideas

1. **Ensemble Methods**: Combine SVM and transformer predictions
2. **Data Augmentation**: Use back-translation or paraphrasing
3. **Domain Adaptation**: Fine-tune on movie reviews, test on product reviews
4. **Prompt Learning**: Use prompt-based approaches
5. **Knowledge Distillation**: Use larger model to teach smaller one

## 📝 Lab Report Structure

### 1. Introduction
- Explain transformer fine-tuning motivation
- Describe the sentiment analysis task

### 2. Dataset Analysis
- Present dataset statistics and exploration findings
- Show sample data and label distribution

### 3. Methodology
- Detail each approach (SVM, full fine-tuning, LoRA)
- Explain hyperparameter choices
- Describe evaluation metrics

### 4. Results
- Present performance comparison table
- Include training curves if using wandb
- Analyze computational efficiency

### 5. Discussion
- Compare different approaches
- Discuss when to use each method
- Analyze failure cases

### 6. Conclusion
- Summarize key findings
- Suggest future improvements

## Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PEFT Library Guide](https://huggingface.co/docs/peft/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

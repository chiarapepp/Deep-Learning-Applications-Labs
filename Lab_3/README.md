# Laboratory 3 - Working with Transformers in the HuggingFace Ecosystem

## Overview

This laboratory explores the HuggingFace ecosystem for adapting pre-trained transformer models to downstream tasks. The labo mainly explores sentiment analysis on the *"rotten_tomatoes"* dataset using **DistilBERT** as the backbone model.


The main objectives are:

- Exploration of the dataset and probing of pre-trained models.
- Construction of a stable baseline using a pre-trained DistilBERT model.
- Tokenization of the dataset and fine-tuning of the model with HuggingFace `Trainer`.
- Application of parameter-efficient fine-tuning using LoRA.

### Project Structure
```
Lab_3/
├── main.py                    # Main entry point with argument parsing
├── exercise1.py               # Dataset exploration and SVM baseline
├── exercise2.py               # Full model fine-tuning with Trainer
├── exercise3.py               # Parameter-efficient fine-tuning with LoRA
├── utils.py                   # Configuration and utility functions
├── run_experiments.sh         # Comprehensive experiment runner script
└── README.md                  # This file
```

### Requirements
All core dependencies are already listed in the main repository’s `requirements.txt`.

Alternatively, it's possible to install them manually: 
```bash 
pip install torch transformers datasets scikit-learn numpy tqdm wandb peft
```
(Optional but recommended) Log in to Weights & Biases:
```bash
wandb login
```
## Running Experiments

All experiments are managed through a single script `main.py.` The `--step` argument selects the exercise to run.

```bash
# Exercise 1
# Inspect dataset and explore pre-trained model (Exercise 1.1 and 1.2)
python main.py --step e11
python main.py --step e12

# Run SVM baseline (Exercise 1.3) (optional wandb log)
python main.py --step e13 

# Tokenize dataset for fine-tuning (Exercise 2.1)
python main.py --step e21 

# Fine-tune DistilBERT with Trainer (Exercise 2.3)
python main.py --step e23 $ARGS_finetuning

# Fine-tune DistilBERT using LoRA (Exercise 3.1)
python main.py --step e31 $ARGS_finetuning_LORA
```
### Arguments 

1. Exercise 1.2: Model and tokenizer exploration
    - `--sample_text`: Optional input text(s) for testing the tokenizer.

2. Finetuning hyperparameters
    - `--lr`: Learning rate for fine-tuning (default: `2e-5`).
    - `--epochs`: Number of training epochs (default: `5`).
    - `--batch_size`: Training batch size (default: `16`).
    - `--use_fixed_padding`: Use fixed padding to `max_length=512`, if it's not passed, by default, padding is dynamic.

3. LoRA parameters
    - `--lora_alpha`: LoRA alpha parameter (default: `32`).
    - `--lora_rank`: LoRA rank parameter (default: `8`).
    - `--target_modules`: Target modules to apply LoRA. Examples: `q_lin, k_lin, v_lin, out_lin` / `q_lin, k_lin, v_lin, out_lin lin1 lin2`.

4. Output and logging
    - `--output_dir`: Directory for saving models and logs.
    - `--use_wandb`: Enable Weights & Biases logging.
    - `--run_name`: Name of the WandB run (optional, there are already automated run_name but it's possible to use a custom one).

### Finetuning Experiment Suite
It's possible to run all the main finetuning experiments with the provided script:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

## Experiments and Results

### Exercise 1: Dataset and Model Exploration
In this exercise, we explored the Rotten Tomatoes dataset and the pre-trained DistilBERT model. 
Key observations:
1. **Dataset structure and splits**:
    - The dataset contains **5,331** positive and **5,331** negative sentences (label distribution is balanced across splits).
    - Standard splits (train (8530 samples), validation (1066), test (1066)) were available.

2. **Tokenizer and sample exploration**:
    - Sample sentences were used to understand how the tokenizer works. Tokenizer correctly splits text into subword tokens and handles padding/truncation automatically.
    - The tokenizer converts text into subword tokens and adds special tokens like `[CLS]` (ID `101`) at the start of the sentence and `[SEP]` (ID `102`) at the end. 
    - The [CLS] token embedding represents the full sentence and can be used for downstream tasks.
    - Padding and truncation are used to handle different text lengths, ensuring a consistent size for batch processing.
    - DistilBERT outputs hidden states of size `768` for each token.

3. **SVM baseline**:
    - The pre trained `DistilBERT` model was used as a feature extractor. 
    - `CLS` token embeddings were extracted from the training and validation sets.
    - A **Linear SVM classifier** trained on these features provides a simple but stable baseline for sentiment classification.
    - Metrics (accuracy, precision, recall, F1) give an initial reference point before fine-tuning the transformer.

    The results were  
Validation Results:
  Accuracy:  0.8180
  Precision: 0.8317
  Recall:    0.7974
  F1 Score:  0.8142

Test Results:
  Accuracy:  0.7946
  Precision: 0.8054
  Recall:    0.7767
  F1 Score:  0.7908

### Exercise 2: Tokenization, Model Setup, and Fine-tuning
This exercise, located in `exercise2.py`, prepared the Rotten Tomatoes dataset for fine-tuning a DistilBERT model for binary sentiment classification.

**Dataset Tokenization**:
In `exercise2.py` is defined a function called `tokenize_dataset` that returns a Hugging Face `DatasetDict` with tokenized splits (e.g., train, validation, test). Each split contains the original text and label plus: 
    - `input_ids` → numerical token IDs 
    - `attention_mask` → indicating which tokens are real and which are padding.

In particular it works by using a function (tokenize_function) that takes a batch of examples and applies the pretrained tokenizer (tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')) with arguments the batch, truncation = True (to fit the model), and optional padding definition , and max_length =512 (classico per distilbert ??)

For the padding possibilieties it's possible to pick fixed padding ("max_length") or no padding at tokenization time (False). 
padding (bool, str or PaddingStrategy, optional, defaults to False) — Activates and controls padding. Accepts the following values:
True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).
'max_length': Pad to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.
False or 'do_not_pad' (default): No padding (i.e., can output a batch with sequences of different lengths).


Here’s what each option means and what happens:

- padding=False (default in this exercise when use_fixed_padding is not called): What it does: Do not add any padding during tokenization. Examples keep their natural lengths (after truncation to max_length if needed). 

How batches are padded then: At training time, DataCollatorWithPadding pads dynamically per batch to the longest example in that batch.

This is particularly good because of 
- Efficiency: Less wasted computation and memory because short sequences aren’t padded up to a global maximum.
- Trade-offs: Batch shapes vary at runtime. This is generally fine (and typical) on CPU/GPU training.

- adding="max_length" (when use_fixed_padding=True) : Always pad each example to max_length=512 tokens (and also truncate any longer example to 512). 

I decided to do for every one of my experiments both padding option and what I found was that 
dynamic paddis is \sim 7.5x more fast a parità of the other parameters. 

In particular for example for the case of DistilBErt finetuning (--batch_size 16 --epochs 5 --lr 2e-5): 
Dynamic Padding-> Train runtime: 103.6s (~412 samples/s, 25.8 steps/s)
Fixed Padding- > Train runtime: 781.5s (~54.6 samples/s, 3.4 steps/s)

And for DistilBErt finetuning with Lora (--batch_size 16 --epochs 5 --lr 2e-5 --lora_rank 8 --lora alpha 32): 
Dynamic Padding:58.8s, ~726 samples/s, 45.4 steps/s
Fixed Padding:709.2s, ~60 samples/s, 3.8 steps/s
Dynamic padding rimane ~12x più veloce.

![DistilBERT comparison padding fixed vs dynamics](images/d_p_lora.png.png)
![DistilBERT+ Lora comparison padding fixed vs dynamics](images/dynamic_fixed_padding.png.png)
Speed & memory: Dynamic batch padding is typically faster and lighter.

Shape uniformity: Fixed padding yields constant shapes at the cost of extra compute/memory.

Caching: Padding done at tokenization time is baked into the saved dataset; dynamic padding is applied on the fly per training batch.
### Fine-tuning Distilbert
For the fine tuning of the DistilBERT model for binary sequence classification I used Hugging Face's `Trainer` API. 
It supports configurable learning rate, number of epochs, batch size, asnd padding strategy (dynamic or fixed). 
I decided to do the experiments with 5 epoch, since BERT models (Transformer models) usually convergono veloce nelle prime epoche. La batch size è stata messa a 16 standard.
Infine ho svolto un confronto tra diversi learning rate.

- With a learning rate of 2e-5, the model achieved a lower training loss (~0.19), indicating stable and accurate learning.
- With a learning rate of 2e-4, the loss remained higher (~0.22), suggesting that a too high learning rate causes oscillations and reduces the model's generalization capability.
- No significant differences were observed between dynamic padding and fixed padding to 512 tokens: performance remained comparable, but fixed padding led to longer runtimes (more tokens processed on average).


### Exercise 3: Efficient Fine-tuning with LoR
Per quanto riguarda l'esercizio 3, ho deciso di utilizzare LoRa e ho deciso di utilizzare come parametri Lora rank e alpha loro e loro. 
Poi ho deciso di rendere possibile scegleire quali moduli applicare LoRa , e ho fatto vari esperimenti, il variare del padding, il variare dei parametri, i valori del learning rate e i moduli scelti. 

Le osservazioni sono : 
verall, LoRA models achieved competitive results compared to full fine-tuning while reducing computational costs.
- With a learning rate of 2e-5, ranks 8/α=32 and 16/α=64, the training loss stayed low (~0.22–0.25) and test metrics were good.
- With a learning rate of 2e-4, performance degraded: the loss remained higher and test/validation metrics dropped, as also observed in full fine-tuning.
- Extending target modules (q_lin, k_lin, v_lin, out_lin, lin1, lin2) did not provide substantial advantages over only tuning attention layers. This suggests that localized adaptation on attention blocks is sufficient to capture the necessary task information.


### Results Comparison

| Method | Parameters Trained | Training Time | Validation Acc | Test Acc |
|--------|-------------------|---------------|----------------|----------|
| SVM Baseline | ~1M (SVM only) | 2-5 minutes | ~82% | ~79% |
| Full Fine-tuning | ~67M (all) | 15-30 minutes | ~88% | ~85% |
| LoRA Fine-tuning | ~0.3M (adapters) | 5-10 minutes | ~87% | ~84% |



### Summary of Trends
- A learning rate of 2e-5 is optimal for both full fine-tuning and LoRA.
- Using fixed padding increases runtime without improving performance.
- LoRA maintains comparable performance to full fine-tuning while reducing computational cost, confirming its effectiveness for efficient Transformer adaptation.


## Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [](https://huggingface.co/tasks/feature-extraction)
- [](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)
- [PEFT Library Guide](https://huggingface.co/docs/peft/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
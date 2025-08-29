import argparse
import torch
from utils import config
from exercise1 import load_and_explore_dataset, explore_model_and_tokenizer, run_svm_baseline
from exercise2 import tokenize_dataset, fine_tune_model
from exercise3 import fine_tune_with_lora

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Lab 3: Transformer Fine-tuning.")
    
    # Exercise selection
    parser.add_argument(
        "--step", 
        required=True,
        choices=["e11", "e12", "e13", "e21", "e23", "e31"],
        help="Which exercise to run"
    )
    
    # Dataset options
    parser.add_argument("--subset", type=int, help="Use subset of data for testing")
    
    parser.add_argument("--sample_text", type=str, nargs="+", default=None,
                        help="Optional sample text(s) for exploration")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--use_fixed_padding", action="store_true", help="Use fixed padding to max_length (512), without is dynamic padding")

    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--target_modules", type=str, nargs="+", default=None, help="Target modules for LoRA")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="runs/lab3_experiment")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--run_name", type=str, default=None, help="Name of the WandB run")

    return parser.parse_args()

def main():
    """Main function to run selected exercise"""
    args = get_args()

    torch.manual_seed(10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(10)  
    
    print("Lab 3 : Working with Transformers in the HuggingFace Ecosystem")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    
    if args.step == "e11":
        # Exercise 1.1: Dataset exploration
        load_and_explore_dataset(args.subset)
        
    elif args.step == "e12":
        # Exercise 1.2: Model and tokenizer exploration
        explore_model_and_tokenizer(sample_texts=args.sample_text, use_dataset=args.sample_text is None)
        
    elif args.step == "e13":
        # Exercise 1.3: SVM baseline
        run_svm_baseline(args.use_wandb)
        
    elif args.step == "e21":
        # Exercise 2.1: Tokenize dataset
        tokenize_dataset(use_fixed_padding=args.use_fixed_padding)

    elif args.step == "e23":
        # Exercise 2.3: Fine-tune with Trainer
        fine_tune_model(
            lr=args.lr,
            epochs=args.epochs, 
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            use_wandb=args.use_wandb,
            run_name=args.run_name,
            use_fixed_padding=args.use_fixed_padding  
        )
        
    elif args.step == "e31":
        # Exercise 3.1: LoRA fine-tuning
        fine_tune_with_lora(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            use_wandb=args.use_wandb,
            run_name=args.run_name,
            target_modules=args.target_modules,
            use_fixed_padding=args.use_fixed_padding  # Use padding = max_length (512)
        )
    

if __name__ == "__main__":
    main()
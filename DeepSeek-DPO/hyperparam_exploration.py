#!/usr/bin/env python
# Hyperparameter exploration script for DPO Fine-tuning using Accelerate

import os
import json
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOTrainer, DPOConfig
import wandb
from accelerate import Accelerator

# Configuration grid for hyperparameter search
HYPERPARAMS = {
    "learning_rate": [1e-7, 3e-7, 5e-7, 1e-6],
    "beta": [0.05, 0.1, 0.2]
}

# Fixed parameters
FIXED_PARAMS = {
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "bf16": True,
    "tf32": True,
    "max_length": 8196,
    "max_prompt_length": 512,
    "max_completion_length": 8196,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "optim": "adamw_torch",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0
}

# Model and dataset paths
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATASET_PATH = "samitizerxu/aimo-dpo-ds"
BASE_OUTPUT_DIR = "./deepseek-r1-dpo-experiments"

def prepare_dataset(dataset_path, tokenizer, subset_size=2000, num_workers=32):
    """Prepare and return dataset splits, using a subset for quicker exploration.
    Includes multiprocessing for faster tokenization."""
    dataset = load_dataset(dataset_path)
    
    if "train" in dataset and "validation" in dataset:
        train_dataset = dataset["train"].select(range(min(subset_size, len(dataset["train"]))))
        eval_dataset = dataset["validation"].select(range(min(int(subset_size * 0.2), len(dataset["validation"]))))
    else:
        dataset = dataset["train"].train_test_split(test_size=0.1)
        train_dataset = dataset["train"].select(range(min(subset_size, len(dataset["train"]))))
        eval_dataset = dataset["test"].select(range(min(int(subset_size * 0.2), len(dataset["test"]))))
    
    return train_dataset, eval_dataset

def load_model(model_path, tokenizer):
    """Load model, resizing embeddings to match tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    
    # Resize model embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    return model

def run_experiment(accelerator, hyperparam_config, experiment_id, train_dataset, eval_dataset, tokenizer):
    """Run a single experiment with the specified hyperparameters."""
    # Create output directory for this experiment
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"exp_{experiment_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb for this experiment - only on main process
    run_name = f"lr_{hyperparam_config['learning_rate']}_beta_{hyperparam_config['beta']}"
    if accelerator.is_main_process:
        wandb.init(
            project="deepseek-r1-dpo-hp-search",
            name=run_name,
            config=hyperparam_config,
            group="hp_search",
            reinit=True
        )
    
    # Load fresh model and tokenizer for each experiment
    model = load_model(MODEL_PATH, tokenizer)
    # Configure DPO training
    dpo_config = DPOConfig(
        **FIXED_PARAMS,
        learning_rate=hyperparam_config["learning_rate"],
        beta=hyperparam_config["beta"],
        dataset_num_proc=32,
        # num_train_epochs=None,  # We'll use max_steps instead
        max_steps=200,  # Reduced steps for exploration
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        eval_strategy="steps",
        output_dir=output_dir,
        report_to=["wandb"] if accelerator.is_main_process else [],
        save_total_limit=1,  # Save only the best checkpoint
    )
    
    # Initialize DPO trainer with Accelerate
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # TRL expects tokenizer here for data_collator
    )
    
    # Add a callback to track tag usage
    class GenerationCallback:
        def __init__(self, trainer, tokenizer, interval=50):
            self.trainer = trainer
            self.tokenizer = tokenizer
            self.interval = interval
            self.step = 0
            self.test_prompts = [
                "Solve this math problem step by step: If f(x) = 2x^2 + 3x - 5, find f'(x).",
                "Explain the concept of blockchain to a 10-year-old."
            ]
        
        def on_step_end(self, args, state, control, **kwargs):
            self.step += 1
            # Only run on main process
            if not accelerator.is_main_process:
                return
                
            if self.step % self.interval == 0:
                outputs = []
                for prompt in self.test_prompts:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.trainer.model.device)
                    with torch.no_grad():
                        output = self.trainer.model.generate(
                            **inputs,
                            max_length=1024,
                            temperature=0.7,
                            do_sample=True
                        )
                    text = self.tokenizer.decode(output[0], skip_special_tokens=False)
                    outputs.append(text)
                    
                    # Count keep tags
                    keep_tags = text.count("<keep>")
                    keep_close_tags = text.count("</keep>")
                    
                    wandb.log({
                        "sample_generation": wandb.Table(
                            columns=["prompt", "generation", "keep_tags", "close_tags"],
                            data=[[prompt, text, keep_tags, keep_close_tags]]
                        ),
                        "training_step": self.step
                    })
    
    # Add callback if main process
    if accelerator.is_main_process:
        trainer.add_callback(GenerationCallback(trainer, tokenizer))
    
    # Train
    trainer.train()
    
    # Get evaluation results
    metrics = trainer.evaluate()
    
    # Log final metrics to wandb - only on main process
    if accelerator.is_main_process:
        wandb.log(metrics)
        # Close wandb run
        wandb.finish()
    
    return metrics if accelerator.is_main_process else None

def main():
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Create base output directory
    if accelerator.is_main_process:
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        print(f"Device count: {accelerator.num_processes}")
        print(f"Distributed type: {accelerator.distributed_type}")
    
    # Load model and tokenizer
    if accelerator.is_main_process:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Add special tokens for keep tags if not already present
    special_tokens = ['<keep>', '</keep>']
    if not all(token in tokenizer.get_vocab() for token in special_tokens):
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
    
    # Prepare dataset with multiprocessing tokenization - only on main process
    if accelerator.is_main_process:
        print("Loading and preparing dataset with multiprocessing...")
        train_dataset, eval_dataset = prepare_dataset(DATASET_PATH, tokenizer, num_workers=8)
        # Save processed datasets
        train_dataset.save_to_disk(os.path.join(BASE_OUTPUT_DIR, "hp_processed_train"))
        eval_dataset.save_to_disk(os.path.join(BASE_OUTPUT_DIR, "hp_processed_eval"))
    
    # Wait for main process to finish preprocessing
    accelerator.wait_for_everyone()
    
    # All processes load the prepared datasets
    train_dataset = Dataset.load_from_disk(os.path.join(BASE_OUTPUT_DIR, "hp_processed_train"))
    eval_dataset = Dataset.load_from_disk(os.path.join(BASE_OUTPUT_DIR, "hp_processed_eval"))
    
    if accelerator.is_main_process:
        print(f"Prepared dataset with {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples")
    
    # Generate all hyperparameter combinations
    hp_configs = []
    for lr in HYPERPARAMS["learning_rate"]:
        for beta in HYPERPARAMS["beta"]:
            hp_configs.append({
                "learning_rate": lr,
                "beta": beta
            })
    
    if accelerator.is_main_process:
        print(f"Running {len(hp_configs)} experiments with different hyperparameter configurations")
    
    # Run each experiment
    results = []
    for i, config in enumerate(hp_configs):
        if accelerator.is_main_process:
            print(f"\n--- Starting experiment {i+1}/{len(hp_configs)} ---")
            print(f"Config: {config}")
        
        # Run experiment
        metrics = run_experiment(accelerator, config, i, train_dataset, eval_dataset, tokenizer)
        
        # Record results on main process
        if accelerator.is_main_process and metrics:
            results.append({
                "config": config,
                "metrics": metrics
            })
        
        # Synchronize before next experiment
        accelerator.wait_for_everyone()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Print and save summary of results on main process
    if accelerator.is_main_process:
        print("\n--- Experiment Results Summary ---")
        for result in results:
            config = result["config"]
            metrics = result["metrics"]
            print(f"LR: {config['learning_rate']}, Beta: {config['beta']}")
            print(f"Eval Loss: {metrics.get('eval_loss', 'N/A')}")
            print("---")
        
        # Find best configuration based on eval loss
        best_result = min(results, key=lambda x: x["metrics"].get("eval_loss", float("inf")))
        best_config = best_result["config"]
        best_metrics = best_result["metrics"]
        
        print("\n--- Best Configuration ---")
        print(f"Learning Rate: {best_config['learning_rate']}")
        print(f"Beta: {best_config['beta']}")
        print(f"Eval Loss: {best_metrics.get('eval_loss', 'N/A')}")
        
        # Save best config for full training
        with open(os.path.join(BASE_OUTPUT_DIR, "best_config.json"), "w") as f:
            json.dump({
                "config": best_config,
                "metrics": best_metrics
            }, f, indent=2)
        
        print(f"\nBest configuration saved to {os.path.join(BASE_OUTPUT_DIR, 'best_config.json')}")
        print("You can now run a full training run with these hyperparameters")

if __name__ == "__main__":
    main()
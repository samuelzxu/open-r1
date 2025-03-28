# Full DPO training script using best hyperparameters

import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
import wandb

# Model and dataset paths
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATASET_PATH = "samitizerxu/aimo-dpo-ds"
EXPERIMENTS_DIR = "../outputs/dpo-experiments"
OUTPUT_DIR = "../outputs/dpo-final"

# Fixed parameters
FIXED_PARAMS = {
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "gradient_checkpointing": True,
    "bf16": True,
    "tf32": True,
    "max_length": 8196,
    "max_prompt_length": 512,
    "max_target_length": 8196,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "optim": "adamw_torch",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0
}

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load best hyperparameters from exploration
    try:
        with open(os.path.join(EXPERIMENTS_DIR, "best_config.json"), "r") as f:
            best_config = json.load(f)["config"]
        print(f"Loaded best hyperparameters: {best_config}")
    except FileNotFoundError:
        print("Best config file not found. Using default hyperparameters.")
        best_config = {
            "learning_rate": 5e-7,
            "beta": 0.1
        }
    
    # Initialize wandb
    wandb.init(
        project="deepseek-r1-dpo-final",
        name=f"final_training_lr_{best_config['learning_rate']}_beta_{best_config['beta']}",
        config={**FIXED_PARAMS, **best_config}
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    
    if "train" in dataset and "validation" in dataset:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    else:
        dataset = dataset["train"].train_test_split(test_size=0.05)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    
    print(f"Dataset loaded with {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Add special tokens if needed
    special_tokens = ['<keep>', '</keep>']
    if not all(token in tokenizer.get_vocab() for token in special_tokens):
        print("Adding special tokens to tokenizer...")
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    
    # Configure DPO training
    dpo_config = DPOConfig(
        **FIXED_PARAMS,
        learning_rate=best_config["learning_rate"],
        beta=best_config["beta"],
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        output_dir=OUTPUT_DIR,
        report_to=["wandb"],
        deepspeed="../configs/ds_config.json",
        save_total_limit=3,
    )
    
    # Add a callback to log samples during training
    class GenerationCallback:
        def __init__(self, trainer, tokenizer, interval=500):
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
    
    # Initialize DPO trainer
    print("Initializing trainer...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train model
    print("Starting training...")
    trainer.add_callback(GenerationCallback(trainer, tokenizer))
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    
    print("Training complete!")

if __name__ == "__main__":
    main()
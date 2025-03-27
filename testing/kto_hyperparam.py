# train_kto.py
from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


if __name__ == "__main__":
    # parse lr and beta from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", padding_side='left')
    train_dataset = load_dataset("samitizerxu/aimo-kto-ds", split="train")
    eval_dataset = load_dataset("samitizerxu/aimo-kto-ds", split="test")

    special_tokens = ['<keep>', '</keep>']
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    training_args = KTOConfig(
        output_dir="DeepSeek-R1-Distill-Qwen-7B-KTO",
        logging_steps=10,
        gradient_accumulation_steps=8,
        bf16=True,
        max_length=8196,
        max_prompt_length=512,
        max_completion_length=8196,
        dataset_num_proc=150,
        per_device_train_batch_size=2,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=2,
        gradient_checkpointing=True,
        eval_strategy='steps',
        eval_steps=50,
        max_steps=250,
        # save_strategy='steps',
        # save_steps=400,
        # save_total_limit=1,
        # hub_strategy='every_save',
        # num_train_epochs=2,
        # eval_steps=400
        run_name=f"lr_{args.lr}_beta_{args.beta}",
        
        learning_rate=args.lr,
        beta=args.beta,
        # push_to_hub=True,
        # hub_model_id='samitizerxu/DS-distil-keep',
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
    )
    trainer = KTOTrainer(
        model=model, 
        args=training_args, 
        processing_class=tokenizer, 
        eval_dataset=eval_dataset, 
        train_dataset=train_dataset
    )
    trainer.train()
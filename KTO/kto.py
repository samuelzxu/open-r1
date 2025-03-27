# train_kto.py
from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


if __name__ == "__main__":
    lr = 5e-6
    beta = 0.01

    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", padding_side='left')
    train_dataset = load_dataset("samitizerxu/aimo-kto-ds", split="train")
    eval_dataset = load_dataset("samitizerxu/aimo-kto-ds", split="test")

    special_tokens = ['<keep>', '</keep>']
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    training_args = KTOConfig(
        push_to_hub=True,
        hub_model_id='samitizerxu/DS-7B-Qwen-distil-KTO-keep',
        output_dir="DeepSeek-R1-Distill-Qwen-7B-KTO-keep",

        logging_steps=5,
        gradient_accumulation_steps=6,
        bf16=True,
        max_length=8196,
        max_prompt_length=512,
        max_completion_length=8196,
        dataset_num_proc=150,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,

        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_checkpointing=True,

        eval_strategy='steps',
        eval_steps=200,
        save_strategy='steps',
        save_steps=200,

        save_total_limit=3,
        hub_strategy='end',
        num_train_epochs=2,
        run_name=f"FINAL-lr_5e-6_beta_0.01",
        
        learning_rate=lr,
        beta=beta,
        
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
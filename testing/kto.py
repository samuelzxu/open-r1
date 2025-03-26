# train_kto.py
from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
train_dataset = load_dataset("samitizerxu/aimo-kto-ds", split="train")

training_args = KTOConfig(output_dir="DeepSeek-R1-Distill-Qwen-7B-KTO", logging_steps=10, gradient_accumulation_steps=8, bf16=True, max_length=8196, max_prompt_length=512, max_completion_length=8196, dataset_num_proc=150, output_dir="DeepSeek-R1-Distill-Qwen-7B-KTO")
trainer = KTOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
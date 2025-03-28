from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import re

# model_dir = "DS-7B-Qwen-distil-KTO-keep-awq"
model_dir = 'samitizerxu/DS-7B-Qwen-distil-KTO-keep'

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="cuda")

system_prompt = "You are a helpful AI Assistant who provides well-reasoned and accurate responses to math problems. You should think step-by-step, and the answer is an integer. You should wrap important steps in <keep> .. </keep> tags. Return the final answer within \\boxed{} after taking modulo 1000."

user_prompt = "Let $N$ denote the number of ordered triples of positive integers $(a, b, c)$ such that $a, b, c \leq 3^6$ and $a^3 + b^3 + c^3$ is a multiple of $3^7$. Find the remainder when $N$ is divided by $1000$."

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    tokenize=False,
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating...")
# Force the model to use the keep tags
outputs = model.generate(**inputs, max_new_tokens=3000)

def extract_keep_text(ex_keep_output) -> list[str]:
    return re.findall(r': <keep>(.*?)</keep>', ex_keep_output)

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)

print("*" * 100)

print(extract_keep_text(response))
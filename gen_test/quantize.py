from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'samitizerxu/DS-7B-Qwen-distil-DPO-keep'
quant_path = 'DS-7B-Qwen-distil-DPO-keep-awq'
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_data():
    data = load_dataset('samitizerxu/aimo-dpo-ds-v2', split="train")
    data = data.filter(lambda x: len(x["prompt"])+len(x["chosen"]) >= 3000)
    all_rows = [row["prompt"]+row["chosen"] for row in data]
    print(len(all_rows))
    print(all_rows[0])
    return all_rows

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=load_data(),
    # n_parallel_calib_samples=32,
    max_calib_samples=128,
    max_calib_seq_len=1024
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')


from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import re
import argparse
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_dir", type=str, default="samitizerxu/DS-7B-Qwen-distil-DPO-keep")
    args = args.parse_args()

    model_dir = args.model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="cuda")

    system_prompt = "You are a helpful AI Assistant who provides well-reasoned and accurate responses to math problems. You should think step-by-step, and the answer is an integer. You should wrap important steps in <keep> .. </keep> tags. Return the final answer within \\boxed{} after taking modulo 1000."

    problems = [
        "Let $N$ denote the number of ordered triples of positive integers $(a, b, c)$ such that $a, b, c \leq 3^6$ and $a^3 + b^3 + c^3$ is a multiple of $3^7$. Find the remainder when $N$ is divided by $1000$.",
        "Let $\triangle ABC$ be an equilateral triangle with side length $55.$ Points $D,$ $E,$ and $F$ lie on $\overline{BC},$ $\overline{CA},$ and $\overline{AB},$ respectively, with $BD = 7,$ $CE=30,$ and $AF=40.$ Point $P$ inside $\triangle ABC$ has the property that\[\angle AEP = \angle BFP = \angle CDP.\]Find $\tan^2(\angle AEP).$",
        "Let $S$ be the set of all rational numbers that can be expressed as a repeating decimal in the form $0.\overline{abcd},$ where at least one of the digits $a,$ $b,$ $c,$ or $d$ is nonzero. Let $N$ be the number of distinct numerators obtained when numbers in $S$ are written as fractions in lowest terms. For example, both $4$ and $410$ are counted among the distinct numerators for numbers in $S$ because $0.\overline{3636} = \frac{4}{11}$ and $0.\overline{1230} = \frac{410}{3333}.$ Find the remainder when $N$ is divided by $1000.$",
        "Let $ABCDE$ be a convex pentagon with $AB=14,$ $BC=7,$ $CD=24,$ $DE=13,$ $EA=26,$ and $\angle B=\angle E=60^{\circ}.$ For each point $X$ in the plane, define $f(X)=AX+BX+CX+DX+EX.$ The least possible value of $f(X)$ can be expressed as $m+n\sqrt{p},$ where $m$ and $n$ are positive integers and $p$ is not divisible by the square of any prime. Find $m+n+p.$"
    ]


        
    prompts = [tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ],
        tokenize=False,
        add_generation_prompt=False,
    ) for problem in problems]

    inputs = [tokenizer(prompt, return_tensors="pt").to("cuda") for prompt in prompts]

    print(f"Generating for {model_dir} .................")
    # Force the model to use the keep tags

    def extract_keep_text(ex_keep_output) -> list[str]:
        return re.findall(r': <keep>(.*?)</keep>', ex_keep_output)

    for input in inputs:
        outputs = model.generate(**input, temperature=0.6, max_new_tokens=8100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(response[:4000])
        print("Keep text: ", "*" * 100)
        print(extract_keep_text(response))
        print("*" * 100)

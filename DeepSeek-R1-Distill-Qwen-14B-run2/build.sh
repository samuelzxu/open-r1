pip install uv
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip --link-mode=copy
# uv pip install vllm==0.7.1 --link-mode=copy
uv pip install --upgrade torch trl datasets setuptools --link-mode=copy
uv pip install flash-attn --no-build-isolation --link-mode=copy
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]" --link-mode=copy --no-build-isolation
uv pip install wandb
wandb login
huggingface-cli login
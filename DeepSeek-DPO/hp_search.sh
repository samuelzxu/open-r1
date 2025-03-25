#!/bin/bash
# Launch script for hyperparameter exploration on 8xH100 GPUs

# Activate virtual environment

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY="872751e460617ab35b0f617a6b3339271799c94e" # Replace with your key


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=8 DeepSeek-DPO/hyperparam_exploration.py \
    --wandb_entity samitizerxu --wandb_project aimo-training 
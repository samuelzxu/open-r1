ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config DeepSeek-R1-Distill-Qwen-14B/grpo_run1.yaml --wandb_entity samitizerxu --wandb_project aimo-training 
    
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config DeepSeek-R1-Distill-Qwen-14B/grpo_run2.yaml --wandb_entity samitizerxu --wandb_project aimo-training
    
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config DeepSeek-R1-Distill-Qwen-14B/grpo_run3.yaml --wandb_entity samitizerxu --wandb_project aimo-training
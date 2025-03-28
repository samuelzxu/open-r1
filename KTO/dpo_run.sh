export WANDB_PROJECT=aimoDPO
accelerate launch --config_file KTO/zero2.yaml KTO/dpo.py
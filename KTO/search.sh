#!/bin/bash

# runs the kto training scripts w/ hyperparam exploration
export WANDB_PROJECT=aimoKTO
for lr in 5e-6 1e-6 7e-7; do
    for beta in 0.1 0.05 0.01; do
        accelerate launch --config_file KTO/zero2.yaml KTO/kto_hyperparam.py --lr $lr --beta $beta
    done
done
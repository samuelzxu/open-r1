#!/bin/bash

# runs the kto training scripts w/ hyperparam exploration

for lr in 5e-6 1e-6 7e-7; do
    for beta in 0.1 0.05 0.01; do
        accelerate launch --config_file testing/zero2.yaml testing/kto.py --lr $lr --beta $beta
    done
done
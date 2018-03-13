#!/usr/bin/env bash

for hidden_layer in 5 10 20; do
    for layers in 1 2 3 5; do
        fname="train-hl${hidden_layer}-l${layers}.txt"
        python gym_cartpole.py --hidden_layer=$hidden_layer --layers=$layers --threads=3 > "runs/$fname"
        python gym_cartpole_evaluate.py | tee "runs/res-$fname" | tail
    done
done

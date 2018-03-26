#!/usr/bin/env bash

for hidden_layer_units in 10 20 30 50; do
    for layers in 1 2 3 5; do
        for activation in relu tanh; do
            for learning_rate in 0.01 0.001; do
                python uppercase.py --hidden_layer_units=${hidden_layer_units} --layers=${layers} --activation=${activation} --learning_rate=${learning_rate} --threads=8 --test
            done
        done
    done
done

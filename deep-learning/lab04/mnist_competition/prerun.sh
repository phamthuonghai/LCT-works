#!/usr/bin/env bash

for batch_size in 100 500 1000 5000; do
    python mnist_competition.py --batch_size=${batch_size}
done

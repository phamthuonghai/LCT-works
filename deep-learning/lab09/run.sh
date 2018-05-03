#!/usr/bin/env bash#!/usr/bin/env bash
#$ -cwd
#$ -j y
#$ -S /bin/bash

source ../.env/bin/activate

for batch_size in 16 32; do
    for char_dim in 128 256; do
            CUDA_VISIBLE_DEVICES=$1 ../.env/bin/python -uB lemmatizer_sota.py --batch_size=${batch_size} --char_dim=${char_dim} --rnn_dim=${char_dim}
    done
done

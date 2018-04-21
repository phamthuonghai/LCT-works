#!/usr/bin/env bash
#$ -cwd
#$ -j y
#$ -S /bin/bash

source ../../.env/bin/activate

while [ 1 ]
do
    CUDA_VISIBLE_DEVICES=$1 python -uB nsketch_transfer.py
    done=$?
    if [ "$done" -eq 111 ]; then
        break
    fi
done

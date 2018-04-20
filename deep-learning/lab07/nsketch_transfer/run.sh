#!/usr/bin/env bash

while [ 1 ]
do
    CUDA_VISIBLE_DEVICES=$1 python -uB nsketch_transfer.py
    done=$?
    if [ "$done" -eq 111 ]; then
        break
    fi
done

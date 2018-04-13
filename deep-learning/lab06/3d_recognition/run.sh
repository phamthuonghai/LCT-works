#!/usr/bin/env bash

while [ 1 ]
do
    CUDA_VISIBLE_DEVICES=$1 python 3d_recognition.py --params=./params.json
    done=$?
    if [ "$done" -eq 111 ]; then
        break
    fi
done

#!/usr/bin/env bash

while [ 1 ]
do
    python 3d_recognition.py --params=./params.json
    done=$?
    if [ "$done" -ne 0 ]; then
        break
    fi
done

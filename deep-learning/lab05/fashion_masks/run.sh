#!/usr/bin/env bash

while [ 1 ]
do
    python fashion_masks.py --params="$1"
    done=$?
    if [ "$done" -ne 0 ]; then
        break
    fi
done

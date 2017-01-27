#!/bin/bash

export PYTHON="./env/bin/python"
export DATA_DIR=$1
export CORE_IN_FILE_PREFIX=$2
export CORE_SPC=CORE_SS.$2.ppmi.svd_500.pkl

echo "Building core space"
$PYTHON ./dissect/src/pipelines/build_core_space.py -i $DATA_DIR/$CORE_IN_FILE_PREFIX --input_format=sm -o $DATA_DIR -w ppmi -r svd_500

echo "Finding similar words"
$PYTHON ./dissect/src/pipelines/compute_neighbours.py -i $DATA_DIR/list_top5_highest.txt -n 5 -s $DATA_DIR/$CORE_SPC -o $DATA_DIR -m cos
$PYTHON ./dissect/src/pipelines/compute_neighbours.py -i $DATA_DIR/list_top5_lowest.txt -n 5 -s $DATA_DIR/$CORE_SPC -o $DATA_DIR -m cos

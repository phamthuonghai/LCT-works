#!/bin/bash

mkdir logs

cd ./cmake-build-debug/
echo "=========Random========="
time ./heapgen -s 69 -r | ./fibheap > ../logs/cr.csv
echo "=========Biased========="
time ./heapgen -s 69 -b | ./fibheap > ../logs/cb.csv
echo "=========Special========="
time ./heapgen -s 69 -x | ./fibheap > ../logs/cx.csv
time ./heapgen -s 69 -x | ./fibheap -n > ../logs/nx.csv

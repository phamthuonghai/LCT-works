#!/bin/bash

mkdir ./logs
cd ./cmake-build-debug

echo "========= Sequential ========="
# time ./splaygen -s 69 -l -b | ./splaytree > ../logs/classic-l-b.txt
# time ./splaygen -s 69 -l -b | ./splaytree -n > ../logs/naive-l-b.txt

time ./splaygen -s 69 -b | ./splaytree > ../logs/classic-b.txt
time ./splaygen -s 69 -b | ./splaytree -n > ../logs/naive-b.txt

for i in 10 100 1000 10000 100000 1000000; do
    echo "========= T = $i ========="
#    time ./splaygen -s 69 -t $i -l | ./splaytree > ../logs/classic-l-$i.txt
#    time ./splaygen -s 69 -t $i -l | ./splaytree -n > ../logs/naive-l-$i.txt
    time ./splaygen -s 69 -t $i | ./splaytree > ../logs/classic-$i.txt
    time ./splaygen -s 69 -t $i | ./splaytree -n > ../logs/naive-$i.txt
done

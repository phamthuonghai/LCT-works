#!/usr/bin/env bash

mkdir logs

#echo "1st task, linear, tab"
#./cmake-build-debug/hash 0 0 0 > ./logs/0_li_ta.log
#
#echo "1st task, linear, mul-shift"
#./cmake-build-debug/hash 0 0 1 > ./logs/0_li_mu.log
#
#echo "1st task, linear, naive"
#./cmake-build-debug/hash 0 0 2 > ./logs/0_li_na.log
#
#echo "1st task, cuckoo, tab"
#./cmake-build-debug/hash 0 1 0 > ./logs/0_cu_ta.log
#
#echo "1st task, cuckoo, mul-shift"
#./cmake-build-debug/hash 0 0 1 > ./logs/0_cu_mu.log
#
#echo "2nd task, linear, tab"
#./cmake-build-debug/hash 1 0 0 > ./logs/1_li_ta.log1

echo "2nd task, linear, mul-shift"
./cmake-build-debug/hash 1 0 1 > ./logs/1_li_mu.log

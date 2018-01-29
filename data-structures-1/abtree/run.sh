#!/usr/bin/env bash

./cmake-build-debug/abgen -s 69 | ./cmake-build-debug/abtree 2 3 > uni_2_3.log
./cmake-build-debug/abgen -s 69 | ./cmake-build-debug/abtree 2 4 > uni_2_4.log
./cmake-build-debug/abgen -s 69 -b | ./cmake-build-debug/abtree 2 3 > bia_2_3.log
./cmake-build-debug/abgen -s 69 -b | ./cmake-build-debug/abtree 2 4 > bia_2_4.log

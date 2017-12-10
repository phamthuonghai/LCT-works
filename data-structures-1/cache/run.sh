#!/usr/bin/env bash

CACHE_BIN=../cmake-build-debug/cache
CACHSIM_BIN=../cmake-build-debug/cachesim

mkdir logs
cd logs

B=( 64 64 64 512 4096 )
C=( 64 1024 4096 512 64 )
for idx in "${!B[@]}"; do
    b=${B[$idx]}
    c=${C[$idx]}
    ${CACHE_BIN} cache_ts_${b}_${c}.log -t -s | ${CACHSIM_BIN} ${b} ${c} > cachesim_ts_${b}_${c}.log
    ${CACHE_BIN} cache_s_${b}_${c}.log -s | ${CACHSIM_BIN} ${b} ${c} > cachesim_s_${b}_${c}.log
done

for i in 1 2 3 4 5; do
    ${CACHE_BIN} cache_t_${i}.log -t
    ${CACHE_BIN} cache_${i}.log
done

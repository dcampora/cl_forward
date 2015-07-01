#!/bin/bash

for i in `seq 16 64`; do
  sed -i 's/^NVCC :=.*$/NVCC := $\(CUDA_PATH\)\/bin\/nvcc --maxrregcount='${i}' -std=c++11 -ccbin $\(GCC\)/' Makefile
  make clean
  make
  ./tools/run.sh 2048 > results/$i.out
done

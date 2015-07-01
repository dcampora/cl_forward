#!/bin/bash

DATA_FILES=50
RUN_ITERATIONS=10
INPUT=""

if [ $1 ]
then
  RUN_ITERATIONS=$1
fi

for i in `seq 0 $((RUN_ITERATIONS - 1))`
do
  INPUT+="mcdata/$((${i} % $DATA_FILES)).dat"
  if [ $i != $(($RUN_ITERATIONS - 1)) ]
  then
    INPUT+=","
  fi
done

./gpupixel $INPUT

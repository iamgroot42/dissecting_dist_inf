#!/bin/bash

./train_fast.sh adv two_attr 0.5,0.5
./train_fast.sh victim two_attr 0.5,0.5
./train_fast.sh adv two_attr 0.2,0.1
./train_fast.sh victim two_attr 0.2,0.1
./train_fast.sh adv two_attr 0.5,0.1
./train_fast.sh victim two_attr 0.5,0.1

for i in 0.2,0.5 0.5,0.2 0.1,0.5
do
./train_fast.sh adv two_attr $i
./train_fast.sh victim two_attr $i
python perf_tests.py --filter two_attr --ratio_1 0.5,0.5 --ratio_2 $i
done
python perf_tests.py --filter two_attr --ratio_1 0.2,0.1 --ratio_2 0.5,0.1
python meta.py --filter two_attr --d_0 0.5,0.5
python meta.py --filter two_attr --d_0 0.2,0.1 --trg 0.5,0.1
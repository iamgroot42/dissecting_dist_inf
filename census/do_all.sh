#!/bin/bash
./train_fast.sh adv sex 0.5
./train_fast.sh victim sex 0.5
./train_fast.sh adv race 0.5
./train_fast.sh victim race 0.5
./train_fast.sh adv sex 0.2
./train_fast.sh victim sex 0.2
./train_fast.sh adv race 0.1
./train_fast.sh victim race 0.1
python meta.py --filter sex --d_0 0.5 --trg 0.2 --save True
python meta.py --filter race --d_0 0.5 --trg 0.1 --save True
python meta.py --filter two_attr --d_0 0.5,0.5 --trg 0.2,0.1 --save True
#!/bin/bash
./perf_test_all.sh 0.1 0.2
python test_meta.py --filter sex --md_0 0.5,0.5 --mtrg '0.1,0.2' --d_0 0.5 --trg 0.1
python test_meta.py --filter race --md_0 0.5,0.5 --mtrg '0.1,0.2' --d_0 0.5 --trg 0.2
for i in 0.4 0.6 0.8
do
./perf_test_all.sh 0.1 $i
./perf_test_all.sh $i 0.1
python test_meta.py --filter sex --md_0 0.5,0.5 --mtrg '0.1,'$i --d_0 0.5 --trg $i
python test_meta.py --filter sex --md_0 0.5,0.5 --mtrg $i',0.1' --d_0 0.5 --trg $i
python test_meta.py --filter race --md_0 0.5,0.5 --mtrg '0.1,'$i --d_0 0.5 --trg $i
python test_meta.py --filter race --md_0 0.5,0.5 --mtrg $i',0.1' --d_0 0.5 --trg $i
done



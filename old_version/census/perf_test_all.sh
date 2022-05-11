#!/bin/bash
python perf_tests.py --filter sex --ratio_1 0.5 --ratio_2 $1
python perf_tests.py --filter race --ratio_1 0.5 --ratio_2 $2
python perf_tests.py --filter two_attr --ratio_1 0.5,0.5 --ratio_2 $1','$2
python perf_dif.py --filter sex --ratio_1 0.5 --ratio_2 $1 --filter_d two_attr --ratio_1_d 0.5,0.5 --ratio_2_d $1','$2
python perf_dif.py --filter race --ratio_1 0.5 --ratio_2 $2 --filter_d two_attr --ratio_1_d 0.5,0.5 --ratio_2_d $1','$2
python perf_dif.py --filter two_attr --ratio_1 0.5,0.5 --ratio_2 $1','$2 --filter_d sex --ratio_1_d 0.5 --ratio_2_d $1
python perf_dif.py --filter two_attr --ratio_1 0.5,0.5 --ratio_2 $1','$2 --filter_d race --ratio_1_d 0.5 --ratio_2_d $2

for i in 0.0 0.1 0.2 0.3 0.4
do
python perf_try.py --filter race --ratio_1 0.5 --ratio_2 $i
done

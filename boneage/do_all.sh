for i in $1
do
python perf_all.py  --ratio_1 0.5 --ratio_2 $i --gpu $2
done

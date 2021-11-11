for i in 0.7 0.8
do
python perf_quart.py --ratio_1 0.5 --ratio_2 $i
done
python quart_boxplot.py  --ratio 0.5
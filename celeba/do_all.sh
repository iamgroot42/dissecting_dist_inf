for i in $1
do
python perf_quart.py --filter Male --ratio_1 0.5 --ratio_2 $i &
python perf_quart.py --filter Young --task Male --ratio_1 0.5 --ratio_2 $i 
wait
done
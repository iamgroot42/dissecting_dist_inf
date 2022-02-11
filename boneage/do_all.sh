declare -i j=0
for i in $1
do
nohup python perf_gen.py --ratio_2 $i --use_normal --n_models 200 --steps 100 --n_samples 500 --r 0.2 --step_size 10 --gpu $j > $i 2>&1 &
j=$((j+1))
done
wait

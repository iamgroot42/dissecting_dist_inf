
declare -i j=0
p=./log/robust/gen/perpoint_Male\:0.5
for i in $1
do
#mkdir -p $p/$i
nohup python perf_gen.py --filter Male --ratio_2 $i --use_normal --n_models 200 --steps 50 --n_samples 500 --r 0.2 --use_adv_for_adv --use_adv_for_victim --adv_adv_prefix adv_train_8 --victim_adv_prefix adv_train_8 --gpu $j > $p/$i/robust8.out 2>&1 &
j=$((j+1))
done
wait


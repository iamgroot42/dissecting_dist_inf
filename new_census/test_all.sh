for i in 0.0 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0
do
for j in sex race
do
python blackbox.py --filter $j --ratio_1 0.5 --ratio_2 $i &
python blackbox.py --filter $j --ratio_1 0.5 --ratio_2 $i --drop &
python blackbox.py --filter $j --ratio_1 0.5 --ratio_2 $i --scale 0.1 &
python blackbox.py --filter $j --ratio_1 0.5 --ratio_2 $i --scale 3.0 &
done
wait
done

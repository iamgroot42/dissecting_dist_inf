./train_fast.sh adv race 0.5
./train_fast.sh victim race 0.5
for i in 0.0 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0
do
./train_fast.sh adv race $i
./train_fast.sh victim race $i
python perf_quart.py --filter race --ratio_1 0.5 --ratio_2 $i
done
for i in 0.0 0.1 0.2 0.3 0.4
do
python perf_incre.py --filter race --ratio_1 0.5 --ratio_2 $i
done
python quart_boxplot.py --filter race
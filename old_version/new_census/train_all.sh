for i in $1
do
for k in victim adv
do
for j in sex race
do
./train_fast.sh $k $j $i $2 &
done
done
wait
done
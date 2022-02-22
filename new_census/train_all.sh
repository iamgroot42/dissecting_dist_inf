for i in $2
do
for k in victim adv
do
./train_fast.sh $k $1 $i &
done
wait
done
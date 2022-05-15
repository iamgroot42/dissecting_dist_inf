for i in loss threshold perpoint
do
for j in sex race
do 
python blackbox_plot.py --filter $j --attack $i &
done
done
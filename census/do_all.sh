for i in 0.2 0.4 0.6 0.8
do
python dif_boxplot.py --d_0 0.5,0.5 --ratio '0.1,'$i
python dif_boxplot.py --d_0 0.5,0.5 --ratio $i',0.1'
done

./perf_test_all.sh 0.2 0.1
python test_meta.py --filter sex --md_0 0.5,0.5 --mtrg 0.2,0.1 --d_0 0.5 --trg 0.2
python test_meta.py --filter race --md_0 0.5,0.5 --mtrg 0.2,0.1 --d_0 0.5 --trg 0.1
for i in 0.2 0.4 0.6 0.8
do
python dif_boxplot.py --d_0 0.5,0.5 --ratio '0.1,'$i
python dif_boxplot.py --d_0 0.5,0.5 --ratio $i',0.1'
done

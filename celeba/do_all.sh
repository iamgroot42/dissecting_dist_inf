
#mkdir -p ./log/gen/perpoint_Male\:0.5/$i
#nohup python perf_gen.py --filter Male --ratio_2 0.2 --use_normal --n_models 100 --steps 20 --n_samples 500 --r 0.2 --gpu 2 > ./log/gen/perpoint_Male\:0.5/o0.2/500sam_20step_100m_20r.out 2>&1 &
#nohup python perf_gen.py --filter Male --ratio_2 0.2 --use_normal --n_models 100 --steps 20 --n_samples 500 --r2 0.2 --gpu 3 > ./log/gen/perpoint_Male\:0.5/o0.2/500sam_20step_100m_20r2.out 2>&1 &

p=./log/gen/perpoint_Male\:0.5/
#[ ! -d $p ] && mkdir -p $p
nohup python perf_gen.py --filter Male --ratio_2 0.6 --use_normal --n_models 200 --steps 50 --n_samples 500 --r 0.2 --gpu 0 > $p/0.6/500sam_50step_20r.out 2>&1 &
nohup python perf_gen.py --filter Male --ratio_2 0.7 --use_normal --n_models 200 --steps 50 --n_samples 500 --r 0.2 --gpu 1 > $p/0.7/500sam_50step_20r.out 2>&1 &



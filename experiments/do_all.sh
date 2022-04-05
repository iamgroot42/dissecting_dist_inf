python blackbox_attacks.py --load_config "./configs/new_census/dp0.1_both.json" --en "sex_dp0.1_both.json" & 
python blackbox_attacks.py --load_config "./configs/new_census/dp0.1_victim.json" --en "sex_dp0.1_vic.json" &
wait
python blackbox_attacks.py --load_config "./configs/new_census/dp1.0_both.json" --en "sex_dp1.0_both.json" &
python blackbox_attacks.py --load_config "./configs/new_census/dp1.0_victim.json" --en "sex_dp1.0_vic.json" & 
wait
echo "finished"
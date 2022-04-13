#python blackbox_attacks.py --load_config "./dp0.1_victim.json" --en "race_bbdp0.1_vic" & 
python whitebox_attacks.py --load_config "./celeba/wb_y.json" --en "celeba_young_wb" & 
python whitebox_attacks.py --load_config "./celeba/wb_m.json" --en "celeba_male_wb" & 
wait
python whitebox_attacks.py --load_config "./boneage/wb.json" --en "boneage_wb" 
echo "finished"
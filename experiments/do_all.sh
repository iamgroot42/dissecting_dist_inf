#python blackbox_attacks.py --load_config "./dp0.1_victim.json" --en "race_bbdp0.1_vic" & 
#python whitebox_attacks.py --load_config "./configs/celeba/wb_y.json" --en "celeba_young_wb" & 
#python whitebox_attacks.py --load_config "./configs/celeba/wb_m.json" --en "celeba_male_wb" & 
#wait
python whitebox_attacks.py --load_config "./configs/boneage/wb.json" --en "bone_wb" 
echo "finished"
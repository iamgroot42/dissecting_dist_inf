#python blackbox_attacks.py --load_config "./dp0.1_victim.json" --en "race_bbdp0.1_vic" & 
#python whitebox_attacks.py --load_config "./configs/celeba/wb_y.json" --en "celeba_young_wb" & 
#python whitebox_attacks.py --load_config "./configs/celeba/wb_m.json" --en "celeba_male_wb" & 
#wait
#python whitebox_attacks.py --load_config "./configs/boneage/wb.json" --en "bone_wb"
#j=0
for i in $1
do
python train_more.py --load_config "./texas_epoch.json" --ratio $i --split "victim" &
python train_more.py --load_config "./texas_epoch.json" --ratio $i --split "adv" &
wait
#j=$((($j+1)%3))
done
echo "finished"
#python blackbox_attacks.py --load_config shuffle_bb.json --en race_bb_oversample
#python blackbox_attacks.py --load_config "./dp0.1_victim.json" --en "race_bbdp0.1_vic" & 
#python whitebox_attacks.py --load_config "./configs/celeba/wb_y.json" --en "celeba_young_wb" & 
#python whitebox_attacks.py --load_config "./configs/celeba/wb_m.json" --en "celeba_male_wb" & 
#wait
#python whitebox_attacks.py --load_config "./configs/boneage/wb.json" --en "bone_wb"
j=0
for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
python train_more.py --load_config "./train_nc_noise.json" --split victim --ratio $i --gpu $j &
python train_more.py --load_config "./train_nc_noise.json" --split adv --ratio $i --gpu $j &
j=$((($j+1)%4))
if [ $j -ge 3 ]
then
wait
fi
done
wait
echo "finished"
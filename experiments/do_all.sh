#python blackbox_attacks.py --load_config "./dp0.1_victim.json" --en "race_bbdp0.1_vic" & 
#python whitebox_attacks.py --load_config "./configs/celeba/wb_y.json" --en "celeba_young_wb" & 
#python whitebox_attacks.py --load_config "./configs/celeba/wb_m.json" --en "celeba_male_wb" & 
#wait
#python whitebox_attacks.py --load_config "./configs/boneage/wb.json" --en "bone_wb"

#j=0
for i in $1
do
echo $i
python train_more.py --load_config "./cel_mouth.json" --ratios $i --split adv --offset $2 &
#python comparison_save.py --load_config nc_comp.json --en sex_comp --gpu $j --trial_offset $2 --ratios $i &
#python whitebox_affinity.py --load_config race_aff.json --en "Not_saving" --gpu $j --ratios $i --trial $2 & 
#python generate_metric_plot.py --log_path log/new_census/training/victim_race_mlp2_race_0.json --wanted loss R_cross --title "Census 19 race $i" --ratios $i --savepath race_R_vs_loss_$i
#python generate_metric_plot.py --log_path log/new_census/training/victim_race_mlp2_race_0.json --attack_path log/new_census/whitebox/race_epo_aff.json --wanted R_cross --title "Census 19 race $i" --ratios $i --savepath race_R_vs_aff_$i
#j=$((($j+1)%4))
done
wait
echo "finished"
#python blackbox_attacks.py --load_config shuffle_bb.json --en race_bb_oversample


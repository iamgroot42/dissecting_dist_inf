#!/bin/bash
for (( i = 1001; i <= 1000+$3; i++ )) 
do
    python train_models.py --split $1 --filter Male --ratio $2 --name $i --adv_train --eps 0.031
done
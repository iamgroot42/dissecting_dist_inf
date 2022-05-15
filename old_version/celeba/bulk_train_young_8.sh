#!/bin/bash
for i in {1..500}
do
    python train_models.py --split $1 --filter Young --task Male --ratio $2 --name $i --adv_train --eps 0.031373 --adv_name adv_train_8 --num_workers 16
done

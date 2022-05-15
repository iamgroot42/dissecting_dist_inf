#!/bin/bash
for i in {1..1000}
do
    python train_models.py --split $1 --filter Young --task Male --ratio $2 --name $i --adv_train --eps 0.062745 --adv_name adv_train_16 --num_workers 16
done

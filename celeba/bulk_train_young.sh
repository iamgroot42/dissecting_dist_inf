#!/bin/bash
for i in {1..1000}
do
    python train_models.py --split $1 --filter Young --task Male --ratio $2 --name $i
done
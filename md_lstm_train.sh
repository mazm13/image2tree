#!/usr/bin/env bash
seed=128
MODEL='tree_model_md_att'
export CUDA_VISIBLE_DEVICES=1
ID=${MODEL}_${seed}_rl_sider
python2 -u main.py \
    --id ${ID} \
    --caption_model ${MODEL} \
    --learning_rate 1e-6 \
    --batch_size 16 \
    --start_from tree_model_md_att_125

seed=125
MODEL='tree_model_md'
export CUDA_VISIBLE_DEVICES=0
ID=${MODEL}_${seed}_sider
python2 -u main.py \
    --id ${ID} \
    --caption_model ${MODEL} \
    --learning_rate_decay_start 0 \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --save_checkpoint_every 6000 \
    --seed ${seed} | tee log/log_${ID}


#!/usr/bin/env bash
id="tree_model_md_att_125"
python2 sample.py \
  --caption_model tree_model_md_att \
  --start_from ${id} \
  --raw_image "" \
  --id ${id} \
  --idx ${1}

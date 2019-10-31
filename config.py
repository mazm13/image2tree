from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict

__C = edict()

cfg = __C

# training settings

__C.crop_size = 224
__C.vocab_path = "./data/vocab_4.pkl"
__C.full_tree_path = "./data/kar_train.json"
__C.tiny_tree_path = "./data/small_dataset.json"
__C.cocotalk_fc = "./data/cocotalk_fc"
__C.cocotalk_att = "./data/cocotalk_att"
__C.dataset_coco = "data/dataset_coco.json"
__C.losses_log_every = 25
__C.max_seq_length = 25
__C.load_best_score = 1
__C.num_workers = 0
__C.num_eval_no_improve = 20
__C.max_epoches = 100

# model super-parameters
__C.embed_size = 512
__C.hidden_size = 512
__C.num_layers = 1
__C.fc_feat_size = 2048
__C.att_feat_size = 2048
__C.att_hid_size = 512
__C.att_size = 196

# training setting
__C.learning_rate_decay_start = -1
__C.learning_rate_decay_every = 2
__C.learning_rate_decay_rate = 0.8
__C.grad_clip = 0.1

__C.cider_reward_weight = 1.0
__C.bleu_reward_weight = 0.0

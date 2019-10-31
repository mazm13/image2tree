import torch
import argparse
import pickle
import os
import models
from config import cfg
from build_vocab import Vocabulary
import eval_utils
import eval_seqtree


def main(args):
    cfg.id = args.id
    cfg.caption_model = args.caption_model
    cfg.use_cuda = 0
    cfg.max_seq_length = args.max_seq_length
    cfg.start_from = args.start_from
    cfg.eval_images = args.eval_images

    cfg_dict = vars(cfg)
    for k, v in cfg_dict.items():
        print(k + ': \t' + str(v))
    with open(cfg.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    cfg.vocab_size = len(vocab)
    model = models.setup(cfg)
    model.eval()
    model.cuda()
    with open(os.path.join(cfg.start_from, 'infos_' + cfg.id + '_best.pkl'), 'rb') as f:
        infos = pickle.load(f)
    eval_kwargs = {'eval_split': 'test', 'eval_time': True, 'save_tree_shape': 1, 'eval_mode': 1}
    eval_kwargs.update(vars(cfg))
    # lang_stats = eval_utils.eval_split(model, vocab, eval_kwargs)
    lang_stats = eval_seqtree.eval_split(model, vocab, eval_kwargs)

    for k, v in lang_stats.items():
        print("{}: {}".format(k, v))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--caption_model', type=str, default='tree_model',
                        help="tree_model, att_model")
    parser.add_argument('--max_seq_length', type=int, default=40,
                        help='max sequence length for generated caption including <EOB> token')
    parser.add_argument('--start_from', type=str, required=True,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'infos.pkl'         : configuration;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--eval_images', type=int, default=5000,
                        help="how many images in Karpathy's val/test split should be used for validation?")
    args = parser.parse_args()
    print(args)
    main(args)

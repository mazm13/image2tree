import torch
import argparse
import pickle
from train import train
import models
from config import cfg
from build_vocab import Vocabulary


def main(args):
    cfg.id = args.id
    cfg.dataset = args.dataset
    cfg.caption_model = args.caption_model
    cfg.use_cuda = args.use_cuda
    cfg.seed = args.seed
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    cfg.max_seq_length = args.max_seq_length
    cfg.learning_rate_decay_start = args.learning_rate_decay_start
    cfg.save_checkpoint_every = args.save_checkpoint_every
    cfg.checkpoint_path = args.checkpoint_path
    cfg.start_from = args.start_from
    cfg.use_tiny = args.use_tiny
    cfg.eval_images = args.eval_images

    cfg_dict = vars(cfg)
    for k, v in cfg_dict.items():
        print(k + ': \t' + str(v))

    torch.manual_seed(cfg.seed)
    if cfg.use_cuda:
        torch.cuda.manual_seed(cfg.seed)

    with open(cfg.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    cfg.vocab_size = len(vocab)

    model = models.setup(cfg).cuda()
    # dp_model = torch.nn.DataParallel(model)
    # dp_model.train()
    model.train()
    train(model, vocab, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--dataset', type=str, default='full',
                        help="train on full dataset or just debug on tiny dataset?")
    parser.add_argument('--caption_model', type=str, default='tree_model',
                        help="tree_model, att_model")
    parser.add_argument('--use_cuda', type=int, default=1,
                        help="use cuda?")
    parser.add_argument('--seed', type=int, default=122,
                        help="random seed")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument('--max_seq_length', type=int, default=25,
                        help='max sequence length for generated caption including <EOB> token')
    parser.add_argument('--learning_rate_decay_start', type=int, default=0)
    parser.add_argument('--save_checkpoint_every', type=int, default=10000)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint',
                        help='directory to store checkpointed models')
    parser.add_argument('--start_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'infos.pkl'         : configuration;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--use_tiny', type=int, default=0,
                        help="use tiny dataset which contains 100 samples from COCO dataset?")
    parser.add_argument('--eval_images', type=int, default=-1,
                        help="how many images in Karpathy's val/test split should be used for validation?")
    args = parser.parse_args()
    print(args)
    main(args)
